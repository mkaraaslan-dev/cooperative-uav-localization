"""Real-time AirSim demonstration of the cooperative localization pipeline."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import airsim
import cv2
import numpy as np
import torch
import torchvision

from common import (
    LinearKalmanFilter,
    ResidualMLP,
    compute_ray_cast,
)


def patch_torch_load():
    original_load = torch.load

    def compatible_load(file, map_location=None, **kwargs):
        kwargs["weights_only"] = False
        return original_load(file, map_location=map_location, **kwargs)

    torch.load = compatible_load


def load_yolo(repository, weights, device):
    sys.path.insert(0, str(repository))
    patch_torch_load()
    from models.experimental import attempt_load
    model = attempt_load(str(weights), map_location=device)
    model.eval()
    return model


def run_yolo(
    model,
    image,
    device,
    image_width,
    image_height,
    confidence_threshold,
    iou_threshold,
):
    from utils.general import xywh2xyxy

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_width, image_height))
    tensor = (
        torch.from_numpy(resized)
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        output, _ = model(tensor)
    prediction = output[0]
    scores = prediction[:, 4] * prediction[:, 5]
    selected = scores > confidence_threshold
    if selected.sum() == 0:
        return None
    prediction = prediction[selected]
    scores = scores[selected]
    boxes = xywh2xyxy(prediction[:, :4])
    keep = torchvision.ops.nms(boxes, scores, iou_threshold)
    if len(keep) == 0:
        return None
    best = keep[scores[keep].argmax()]
    x1, y1, x2, y2 = boxes[best].tolist()
    return {
        "u": (x1 + x2) / 2.0 / image_width,
        "v": (y1 + y2) / 2.0 / image_height,
        "bw": (x2 - x1) / image_width,
        "bh": (y2 - y1) / image_height,
        "distance": float(prediction[best, 6].item()),
        "confidence": float(scores[best].item()),
        "box": [int(x1), int(y1), int(x2), int(y2)],
    }


def state_vector(state, key):
    value = state[key]
    return np.array([value["x"], value["y"], value["z"]], dtype=np.float32)


def build_feature(
    detection,
    observer_state,
    reference_state,
    ray_world,
    ray_cast_position,
    kf_position,
    segment_origin,
):
    ray_cast_relative = ray_cast_position - segment_origin
    kf_relative = kf_position - segment_origin
    reference_position = state_vector(reference_state, "position")
    reference_relative = reference_position - segment_origin
    orientation = observer_state["orientation"]
    linear_velocity = observer_state["linear_velocity"]
    angular_velocity = observer_state["angular_velocity"]

    return np.asarray(
        [
            detection["u"],
            detection["v"],
            detection["bw"],
            detection["bh"],
            detection["distance"],
            *ray_world.tolist(),
            *ray_cast_relative.tolist(),
            orientation["w"],
            orientation["x"],
            orientation["y"],
            orientation["z"],
            linear_velocity["x"],
            linear_velocity["y"],
            linear_velocity["z"],
            angular_velocity["x"],
            angular_velocity["y"],
            angular_velocity["z"],
            *kf_relative.tolist(),
            *reference_relative.tolist(),
        ],
        dtype=np.float32,
    )


def get_state(client, vehicle_name):
    kinematics = client.simGetGroundTruthKinematics(
        vehicle_name=vehicle_name
    )
    return {
        "position": {
            "x": kinematics.position.x_val,
            "y": kinematics.position.y_val,
            "z": kinematics.position.z_val,
        },
        "orientation": {
            "w": kinematics.orientation.w_val,
            "x": kinematics.orientation.x_val,
            "y": kinematics.orientation.y_val,
            "z": kinematics.orientation.z_val,
        },
        "linear_velocity": {
            "x": kinematics.linear_velocity.x_val,
            "y": kinematics.linear_velocity.y_val,
            "z": kinematics.linear_velocity.z_val,
        },
        "angular_velocity": {
            "x": kinematics.angular_velocity.x_val,
            "y": kinematics.angular_velocity.y_val,
            "z": kinematics.angular_velocity.z_val,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolov7-repository", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument(
        "--localization-checkpoint", type=Path, required=True
    )
    parser.add_argument("--observer-uav", default="Drone1")
    parser.add_argument("--reference-uav", default="Drone2")
    parser.add_argument("--camera-name", default="0")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--horizontal-fov", type=float, default=90.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.10)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--frame-interval", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = load_yolo(
        args.yolov7_repository, args.yolo_weights, device
    )

    checkpoint = torch.load(
        args.localization_checkpoint,
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint.get(
        "model_state", checkpoint.get("state")
    )
    feature_indices = checkpoint["feat_idx"]
    feature_scaler = checkpoint["sx"]
    target_scaler = checkpoint["sy"]
    model = ResidualMLP(checkpoint["dim"]).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    client = airsim.MultirotorClient()
    client.confirmConnection()
    kalman_filter = LinearKalmanFilter()
    segment_origin = None
    previous_time = None
    frame_index = 0

    print(
        "This script is an AirSim demonstration. Observer-state quantities "
        "are obtained from simGetGroundTruthKinematics."
    )
    print("Press q in the display window to stop.")

    while True:
        start_time = time.perf_counter()
        responses = client.simGetImages(
            [
                airsim.ImageRequest(
                    args.camera_name,
                    airsim.ImageType.Scene,
                    False,
                    False,
                )
            ],
            vehicle_name=args.observer_uav,
        )
        if not responses or responses[0].width == 0:
            time.sleep(0.1)
            continue

        response = responses[0]
        image = np.frombuffer(
            response.image_data_uint8, dtype=np.uint8
        ).reshape(response.height, response.width, 3).copy()

        observer = get_state(client, args.observer_uav)
        reference = get_state(client, args.reference_uav)
        observer_position = state_vector(observer, "position")
        reference_position = state_vector(reference, "position")
        real_distance = float(
            np.linalg.norm(observer_position - reference_position)
        )

        detection = run_yolo(
            yolo,
            image,
            device,
            args.image_width,
            args.image_height,
            args.confidence_threshold,
            args.iou_threshold,
        )

        current_time = time.perf_counter()
        delta_time = (
            args.frame_interval
            if previous_time is None
            else min(max(current_time - previous_time, 0.01), 2.0)
        )
        previous_time = current_time

        ray_cast_position = None
        final_position = None
        if detection is not None:
            orientation = observer["orientation"]
            quaternion = [
                orientation["w"],
                orientation["x"],
                orientation["y"],
                orientation["z"],
            ]
            ray_world, ray_cast_position = compute_ray_cast(
                detection["u"],
                detection["v"],
                detection["distance"],
                quaternion,
                reference_position,
                args.image_width,
                args.image_height,
                args.horizontal_fov,
            )

            if segment_origin is None:
                segment_origin = ray_cast_position.copy()
                kalman_filter.reset(ray_cast_position)
            else:
                kalman_filter.predict(
                    delta_time,
                    state_vector(observer, "linear_velocity"),
                )
                kalman_filter.update(ray_cast_position)

            kf_position = kalman_filter.position()
            full_feature = build_feature(
                detection,
                observer,
                reference,
                ray_world,
                ray_cast_position,
                kf_position,
                segment_origin,
            )
            selected_feature = full_feature[
                feature_indices
            ].reshape(1, -1)
            normalized_feature = feature_scaler.transform(
                selected_feature
            )
            with torch.no_grad():
                normalized_correction = model(
                    torch.as_tensor(
                        normalized_feature,
                        dtype=torch.float32,
                        device=device,
                    )
                ).cpu().numpy()
            correction = target_scaler.inverse_transform(
                normalized_correction
            )[0]
            final_position = ray_cast_position + correction

            x1, y1, x2, y2 = detection["box"]
            cv2.rectangle(
                image, (x1, y1), (x2, y2), (0, 255, 0), 2
            )
            cv2.putText(
                image,
                f"YOLO distance: {detection['distance']:.2f} m",
                (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (0, image.shape[0] - 100),
            (image.shape[1], image.shape[0]),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
        cv2.putText(
            image,
            f"Frame {frame_index} | True distance: {real_distance:.2f} m",
            (10, image.shape[0] - 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            2,
        )
        if ray_cast_position is not None:
            rc_error = np.linalg.norm(
                ray_cast_position - observer_position
            )
            final_error = np.linalg.norm(
                final_position - observer_position
            )
            cv2.putText(
                image,
                f"Ray casting error: {rc_error:.2f} m | "
                f"Final error: {final_error:.2f} m",
                (10, image.shape[0] - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 230, 255),
                2,
            )
        else:
            cv2.putText(
                image,
                "Reference UAV not detected",
                (10, image.shape[0] - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 230, 255),
                2,
            )

        cv2.imshow("Cooperative UAV Localization - AirSim", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_index += 1
        elapsed = time.perf_counter() - start_time
        time.sleep(max(0.0, args.frame_interval - elapsed))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
