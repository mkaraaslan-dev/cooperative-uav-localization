"""Run the external YOLOv7 ranging model and build the localization NPZ file."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from common import ENVIRONMENTS


def patch_torch_load() -> None:
    original_load = torch.load

    def compatible_load(file, map_location=None, **kwargs):
        kwargs["weights_only"] = False
        return original_load(file, map_location=map_location, **kwargs)

    torch.load = compatible_load


def load_yolo(repository: Path, weights: Path, device: torch.device):
    sys.path.insert(0, str(repository))
    patch_torch_load()
    from models.experimental import attempt_load

    model = attempt_load(str(weights), map_location=device)
    model.eval()
    return model


def run_yolo(
    model,
    image_bgr: np.ndarray,
    device: torch.device,
    image_width: int,
    image_height: int,
    confidence_threshold: float,
    iou_threshold: float,
):
    from utils.general import non_max_suppression

    image = cv2.cvtColor(
        cv2.resize(image_bgr, (image_width, image_height)),
        cv2.COLOR_BGR2RGB,
    )
    tensor = (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        output, _ = model(tensor)

    detections = non_max_suppression(
        output, confidence_threshold, iou_threshold
    )[0]
    if detections is None or len(detections) == 0:
        return None

    best_index = detections[:, 4].argmax().item()
    x1, y1, x2, y2, confidence, class_id, distance = (
        detections[best_index].tolist()
    )
    return {
        "u": (x1 + x2) / 2.0 / image_width,
        "v": (y1 + y2) / 2.0 / image_height,
        "bw": (x2 - x1) / image_width,
        "bh": (y2 - y1) / image_height,
        "distance": float(distance),
        "confidence": float(confidence),
        "box": [int(x1), int(y1), int(x2), int(y2)],
    }


def vector3(mapping: dict) -> list[float]:
    return [mapping["x"], mapping["y"], mapping["z"]]


def quaternion(mapping: dict) -> list[float]:
    return [mapping["w"], mapping["x"], mapping["y"], mapping["z"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument("--yolov7-repository", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observer-uav", default="Drone1")
    parser.add_argument("--reference-uav", default="Drone2")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--confidence-threshold", type=float, default=0.10)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--max-per-environment", type=int, default=1200)
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--preview-per-environment", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_yolo(args.yolov7_repository, args.weights, device)

    records = {
        "yolo_u": [],
        "yolo_v": [],
        "yolo_bw": [],
        "yolo_bh": [],
        "yolo_distance": [],
        "yolo_confidence": [],
        "observer_position": [],
        "observer_orientation": [],
        "observer_linear_velocity": [],
        "observer_angular_velocity": [],
        "reference_position": [],
        "timestamps": [],
        "frame_ids": [],
        "environment_ids": [],
        "environment_names": [],
    }

    environment_statistics = {}
    for environment_id, environment in enumerate(ENVIRONMENTS):
        image_dir = args.raw_dataset / environment / "images"
        metadata_dir = args.raw_dataset / environment / "metadata"
        metadata_files = sorted(metadata_dir.glob("*.json"))
        successful = 0
        missed = 0

        if args.preview_dir:
            (args.preview_dir / environment).mkdir(
                parents=True, exist_ok=True
            )

        for metadata_path in metadata_files:
            if successful >= args.max_per_environment:
                break

            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            frame_id = int(metadata["frame_id"])
            image_path = image_dir / f"frame_{frame_id:06d}.png"
            image = cv2.imread(str(image_path))
            if image is None:
                missed += 1
                continue

            detection = run_yolo(
                model,
                image,
                device,
                args.image_width,
                args.image_height,
                args.confidence_threshold,
                args.iou_threshold,
            )
            if detection is None:
                missed += 1
                continue

            observer = metadata["states"][args.observer_uav]
            reference = metadata["states"][args.reference_uav]

            records["yolo_u"].append(detection["u"])
            records["yolo_v"].append(detection["v"])
            records["yolo_bw"].append(detection["bw"])
            records["yolo_bh"].append(detection["bh"])
            records["yolo_distance"].append(detection["distance"])
            records["yolo_confidence"].append(
                detection["confidence"]
            )
            records["observer_position"].append(
                vector3(observer["position"])
            )
            records["observer_orientation"].append(
                quaternion(observer["orientation"])
            )
            records["observer_linear_velocity"].append(
                vector3(observer["linear_velocity"])
            )
            records["observer_angular_velocity"].append(
                vector3(observer["angular_velocity"])
            )
            records["reference_position"].append(
                vector3(reference["position"])
            )
            records["timestamps"].append(metadata["timestamp"])
            records["frame_ids"].append(frame_id)
            records["environment_ids"].append(environment_id)
            records["environment_names"].append(environment)

            if (
                args.preview_dir
                and successful < args.preview_per_environment
            ):
                x1, y1, x2, y2 = detection["box"]
                preview = image.copy()
                cv2.rectangle(
                    preview, (x1, y1), (x2, y2), (0, 255, 0), 2
                )
                cv2.putText(
                    preview,
                    f"{detection['distance']:.2f} m | "
                    f"conf={detection['confidence']:.2f}",
                    (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                cv2.imwrite(
                    str(
                        args.preview_dir
                        / environment
                        / f"{successful:03d}_{frame_id:06d}.png"
                    ),
                    preview,
                )

            successful += 1

        environment_statistics[environment] = {
            "selected": successful,
            "missed_or_missing": missed,
        }
        print(
            f"{environment}: selected={successful}, "
            f"missed_or_missing={missed}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        yolo_u=np.asarray(records["yolo_u"], dtype=np.float32),
        yolo_v=np.asarray(records["yolo_v"], dtype=np.float32),
        yolo_bw=np.asarray(records["yolo_bw"], dtype=np.float32),
        yolo_bh=np.asarray(records["yolo_bh"], dtype=np.float32),
        yolo_distance=np.asarray(
            records["yolo_distance"], dtype=np.float32
        ),
        yolo_confidence=np.asarray(
            records["yolo_confidence"], dtype=np.float32
        ),
        observer_position=np.asarray(
            records["observer_position"], dtype=np.float32
        ),
        observer_orientation=np.asarray(
            records["observer_orientation"], dtype=np.float32
        ),
        observer_linear_velocity=np.asarray(
            records["observer_linear_velocity"], dtype=np.float32
        ),
        observer_angular_velocity=np.asarray(
            records["observer_angular_velocity"], dtype=np.float32
        ),
        reference_position=np.asarray(
            records["reference_position"], dtype=np.float32
        ),
        timestamps=np.asarray(records["timestamps"]),
        frame_ids=np.asarray(records["frame_ids"], dtype=np.int32),
        environment_ids=np.asarray(
            records["environment_ids"], dtype=np.int32
        ),
        environment_names=np.asarray(records["environment_names"]),
    )
    print(f"Saved localization dataset: {args.output}")
    print(json.dumps(environment_statistics, indent=2))


if __name__ == "__main__":
    main()
