"""Collect environment-labeled images and UAV states from AirSim."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import airsim
import cv2
import numpy as np

from common import ENVIRONMENTS


class AirSimDataCollector:
    def __init__(
        self,
        output_root: Path,
        environment: str,
        observer_uav: str,
        reference_uav: str,
        camera_name: str,
        horizontal_fov_deg: float,
    ) -> None:
        if environment not in ENVIRONMENTS:
            raise ValueError(
                f"Unknown environment '{environment}'. "
                f"Expected one of {ENVIRONMENTS}."
            )

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self.environment = environment
        self.observer_uav = observer_uav
        self.reference_uav = reference_uav
        self.camera_name = camera_name
        self.horizontal_fov_deg = horizontal_fov_deg

        self.environment_dir = output_root / environment
        self.images_dir = self.environment_dir / "images"
        self.metadata_dir = self.environment_dir / "metadata"
        self.index_path = self.environment_dir / "index.json"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.current_index, self.total_collected = self._load_index()

    def _load_index(self) -> tuple[int, int]:
        if not self.index_path.exists():
            return 0, 0
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        return int(data["next_index"]), int(data["total_collected"])

    def _save_index(self) -> None:
        payload = {
            "next_index": self.current_index,
            "total_collected": self.total_collected,
            "last_updated": datetime.now().isoformat(),
            "environment": self.environment,
            "observer_uav": self.observer_uav,
            "reference_uav": self.reference_uav,
        }
        self.index_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def _get_state(self, vehicle_name: str) -> dict:
        kinematics = self.client.simGetGroundTruthKinematics(
            vehicle_name=vehicle_name
        )
        state = {
            "name": vehicle_name,
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
        try:
            gps = self.client.getGpsData(vehicle_name=vehicle_name)
            state["gps"] = {
                "latitude": gps.gnss.geo_point.latitude,
                "longitude": gps.gnss.geo_point.longitude,
                "altitude": gps.gnss.geo_point.altitude,
            }
        except Exception:
            state["gps"] = None
        return state

    def _get_image(self) -> np.ndarray:
        responses = self.client.simGetImages(
            [
                airsim.ImageRequest(
                    self.camera_name,
                    airsim.ImageType.Scene,
                    False,
                    False,
                )
            ],
            vehicle_name=self.observer_uav,
        )
        if not responses or responses[0].width == 0:
            raise RuntimeError("AirSim returned an invalid image.")
        response = responses[0]
        image = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        return image.reshape(response.height, response.width, 3).copy()

    def collect_frame(self) -> bool:
        try:
            image = self._get_image()
            height, width = image.shape[:2]
            states = {
                self.observer_uav: self._get_state(self.observer_uav),
                self.reference_uav: self._get_state(self.reference_uav),
            }

            stem = f"frame_{self.current_index:06d}"
            cv2.imwrite(str(self.images_dir / f"{stem}.png"), image)

            metadata = {
                "frame_id": self.current_index,
                "timestamp": datetime.now().isoformat(),
                "environment": self.environment,
                "image_size": {"width": width, "height": height},
                "camera_params": {
                    "camera_name": self.camera_name,
                    "horizontal_fov_deg": self.horizontal_fov_deg,
                    "image_width": width,
                    "image_height": height,
                },
                "states": states,
            }
            (self.metadata_dir / f"{stem}.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )

            self.current_index += 1
            self.total_collected += 1
            if self.current_index % 10 == 0:
                self._save_index()
            return True
        except Exception as exc:
            print(f"[ERROR] Frame collection failed: {exc}")
            return False

    def collect_session(self, number_of_frames: int, interval: float) -> None:
        print(
            f"Collecting {number_of_frames} frames from '{self.environment}' "
            f"at {interval:.3f} s intervals."
        )
        successful = 0
        start_time = time.time()
        try:
            for _ in range(number_of_frames):
                if self.collect_frame():
                    successful += 1
                    if successful % 50 == 0:
                        elapsed = max(time.time() - start_time, 1e-9)
                        print(
                            f"  {successful}/{number_of_frames} frames | "
                            f"{successful / elapsed:.2f} frames/s"
                        )
                time.sleep(interval)
        except KeyboardInterrupt:
            print("Collection stopped by the user.")
        finally:
            self._save_index()
            print(
                f"Session completed. Collected {successful} frames. "
                f"Environment total: {self.total_collected}."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--environment", choices=ENVIRONMENTS, required=True
    )
    parser.add_argument("--observer-uav", default="Drone1")
    parser.add_argument("--reference-uav", default="Drone2")
    parser.add_argument("--camera-name", default="0")
    parser.add_argument("--horizontal-fov", type=float, default=90.0)
    parser.add_argument("--num-frames", type=int, default=1500)
    parser.add_argument("--interval", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collector = AirSimDataCollector(
        args.output_root,
        args.environment,
        args.observer_uav,
        args.reference_uav,
        args.camera_name,
        args.horizontal_fov,
    )
    collector.collect_session(args.num_frames, args.interval)


if __name__ == "__main__":
    main()
