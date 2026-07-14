"""Build a GT NPZ dataset from YOLO-format labels and AirSim metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import ENVIRONMENTS


def vector3(mapping: dict) -> list[float]:
    return [mapping["x"], mapping["y"], mapping["z"]]


def quaternion(mapping: dict) -> list[float]:
    return [mapping["w"], mapping["x"], mapping["y"], mapping["z"]]


def discover_label_files(dataset_root: Path):
    flat_labels = dataset_root / "labels"
    if flat_labels.exists():
        for path in sorted(flat_labels.glob("*.txt")):
            yield None, path
        return

    for environment in ENVIRONMENTS:
        labels_dir = dataset_root / environment / "labels"
        for path in sorted(labels_dir.glob("*.txt")):
            yield environment, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observer-uav", default="Drone1")
    parser.add_argument("--reference-uav", default="Drone2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = {
        "gt_u": [],
        "gt_v": [],
        "gt_bw": [],
        "gt_bh": [],
        "gt_distance": [],
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
    skipped = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for discovered_environment, label_path in discover_label_files(
        args.dataset_root
    ):
        stem = label_path.stem
        if discovered_environment is None:
            metadata_path = args.dataset_root / "metadata" / f"{stem}.json"
        else:
            metadata_path = (
                args.dataset_root
                / discovered_environment
                / "metadata"
                / f"{stem}.json"
            )

        if not metadata_path.exists():
            skip("missing_metadata")
            continue

        lines = [
            line.strip()
            for line in label_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        if not lines:
            skip("empty_label")
            continue

        fields = lines[0].split()
        if len(fields) < 6:
            skip("invalid_label")
            continue

        try:
            _, center_x, center_y, width, height, distance = fields[:6]
            center_x = float(center_x)
            center_y = float(center_y)
            width = float(width)
            height = float(height)
            distance = float(distance)
        except ValueError:
            skip("invalid_numeric_label")
            continue

        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        timestamp = metadata.get("timestamp")
        if not timestamp:
            skip("missing_timestamp")
            continue

        try:
            observer = metadata["states"][args.observer_uav]
            reference = metadata["states"][args.reference_uav]
        except KeyError:
            skip("missing_uav_state")
            continue

        environment = (
            discovered_environment
            or metadata.get("environment")
            or "unknown"
        )
        environment_id = (
            ENVIRONMENTS.index(environment)
            if environment in ENVIRONMENTS
            else -1
        )

        records["gt_u"].append(center_x)
        records["gt_v"].append(center_y)
        records["gt_bw"].append(width)
        records["gt_bh"].append(height)
        records["gt_distance"].append(distance)
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
        records["timestamps"].append(timestamp)
        records["frame_ids"].append(int(stem.split("_")[-1]))
        records["environment_ids"].append(environment_id)
        records["environment_names"].append(environment)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        gt_u=np.asarray(records["gt_u"], dtype=np.float32),
        gt_v=np.asarray(records["gt_v"], dtype=np.float32),
        gt_bw=np.asarray(records["gt_bw"], dtype=np.float32),
        gt_bh=np.asarray(records["gt_bh"], dtype=np.float32),
        gt_distance=np.asarray(
            records["gt_distance"], dtype=np.float32
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
    print(f"Saved GT dataset with {len(records['gt_u'])} samples.")
    if skipped:
        print(f"Skipped samples: {json.dumps(skipped, indent=2)}")


if __name__ == "__main__":
    main()
