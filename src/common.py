"""Shared models, geometry, feature construction, and evaluation utilities."""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


ENVIRONMENTS = ("base", "forest", "snowy", "canyon")

FEATURE_GROUP_INDICES = {
    "yolo": list(range(0, 5)),
    "geometry": list(range(5, 11)),
    # These are simulator-provided observer-state quantities:
    # quaternion, linear velocity, and angular velocity.
    "imu": list(range(11, 21)),
    "kf": list(range(21, 24)),
    "reference": list(range(24, 27)),
}

ABLATION_CONFIGS = {
    "A1_yolo_geometry_reference": ["yolo", "geometry", "reference"],
    "A2_yolo_geometry_imu_reference": ["yolo", "geometry", "imu", "reference"],
    "A3_yolo_geometry_kf_reference": ["yolo", "geometry", "kf", "reference"],
    "A4_yolo_geometry_imu_kf_reference": [
        "yolo", "geometry", "imu", "kf", "reference"
    ],
}


class ResidualMLP(nn.Module):
    """MLP used to estimate the 3D residual correction."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ArrayDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return self.features[index], self.targets[index]


class LinearKalmanFilter:
    """
    Linear constant-velocity Kalman filter.

    State: [px, py, pz, vx, vy, vz].
    The supplied observer velocity is inserted into the velocity state before
    prediction, matching the experimental implementation.
    """

    def __init__(self) -> None:
        self.state = np.zeros(6, dtype=np.float64)
        self.covariance = np.eye(6, dtype=np.float64) * 100.0
        self.process_noise = np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
        self.measurement_noise = np.diag([5.0, 5.0, 5.0])
        self.initialized = False

    def reset(self, initial_position: np.ndarray | None = None) -> None:
        self.state = np.zeros(6, dtype=np.float64)
        if initial_position is not None:
            self.state[:3] = np.asarray(initial_position, dtype=np.float64)
        self.covariance = np.eye(6, dtype=np.float64) * 100.0
        self.initialized = initial_position is not None

    def predict(self, delta_time: float, velocity: np.ndarray) -> None:
        transition = np.eye(6, dtype=np.float64)
        transition[0, 3] = delta_time
        transition[1, 4] = delta_time
        transition[2, 5] = delta_time
        self.state[3:6] = np.asarray(velocity, dtype=np.float64)
        self.state = transition @ self.state
        self.covariance = (
            transition @ self.covariance @ transition.T + self.process_noise
        )

    def update(self, position_measurement: np.ndarray) -> None:
        measurement_matrix = np.zeros((3, 6), dtype=np.float64)
        measurement_matrix[0, 0] = 1.0
        measurement_matrix[1, 1] = 1.0
        measurement_matrix[2, 2] = 1.0

        innovation = (
            np.asarray(position_measurement, dtype=np.float64)
            - measurement_matrix @ self.state
        )
        innovation_covariance = (
            measurement_matrix @ self.covariance @ measurement_matrix.T
            + self.measurement_noise
        )
        gain = (
            self.covariance
            @ measurement_matrix.T
            @ np.linalg.inv(innovation_covariance)
        )
        self.state = self.state + gain @ innovation
        self.covariance = (
            np.eye(6) - gain @ measurement_matrix
        ) @ self.covariance

    def position(self) -> np.ndarray:
        return self.state[:3].copy()


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def quaternion_to_rotation(quaternion: Sequence[float]) -> np.ndarray:
    """Convert [w, x, y, z] quaternion to a 3x3 rotation matrix."""
    w, x, y, z = [float(value) for value in quaternion]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def compute_ray_cast(
    normalized_u: float,
    normalized_v: float,
    distance: float,
    observer_orientation: Sequence[float],
    reference_position: Sequence[float],
    image_width: int = 640,
    image_height: int = 480,
    horizontal_fov_deg: float = 90.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the world-frame viewing ray and observer position estimate.

    AirSim camera convention used by the original experiments:
    camera forward axis = +x, image horizontal axis = +y, image vertical axis = +z.
    """
    focal_length = (image_width / 2.0) / np.tan(
        np.radians(horizontal_fov_deg / 2.0)
    )
    ray_camera = np.array(
        [
            1.0,
            (normalized_u * image_width - image_width / 2.0) / focal_length,
            (normalized_v * image_height - image_height / 2.0) / focal_length,
        ],
        dtype=np.float64,
    )
    ray_camera /= np.linalg.norm(ray_camera)
    ray_world = quaternion_to_rotation(observer_orientation) @ ray_camera
    observer_estimate = (
        np.asarray(reference_position, dtype=np.float64)
        - float(distance) * ray_world
    )
    return ray_world.astype(np.float32), observer_estimate.astype(np.float32)


def resolve_feature_indices(groups: Iterable[str]) -> list[int]:
    indices: list[int] = []
    for group in groups:
        if group not in FEATURE_GROUP_INDICES:
            raise KeyError(f"Unknown feature group: {group}")
        indices.extend(FEATURE_GROUP_INDICES[group])
    return sorted(set(indices))


def npz_get(data, *keys: str):
    """Return the first available key from an NPZ file."""
    for key in keys:
        if key in data.files:
            return data[key]
    raise KeyError(f"None of the requested keys exist: {keys}")


def parse_timestamp(value) -> dt.datetime:
    text = str(value)
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO timestamp: {text}") from exc


def create_segments(
    observer_positions: np.ndarray,
    environment_ids: np.ndarray | None,
    jump_threshold: float,
) -> list[tuple[int, int]]:
    """
    Create contiguous segments. A new segment starts after a large position
    discontinuity or when the environment changes.
    """
    count = len(observer_positions)
    if count < 2:
        return [(0, count)] if count else []

    position_jumps = np.linalg.norm(
        np.diff(observer_positions, axis=0), axis=1
    ) > jump_threshold

    boundaries = set(np.where(position_jumps)[0] + 1)
    if environment_ids is not None:
        env_changes = np.where(np.diff(environment_ids) != 0)[0] + 1
        boundaries.update(env_changes.tolist())

    ordered = [0] + sorted(boundaries) + [count]
    return [
        (ordered[index], ordered[index + 1])
        for index in range(len(ordered) - 1)
        if ordered[index + 1] - ordered[index] >= 2
    ]


@dataclass
class PreparedData:
    features: np.ndarray
    residual_targets: np.ndarray
    ray_cast_positions: np.ndarray
    kf_positions: np.ndarray
    environment_ids: np.ndarray
    segments: list[tuple[int, int]]


def build_feature_dataset(
    yolo_u: np.ndarray,
    yolo_v: np.ndarray,
    yolo_bw: np.ndarray,
    yolo_bh: np.ndarray,
    yolo_distance: np.ndarray,
    observer_position: np.ndarray,
    observer_orientation: np.ndarray,
    observer_velocity: np.ndarray,
    observer_angular_velocity: np.ndarray,
    reference_position: np.ndarray,
    timestamps: np.ndarray,
    environment_ids: np.ndarray | None = None,
    jump_threshold: float = 10.0,
    image_width: int = 640,
    image_height: int = 480,
    horizontal_fov_deg: float = 90.0,
) -> PreparedData:
    count = len(yolo_u)
    if environment_ids is None:
        environment_ids = np.zeros(count, dtype=np.int32)

    segments = create_segments(
        observer_position, environment_ids, jump_threshold
    )
    rays = np.zeros((count, 3), dtype=np.float32)
    ray_cast_positions = np.zeros((count, 3), dtype=np.float32)
    kf_positions = np.zeros((count, 3), dtype=np.float32)

    for segment_start, segment_end in segments:
        kalman_filter = LinearKalmanFilter()
        previous_time: dt.datetime | None = None

        for index in range(segment_start, segment_end):
            ray, position_estimate = compute_ray_cast(
                yolo_u[index],
                yolo_v[index],
                yolo_distance[index],
                observer_orientation[index],
                reference_position[index],
                image_width,
                image_height,
                horizontal_fov_deg,
            )
            rays[index] = ray
            ray_cast_positions[index] = position_estimate

            current_time = parse_timestamp(timestamps[index])
            delta_time = (
                0.5
                if previous_time is None
                else min(
                    max((current_time - previous_time).total_seconds(), 0.01),
                    2.0,
                )
            )
            previous_time = current_time

            if not kalman_filter.initialized:
                kalman_filter.reset(position_estimate)
            else:
                kalman_filter.predict(
                    delta_time, observer_velocity[index]
                )
                kalman_filter.update(position_estimate)
            kf_positions[index] = kalman_filter.position()

    features = np.zeros((count, 27), dtype=np.float32)
    residual_targets = np.zeros((count, 3), dtype=np.float32)

    for segment_start, segment_end in segments:
        segment_origin = ray_cast_positions[segment_start].copy()
        for index in range(segment_start, segment_end):
            ray_cast_relative = ray_cast_positions[index] - segment_origin
            kf_relative = kf_positions[index] - segment_origin
            reference_relative = reference_position[index] - segment_origin
            ground_truth_relative = observer_position[index] - segment_origin

            features[index] = np.asarray(
                [
                    yolo_u[index],
                    yolo_v[index],
                    yolo_bw[index],
                    yolo_bh[index],
                    yolo_distance[index],
                    *rays[index].tolist(),
                    *ray_cast_relative.tolist(),
                    *observer_orientation[index].tolist(),
                    *observer_velocity[index].tolist(),
                    *observer_angular_velocity[index].tolist(),
                    *kf_relative.tolist(),
                    *reference_relative.tolist(),
                ],
                dtype=np.float32,
            )
            residual_targets[index] = (
                ground_truth_relative - ray_cast_relative
            ).astype(np.float32)

    return PreparedData(
        features=features,
        residual_targets=residual_targets,
        ray_cast_positions=ray_cast_positions,
        kf_positions=kf_positions,
        environment_ids=np.asarray(environment_ids, dtype=np.int32),
        segments=segments,
    )


def position_error(
    predictions: np.ndarray,
    targets: np.ndarray,
    metric: str = "rmse3d",
) -> float:
    """
    Compute a 3D localization error.

    rmse3d:
        sqrt(mean(||prediction-target||_2^2)); matches the manuscript equation.
    mean_euclidean:
        mean(||prediction-target||_2); retained for compatibility with earlier runs.
    """
    errors = np.asarray(predictions) - np.asarray(targets)
    squared_norms = np.sum(errors * errors, axis=1)
    if metric == "rmse3d":
        return float(np.sqrt(np.mean(squared_norms)))
    if metric == "mean_euclidean":
        return float(np.mean(np.sqrt(squared_norms)))
    raise ValueError("metric must be 'rmse3d' or 'mean_euclidean'")


def improvement_percent(baseline: float, model_error: float) -> float:
    if baseline <= 0:
        return float("nan")
    return float((baseline - model_error) / baseline * 100.0)


def fit_scalers(
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> tuple[StandardScaler, StandardScaler]:
    feature_scaler = StandardScaler().fit(train_features)
    target_scaler = StandardScaler().fit(train_targets)
    return feature_scaler, target_scaler
