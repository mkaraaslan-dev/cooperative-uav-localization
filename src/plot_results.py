"""Generate publication-ready ablation and LOEO loss figures from saved NPZ files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ABLATION_CONFIGS, ENVIRONMENTS


DISPLAY_LABELS = {
    "A1_yolo_geometry_reference": "A1: YOLO + Geometry + Reference",
    "A2_yolo_geometry_imu_reference": "A2: YOLO + Geometry + IMU + Reference",
    "A3_yolo_geometry_kf_reference": "A3: YOLO + Geometry + KF + Reference",
    "A4_yolo_geometry_imu_kf_reference": (
        "A4: YOLO + Geometry + IMU + KF + Reference"
    ),
}

ENVIRONMENT_LABELS = {
    "base": "Open Area",
    "forest": "Forest",
    "snowy": "Snowy Environment",
    "canyon": "Canyon",
}

COLORS = ["#E53935", "#1E88E5", "#43A047", "#FB8C00"]


def load_curves(path: Path):
    if not path.exists():
        print(f"[WARNING] File not found: {path}")
        return None
    return np.load(path)


def plot_ablation(data_dir: Path, output_dir: Path):
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    figure.suptitle(
        "Feature Ablation — Training and Validation MSE Loss",
        fontsize=14,
        fontweight="bold",
    )
    for axis, (configuration, color) in zip(
        axes.flat, zip(ABLATION_CONFIGS, COLORS)
    ):
        data = load_curves(
            data_dir / f"{configuration}_curves.npz"
        )
        if data is None:
            axis.set_visible(False)
            continue
        training = data["tr_losses"]
        validation = data["te_losses"]
        if training.ndim == 1:
            training = training[None, :]
            validation = validation[None, :]
        epochs = np.arange(1, training.shape[1] + 1)
        training_mean = training.mean(axis=0)
        validation_mean = validation.mean(axis=0)
        validation_std = validation.std(axis=0)

        axis.plot(
            epochs,
            training_mean,
            linestyle="--",
            alpha=0.55,
            color=color,
            label="Mean Training MSE",
        )
        axis.fill_between(
            epochs,
            validation_mean - validation_std,
            validation_mean + validation_std,
            alpha=0.15,
            color=color,
            label="Validation ±1 SD",
        )
        axis.plot(
            epochs,
            validation_mean,
            linewidth=2.2,
            color=color,
            label="Mean Validation MSE",
        )
        axis.set_title(DISPLAY_LABELS[configuration])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("MSE Loss")
        axis.grid(alpha=0.25, linestyle="--")
        axis.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(
        output_dir / "ablation_train_validation_loss.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_loeo(data_dir: Path, output_dir: Path):
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    figure.suptitle(
        "Cross-Environment Generalization — MSE Loss Curves",
        fontsize=14,
        fontweight="bold",
    )
    for axis, environment, color in zip(
        axes.flat, ENVIRONMENTS, COLORS
    ):
        data = load_curves(
            data_dir / f"cross_{environment}_curves.npz"
        )
        if data is None:
            axis.set_visible(False)
            continue
        training = data["tr_losses"]
        validation = data["te_losses"]
        epochs = np.arange(1, len(training) + 1)
        axis.plot(
            epochs,
            training,
            linestyle="--",
            alpha=0.55,
            color=color,
            label="Training MSE",
        )
        axis.plot(
            epochs,
            validation,
            linewidth=2.2,
            color=color,
            label="Validation MSE",
        )
        axis.set_title(
            f"Test Environment: {ENVIRONMENT_LABELS[environment]}"
        )
        axis.set_xlabel("Epoch")
        axis.set_ylabel("MSE Loss")
        axis.grid(alpha=0.25, linestyle="--")
        axis.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(
        output_dir / "cross_environment_loss.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot_ablation(args.data_dir, args.output_dir)
    plot_loeo(args.data_dir, args.output_dir)
    print(f"Figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
