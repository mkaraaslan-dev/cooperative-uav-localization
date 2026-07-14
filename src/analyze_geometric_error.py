"""Controlled analysis of ray-casting error using GT and optional YOLO inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (
    ArrayDataset,
    ResidualMLP,
    build_feature_dataset,
    fit_scalers,
    npz_get,
    position_error,
    resolve_feature_indices,
    set_seed,
)


def align_by_frame_id(gt_data, yolo_data):
    gt_ids = npz_get(gt_data, "frame_ids")
    yolo_ids = npz_get(yolo_data, "frame_ids")
    yolo_lookup = {
        int(frame_id): index
        for index, frame_id in enumerate(yolo_ids)
    }
    gt_indices = []
    yolo_indices = []
    for gt_index, frame_id in enumerate(gt_ids):
        yolo_index = yolo_lookup.get(int(frame_id))
        if yolo_index is not None:
            gt_indices.append(gt_index)
            yolo_indices.append(yolo_index)
    return np.asarray(gt_indices), np.asarray(yolo_indices)


def make_loader(features, targets, batch_size, shuffle):
    return DataLoader(
        ArrayDataset(features, targets),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
    )


def run_experiment(name, source, args):
    prepared = build_feature_dataset(**source)
    indices = np.arange(len(prepared.features))
    split = int(len(indices) * 0.8)
    train_indices = indices[:split]
    test_indices = indices[split:]

    feature_groups = [
        group.strip()
        for group in args.feature_groups.split(",")
        if group.strip()
    ]
    feature_indices = resolve_feature_indices(feature_groups)

    baseline = position_error(
        np.zeros_like(prepared.residual_targets[test_indices]),
        prepared.residual_targets[test_indices],
        args.metric,
    )

    train_features = prepared.features[train_indices][
        :, feature_indices
    ]
    train_targets = prepared.residual_targets[train_indices]
    test_features = prepared.features[test_indices][
        :, feature_indices
    ]
    test_targets = prepared.residual_targets[test_indices]

    feature_scaler, target_scaler = fit_scalers(
        train_features, train_targets
    )
    train_x = feature_scaler.transform(train_features)
    train_y = target_scaler.transform(train_targets)
    test_x = feature_scaler.transform(test_features)
    test_y = target_scaler.transform(test_targets)

    train_loader = make_loader(
        train_x, train_y, args.batch_size, True
    )
    test_loader = make_loader(
        test_x, test_y, args.batch_size, False
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualMLP(len(feature_indices)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5
    )
    criterion = nn.MSELoss()
    best_loss = float("inf")
    output_path = args.output_dir / f"{name}.pt"

    for _ in range(args.epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                test_loss += criterion(
                    model(inputs.to(device)), targets.to(device)
                ).item()
        test_loss /= max(len(test_loader), 1)
        scheduler.step(test_loss)
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), output_path)

    model.load_state_dict(
        torch.load(output_path, map_location=device, weights_only=True)
    )
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            predictions.append(model(inputs.to(device)).cpu().numpy())
    predictions = target_scaler.inverse_transform(
        np.vstack(predictions)
    )
    model_error = position_error(
        predictions, test_targets, args.metric
    )

    return {
        "condition": name,
        "baseline": baseline,
        "after_residual_correction": model_error,
        "absolute_improvement": baseline - model_error,
        "metric": args.metric,
        "number_of_test_samples": len(test_indices),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dataset", type=Path, required=True)
    parser.add_argument("--yolo-dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        default="gt",
        help="Comma-separated subset of: gt,mix,yolo",
    )
    parser.add_argument(
        "--feature-groups",
        default="yolo,geometry,kf,reference",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jump-threshold", type=float, default=10.0)
    parser.add_argument(
        "--metric",
        choices=["rmse3d", "mean_euclidean"],
        default="rmse3d",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt = np.load(args.gt_dataset, allow_pickle=True)
    conditions = [
        condition.strip()
        for condition in args.conditions.split(",")
        if condition.strip()
    ]

    common_gt = {
        "observer_position": npz_get(
            gt, "observer_position", "d1_pos"
        ),
        "observer_orientation": npz_get(
            gt, "observer_orientation", "d1_ori"
        ),
        "observer_velocity": npz_get(
            gt, "observer_linear_velocity", "d1_vel"
        ),
        "observer_angular_velocity": npz_get(
            gt, "observer_angular_velocity", "d1_ang"
        ),
        "reference_position": npz_get(
            gt, "reference_position", "d2_pos"
        ),
        "timestamps": npz_get(gt, "timestamps"),
        "environment_ids": (
            npz_get(gt, "environment_ids")
            if "environment_ids" in gt.files
            else np.zeros(len(npz_get(gt, "gt_u")), dtype=np.int32)
        ),
        "jump_threshold": args.jump_threshold,
    }

    experiments = {}
    experiments["gt"] = common_gt | {
        "yolo_u": npz_get(gt, "gt_u"),
        "yolo_v": npz_get(gt, "gt_v"),
        "yolo_bw": npz_get(gt, "gt_bw"),
        "yolo_bh": npz_get(gt, "gt_bh"),
        "yolo_distance": npz_get(gt, "gt_distance", "gt_dist"),
    }

    if any(condition in {"mix", "yolo"} for condition in conditions):
        if args.yolo_dataset is None:
            raise ValueError(
                "--yolo-dataset is required for mix or yolo conditions."
            )
        yolo = np.load(args.yolo_dataset, allow_pickle=True)
        gt_indices, yolo_indices = align_by_frame_id(gt, yolo)
        aligned_common = {
            key: (
                value[gt_indices]
                if isinstance(value, np.ndarray)
                and len(value) == len(npz_get(gt, "gt_u"))
                else value
            )
            for key, value in common_gt.items()
        }
        yolo_fields = {
            "yolo_u": npz_get(yolo, "yolo_u")[yolo_indices],
            "yolo_v": npz_get(yolo, "yolo_v")[yolo_indices],
            "yolo_bw": npz_get(yolo, "yolo_bw")[yolo_indices],
            "yolo_bh": npz_get(yolo, "yolo_bh")[yolo_indices],
            "yolo_distance": npz_get(
                yolo, "yolo_distance", "yolo_d"
            )[yolo_indices],
        }
        experiments["yolo"] = aligned_common | yolo_fields
        experiments["mix"] = aligned_common | (
            yolo_fields
            | {
                "yolo_distance": npz_get(
                    gt, "gt_distance", "gt_dist"
                )[gt_indices]
            }
        )

    results = [
        run_experiment(condition, experiments[condition], args)
        for condition in conditions
    ]
    (args.output_dir / "geometric_error_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
