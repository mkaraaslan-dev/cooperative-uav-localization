"""
Train the residual MLP, run the A1-A4 feature ablation, and perform LOEO tests.

Two evaluation modes are provided:
  strict (recommended):
      separate training, validation, and test subsets; the held-out LOEO
      environment is evaluated only after model selection.
  paper_compat:
      preserves the earlier internal experiment behavior for compatibility.

The metric can also be selected:
  rmse3d (recommended and consistent with the manuscript equation)
  mean_euclidean (compatibility with earlier internal scripts)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from common import (
    ABLATION_CONFIGS,
    ENVIRONMENTS,
    ArrayDataset,
    ResidualMLP,
    build_feature_dataset,
    fit_scalers,
    improvement_percent,
    npz_get,
    position_error,
    resolve_feature_indices,
    set_seed,
)


def load_prepared_dataset(path: Path, jump_threshold: float):
    data = np.load(path, allow_pickle=True)
    return build_feature_dataset(
        yolo_u=npz_get(data, "yolo_u"),
        yolo_v=npz_get(data, "yolo_v"),
        yolo_bw=npz_get(data, "yolo_bw"),
        yolo_bh=npz_get(data, "yolo_bh"),
        yolo_distance=npz_get(data, "yolo_distance", "yolo_d"),
        observer_position=npz_get(data, "observer_position", "d1_pos"),
        observer_orientation=npz_get(
            data, "observer_orientation", "d1_ori"
        ),
        observer_velocity=npz_get(
            data, "observer_linear_velocity", "d1_vel"
        ),
        observer_angular_velocity=npz_get(
            data, "observer_angular_velocity", "d1_ang"
        ),
        reference_position=npz_get(
            data, "reference_position", "d2_pos"
        ),
        timestamps=npz_get(data, "timestamps"),
        environment_ids=npz_get(data, "environment_ids", "env_ids"),
        jump_threshold=jump_threshold,
    )


def stratified_split(
    environment_ids: np.ndarray,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
):
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []

    available = [
        np.where(environment_ids == environment_id)[0]
        for environment_id in range(len(ENVIRONMENTS))
    ]
    minimum_count = min(len(indices) for indices in available)

    for indices in available:
        selected = rng.choice(
            indices, size=minimum_count, replace=False
        )
        rng.shuffle(selected)
        train_end = int(minimum_count * train_fraction)
        validation_end = train_end + int(
            minimum_count * validation_fraction
        )
        train_indices.extend(selected[:train_end].tolist())
        validation_indices.extend(
            selected[train_end:validation_end].tolist()
        )
        test_indices.extend(selected[validation_end:].tolist())

    return train_indices, validation_indices, test_indices


def make_loader(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
):
    return DataLoader(
        ArrayDataset(features, targets),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def evaluate_model(
    model,
    loader,
    target_scaler,
    original_targets,
    device,
    metric,
):
    predictions = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            predictions.append(model(inputs.to(device)).cpu().numpy())
    predictions_m = target_scaler.inverse_transform(
        np.vstack(predictions)
    )
    return (
        position_error(predictions_m, original_targets, metric),
        predictions_m,
    )


def train_one_run(
    configuration_name,
    groups,
    run_index,
    prepared,
    train_indices,
    validation_indices,
    test_indices,
    args,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_indices = resolve_feature_indices(groups)

    train_features = prepared.features[train_indices][:, feature_indices]
    train_targets = prepared.residual_targets[train_indices]
    validation_features = prepared.features[validation_indices][
        :, feature_indices
    ]
    validation_targets = prepared.residual_targets[validation_indices]
    test_features = prepared.features[test_indices][:, feature_indices]
    test_targets = prepared.residual_targets[test_indices]

    feature_scaler, target_scaler = fit_scalers(
        train_features, train_targets
    )
    train_x = feature_scaler.transform(train_features)
    train_y = target_scaler.transform(train_targets)
    validation_x = feature_scaler.transform(validation_features)
    validation_y = target_scaler.transform(validation_targets)
    test_x = feature_scaler.transform(test_features)
    test_y = target_scaler.transform(test_targets)

    train_loader = make_loader(
        train_x, train_y, args.batch_size, True, True
    )
    validation_loader = make_loader(
        validation_x, validation_y, args.batch_size, False
    )
    test_loader = make_loader(
        test_x, test_y, args.batch_size, False
    )

    run_seed = args.model_seed + run_index
    set_seed(run_seed, deterministic=args.deterministic)

    model = ResidualMLP(len(feature_indices)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
    )
    criterion = nn.MSELoss()

    checkpoint_dir = (
        args.output_dir / "runs" / configuration_name
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"run{run_index + 1}.pt"

    best_validation_loss = float("inf")
    training_losses = []
    validation_losses = []

    wandb_run = None
    if args.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=f"{configuration_name}_run{run_index + 1}",
            config=vars(args) | {
                "configuration": configuration_name,
                "groups": groups,
            },
            reinit="finish_previous",
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        training_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip
            )
            optimizer.step()
            training_loss += loss.item()
        training_loss /= max(len(train_loader), 1)

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                prediction = model(inputs.to(device))
                validation_loss += criterion(
                    prediction, targets.to(device)
                ).item()
        validation_loss /= max(len(validation_loader), 1)
        scheduler.step(validation_loss)

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "sx": feature_scaler,
                    "sy": target_scaler,
                    "feat_idx": feature_indices,
                    "dim": len(feature_indices),
                    "groups": groups,
                    "configuration": configuration_name,
                    "metric": args.metric,
                    "run_seed": run_seed,
                },
                checkpoint_path,
            )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/loss": training_loss,
                    "validation/loss": validation_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])

    test_error, _ = evaluate_model(
        model,
        test_loader,
        target_scaler,
        test_targets,
        device,
        args.metric,
    )

    per_environment = {}
    test_environment_ids = prepared.environment_ids[test_indices]
    for environment_id, environment in enumerate(ENVIRONMENTS):
        mask = test_environment_ids == environment_id
        if not np.any(mask):
            continue
        subset_x = test_x[mask]
        subset_y = test_y[mask]
        subset_targets = test_targets[mask]
        subset_loader = make_loader(
            subset_x, subset_y, args.batch_size, False
        )
        subset_error, _ = evaluate_model(
            model,
            subset_loader,
            target_scaler,
            subset_targets,
            device,
            args.metric,
        )
        per_environment[environment] = subset_error

    if wandb_run is not None:
        wandb_run.log({"final/test_error": test_error})
        wandb_run.finish()

    return {
        "error": test_error,
        "per_environment": per_environment,
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "checkpoint": str(checkpoint_path),
    }


def run_feature_ablation(prepared, split, args):
    train_indices, validation_indices, test_indices = split
    baseline = position_error(
        np.zeros_like(prepared.residual_targets[test_indices]),
        prepared.residual_targets[test_indices],
        args.metric,
    )

    results = {}
    curve_dir = args.output_dir / "run_data"
    curve_dir.mkdir(parents=True, exist_ok=True)

    for configuration_name, groups in ABLATION_CONFIGS.items():
        runs = []
        for run_index in range(args.runs):
            result = train_one_run(
                configuration_name,
                groups,
                run_index,
                prepared,
                train_indices,
                validation_indices,
                test_indices,
                args,
            )
            runs.append(result)
            print(
                f"{configuration_name} run {run_index + 1}: "
                f"{result['error']:.4f} m"
            )

        errors = np.asarray(
            [run["error"] for run in runs], dtype=np.float64
        )
        results[configuration_name] = {
            "groups": groups,
            "dimension": len(resolve_feature_indices(groups)),
            "mean": float(errors.mean()),
            "std": float(errors.std()),
            "values": errors.tolist(),
            "improvement_percent": improvement_percent(
                baseline, float(errors.mean())
            ),
            "per_environment": {
                environment: {
                    "mean": float(
                        np.mean(
                            [
                                run["per_environment"][environment]
                                for run in runs
                                if environment in run["per_environment"]
                            ]
                        )
                    ),
                    "std": float(
                        np.std(
                            [
                                run["per_environment"][environment]
                                for run in runs
                                if environment in run["per_environment"]
                            ]
                        )
                    ),
                }
                for environment in ENVIRONMENTS
                if any(
                    environment in run["per_environment"]
                    for run in runs
                )
            },
        }

        np.savez(
            curve_dir / f"{configuration_name}_curves.npz",
            tr_losses=np.asarray(
                [run["training_losses"] for run in runs]
            ),
            te_losses=np.asarray(
                [run["validation_losses"] for run in runs]
            ),
            error_values=errors,
        )

    results["_baseline"] = {
        "error": baseline,
        "metric": args.metric,
    }
    return results


def run_loeo(prepared, best_configuration, args):
    groups = ABLATION_CONFIGS[best_configuration]
    feature_indices = resolve_feature_indices(groups)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    curve_dir = args.output_dir / "run_data"
    curve_dir.mkdir(parents=True, exist_ok=True)

    for held_out_id, held_out_environment in enumerate(ENVIRONMENTS):
        held_out_indices = np.where(
            prepared.environment_ids == held_out_id
        )[0]
        remaining_indices = np.where(
            prepared.environment_ids != held_out_id
        )[0]

        rng = np.random.default_rng(args.split_seed)
        rng.shuffle(remaining_indices)
        validation_count = max(
            1, int(len(remaining_indices) * args.loeo_validation_fraction)
        )
        validation_indices = remaining_indices[:validation_count].tolist()
        train_indices = remaining_indices[validation_count:].tolist()
        test_indices = held_out_indices.tolist()

        # Compatibility mode reproduces the earlier use of the held-out
        # environment for scheduler/checkpoint selection.
        if args.evaluation_mode == "paper_compat":
            validation_indices = test_indices

        single_args = argparse.Namespace(**vars(args))
        single_args.runs = 1
        result = train_one_run(
            f"loeo_{held_out_environment}",
            groups,
            0,
            prepared,
            train_indices,
            validation_indices,
            test_indices,
            single_args,
        )

        baseline = position_error(
            np.zeros_like(prepared.residual_targets[test_indices]),
            prepared.residual_targets[test_indices],
            args.metric,
        )
        results[held_out_environment] = {
            "baseline": baseline,
            "model_error": result["error"],
            "improvement_percent": improvement_percent(
                baseline, result["error"]
            ),
            "number_of_test_samples": len(test_indices),
        }
        np.savez(
            curve_dir / f"cross_{held_out_environment}_curves.npz",
            tr_losses=np.asarray(result["training_losses"]),
            te_losses=np.asarray(result["validation_losses"]),
        )
        print(
            f"LOEO {held_out_environment}: "
            f"baseline={baseline:.4f} m, "
            f"model={result['error']:.4f} m"
        )

    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--scheduler-patience", type=int, default=20)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=1000)
    parser.add_argument("--jump-threshold", type=float, default=10.0)
    parser.add_argument(
        "--metric",
        choices=["rmse3d", "mean_euclidean"],
        default="rmse3d",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=["strict", "paper_compat"],
        default="strict",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument(
        "--loeo-validation-fraction", type=float, default=0.10
    )
    parser.add_argument("--skip-loeo", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cooperative-uav-localization")
    parser.add_argument("--wandb-entity", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = load_prepared_dataset(
        args.dataset, args.jump_threshold
    )

    if args.evaluation_mode == "paper_compat":
        train_fraction = 0.80
        validation_fraction = 0.0
        train_indices, _, test_indices = stratified_split(
            prepared.environment_ids,
            args.split_seed,
            train_fraction,
            validation_fraction,
        )
        validation_indices = test_indices
        split = train_indices, validation_indices, test_indices
    else:
        split = stratified_split(
            prepared.environment_ids,
            args.split_seed,
            args.train_fraction,
            args.validation_fraction,
        )

    ablation_results = run_feature_ablation(
        prepared, split, args
    )
    valid_results = {
        key: value
        for key, value in ablation_results.items()
        if not key.startswith("_")
    }
    best_configuration = min(
        valid_results,
        key=lambda key: valid_results[key]["mean"],
    )

    loeo_results = {}
    if not args.skip_loeo:
        loeo_results = run_loeo(
            prepared, best_configuration, args
        )

    summary = {
        "evaluation_mode": args.evaluation_mode,
        "metric": args.metric,
        "best_configuration": best_configuration,
        "ablation": ablation_results,
        "loeo": loeo_results,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Results saved to {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
