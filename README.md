# Cooperative UAV Localization — Cleaned Code Package

This package contains the cleaned and portable versions of the scripts used in
the cooperative UAV localization study.

## Included scripts

- `src/collect_airsim_data.py`  
  Collects RGB images and simulator-provided UAV states from AirSim. Data are
  stored separately for `base`, `forest`, `snowy`, and `canyon`.

- `src/build_localization_dataset.py`  
  Runs the external YOLOv7 detection-and-ranging implementation and creates the
  localization NPZ dataset.

- `src/build_ground_truth_dataset.py`  
  Converts YOLO-format ground-truth labels and AirSim metadata into a GT NPZ
  file for the controlled geometric-error experiment.

- `src/train_and_evaluate.py`  
  Builds the 27-dimensional feature representation, performs the A1–A4 feature
  ablation, and runs leave-one-environment-out evaluation.

- `src/analyze_geometric_error.py`  
  Compares GT, mixed, and YOLO measurement conditions for controlled analysis
  of ray-casting and residual-correction errors.

- `src/realtime_airsim_demo.py`  
  Runs the complete pipeline online in AirSim. This is a simulation
  demonstration, not a ready-to-deploy physical-UAV implementation.

- `src/plot_results.py`  
  Generates English ablation and cross-environment loss figures.

## External YOLOv7 dependency

The customized YOLOv7 detection-and-ranging repository is not included in this
package. The GitHub repository URL and the trained YOLOv7 checkpoint should be
provided separately.

## Dataset links

The visual-ranging dataset and cooperative-localization dataset are not
included in this ZIP. Google Drive links can be added to the final repository
README.

## Important evaluation options

`train_and_evaluate.py` provides two modes:

- `--evaluation-mode strict`  
  Recommended. Uses separate training, validation, and test subsets. In LOEO,
  the held-out environment is used only for final testing.

- `--evaluation-mode paper_compat`  
  Retained only for compatibility with earlier internal experiment runs.

The script also provides two position metrics:

- `--metric rmse3d`  
  `sqrt(mean(||prediction-target||²))`, consistent with the manuscript
  equation.

- `--metric mean_euclidean`  
  `mean(||prediction-target||)`, retained for compatibility with earlier
  internal scripts.

## Example commands

Collect AirSim data:

```bash
python src/collect_airsim_data.py \
  --output-root data/raw \
  --environment forest \
  --num-frames 1500
```

Build the localization dataset:

```bash
python src/build_localization_dataset.py \
  --raw-dataset data/raw \
  --yolov7-repository external/YOLOv7_distance \
  --weights checkpoints/yolov7_distance_best.pt \
  --output data/processed/localization_dataset.npz \
  --preview-dir outputs/previews
```

Train and evaluate:

```bash
python src/train_and_evaluate.py \
  --dataset data/processed/localization_dataset.npz \
  --output-dir outputs/experiments \
  --runs 5 \
  --metric rmse3d \
  --evaluation-mode strict
```

Run the GT analysis:

```bash
python src/analyze_geometric_error.py \
  --gt-dataset data/processed/ground_truth_dataset.npz \
  --yolo-dataset data/processed/localization_dataset.npz \
  --output-dir outputs/geometric_analysis \
  --conditions gt,mix,yolo
```

Run the AirSim demonstration:

```bash
python src/realtime_airsim_demo.py \
  --yolov7-repository external/YOLOv7_distance \
  --yolo-weights checkpoints/yolov7_distance_best.pt \
  --localization-checkpoint outputs/experiments/runs/A3_yolo_geometry_kf_reference/run1.pt
```

The final GitHub README should be updated with the paper citation, external
YOLOv7 repository URL, Drive dataset links, screenshots, and license.
