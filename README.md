# Cooperative UAV Localization

This repository contains the code and dataset links for a simulation-based cooperative UAV localization framework.

The proposed system estimates the three-dimensional position of an **observer UAV** by visually observing a moving **reference UAV** with a known position. The framework combines:

- YOLOv7-based object-centric visual ranging
- Ray-casting-based geometric localization
- Kalman-filter-based temporal smoothing
- MLP-based residual error correction

![System Architecture](<images/drseminer-tez-ENnoekfSistem Mimarisi (Full Pipeline).drawio.png>)

## Overview

Reliable localization is essential for autonomous and cooperative UAV operations. In the proposed framework:

1. The reference UAV is detected in the observer UAV image.
2. The inter-UAV metric distance is estimated using a YOLOv7-based visual ranging model.
3. The image-plane location, estimated distance, observer orientation, and known reference-UAV position are combined through ray-casting geometry.
4. A linear Kalman filter improves the temporal consistency of the geometric estimate.
5. An MLP predicts the residual error remaining in the ray-casting estimate.
6. The predicted residual is added to the geometric estimate to obtain the final observer-UAV position.

The repository includes scripts for:

- AirSim data collection
- environment-based dataset construction
- feature-ablation experiments
- leave-one-environment-out evaluation
- controlled geometric-error analysis
- result visualization
- real-time AirSim demonstration

## Important Notes

- The experiments in this repository were conducted in **Microsoft AirSim**.
- The real-time script is intended as an **AirSim demonstration** and is not a ready-to-deploy physical-UAV implementation.
- Observer orientation, linear velocity, angular velocity, and ground-truth position are obtained from AirSim during data collection and evaluation.
- The YOLOv7 visual ranging component depends on an external repository and is not reimplemented here.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── config.example.json
├── images/
│   └── drseminer-tez-ENnoekfSistem Mimarisi (Full Pipeline).drawio.png
└── src/
    ├── common.py
    ├── collect_airsim_data.py
    ├── build_localization_dataset.py
    ├── build_ground_truth_dataset.py
    ├── train_and_evaluate.py
    ├── analyze_geometric_error.py
    ├── plot_results.py
    └── realtime_airsim_demo.py
```

## Scripts

### `collect_airsim_data.py`

Collects RGB images and synchronized UAV state information from AirSim. Data are stored separately for the following environments:

- `base`
- `forest`
- `snowy`
- `canyon`

### `build_localization_dataset.py`

Runs the external YOLOv7 detection-and-ranging model on the collected AirSim images and creates the localization dataset in NPZ format.

### `build_ground_truth_dataset.py`

Creates the ground-truth dataset used in the controlled geometric-error analysis. It combines YOLO-format ground-truth bounding-box labels, ground-truth distances, and AirSim metadata.

### `train_and_evaluate.py`

Constructs the 27-dimensional feature representation, trains the residual MLP, and performs:

- A1–A4 feature-ablation experiments
- within-environment evaluation
- leave-one-environment-out cross-environment evaluation

### `analyze_geometric_error.py`

Evaluates the ray-casting baseline under ground-truth, mixed, and YOLO-based visual measurement conditions. This script supports the controlled analysis of systematic geometric error.

### `plot_results.py`

Generates the feature-ablation and cross-environment loss figures.

### `realtime_airsim_demo.py`

Runs the complete localization pipeline online in AirSim using a trained YOLOv7 ranging model and a trained residual localization model.

## Datasets

The visual ranging and cooperative localization datasets are available through Google Drive:

**Dataset folder:**  
https://drive.google.com/drive/folders/1Kv5epGekZt7io7r9bolqL-2aVs6mDA1R?usp=sharing

The shared files include:

- the YOLOv7 visual ranging dataset
- the cooperative UAV localization dataset
- processed experiment files used by the training and evaluation scripts

The localization data were collected in four AirSim environments:

- open area (`base`)
- forest
- snowy terrain
- canyon

## External YOLOv7 Visual Ranging Repository

The visual ranging component uses the distance-estimation branch of the following repository:

**LOOKOUT-AI YOLOv7 Distance and Heading:**  
https://github.com/LOOKOUT-AI/YOLOv7_distance_heading/tree/distance_network

Clone the external repository separately:

```bash
git clone --branch distance_network https://github.com/LOOKOUT-AI/YOLOv7_distance_heading.git
```

The external repository should then be supplied to the relevant scripts using the `--yolov7-repository` argument.

## YOLOv7 Training Configuration

The YOLOv7-based visual ranging model was trained using the following Windows command:

```bat
set KMP_DUPLICATE_LIB_OK=TRUE && python YOLOv7_distance/train.py --workers 2 --device 0 --batch-size 4 --data D:/MK_TRAIN/drseminer/yolov7_dataset/data.yaml --img 480 640 --cfg YOLOv7_distance/cfg/training/yolov7_custom.yaml --weights YOLOv7_distance/init_weights.pt --name uav_dist_1000ep_v2 --hyp YOLOv7_distance/data/hyp.scratch.p5.yaml --epochs 1000
```

Before training, the maximum distance value in the hyperparameter file was set to 20 m:

```text
YOLOv7_distance/data/hyp.scratch.p5.yaml
```

```yaml
max_distance: 20
```

The trained model jointly provides:

- reference-UAV detections
- normalized bounding-box measurements
- inter-UAV metric distance estimates

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

The following external components are also required:

- Microsoft AirSim
- an Unreal Engine AirSim environment
- the external YOLOv7 distance repository
- trained YOLOv7 weights

## Example Workflow

### 1. Collect AirSim Data

Run the collector separately for each environment:

```bash
python src/collect_airsim_data.py \
  --output-root data/raw \
  --environment forest \
  --observer-uav Drone1 \
  --reference-uav Drone2 \
  --num-frames 1500 \
  --interval 0.5
```

Expected directory structure:

```text
data/raw/
├── base/
│   ├── images/
│   └── metadata/
├── forest/
│   ├── images/
│   └── metadata/
├── snowy/
│   ├── images/
│   └── metadata/
└── canyon/
    ├── images/
    └── metadata/
```

### 2. Build the Localization Dataset

```bash
python src/build_localization_dataset.py \
  --raw-dataset data/raw \
  --yolov7-repository external/YOLOv7_distance_heading \
  --weights checkpoints/yolov7_distance_best.pt \
  --output data/processed/localization_dataset.npz \
  --preview-dir outputs/previews
```

By default, the script selects at most 1,200 successful detections from each environment.

### 3. Build the Ground-Truth Dataset

The label format expected by the script is:

```text
class_id center_x center_y width height distance
```

Bounding-box values must be normalized, and distance must be provided in metres.

Example:

```text
0 0.512 0.463 0.084 0.071 11.42
```

Build the ground-truth dataset:

```bash
python src/build_ground_truth_dataset.py \
  --dataset-root data/ground_truth \
  --output data/processed/ground_truth_dataset.npz \
  --observer-uav Drone1 \
  --reference-uav Drone2
```

### 4. Train and Evaluate

Recommended strict evaluation:

```bash
python src/train_and_evaluate.py \
  --dataset data/processed/localization_dataset.npz \
  --output-dir outputs/experiments \
  --runs 5 \
  --metric rmse3d \
  --evaluation-mode strict \
  --deterministic
```

The strict mode uses separate training, validation, and test subsets. During leave-one-environment-out evaluation, the held-out environment is used only for final testing.

A compatibility mode is also available for reproducing earlier internal experiment behavior:

```bash
python src/train_and_evaluate.py \
  --dataset data/processed/localization_dataset.npz \
  --output-dir outputs/paper_compat \
  --runs 5 \
  --metric mean_euclidean \
  --evaluation-mode paper_compat
```

### 5. Run the Controlled Geometric-Error Analysis

Ground-truth condition only:

```bash
python src/analyze_geometric_error.py \
  --gt-dataset data/processed/ground_truth_dataset.npz \
  --output-dir outputs/geometric_analysis \
  --conditions gt \
  --metric rmse3d
```

Ground-truth, mixed, and YOLO conditions:

```bash
python src/analyze_geometric_error.py \
  --gt-dataset data/processed/ground_truth_dataset.npz \
  --yolo-dataset data/processed/localization_dataset.npz \
  --output-dir outputs/geometric_analysis \
  --conditions gt,mix,yolo \
  --metric rmse3d
```

The conditions are defined as follows:

- `gt`: ground-truth bounding box and ground-truth distance
- `mix`: YOLO bounding box and ground-truth distance
- `yolo`: YOLO bounding box and YOLO distance

### 6. Generate Result Figures

```bash
python src/plot_results.py \
  --data-dir outputs/experiments/run_data \
  --output-dir outputs/figures
```

### 7. Run the Real-Time AirSim Demonstration

```bash
python src/realtime_airsim_demo.py \
  --yolov7-repository external/YOLOv7_distance_heading \
  --yolo-weights checkpoints/yolov7_distance_best.pt \
  --localization-checkpoint outputs/experiments/runs/A3_yolo_geometry_kf_reference/run1.pt \
  --observer-uav Drone1 \
  --reference-uav Drone2
```

Press `q` in the OpenCV window to stop the demonstration.

## Feature Representation

The complete feature vector contains 27 dimensions:

| Feature group | Dimension | Description |
|---|---:|---|
| YOLO | 5 | Bounding-box center, width, height, and estimated distance |
| Geometry | 6 | World-frame viewing ray and relative ray-casting estimate |
| Observer state | 10 | Quaternion, linear velocity, and angular velocity |
| KF | 3 | Relative KF-smoothed position |
| Reference | 3 | Relative reference-UAV position |

The following ablation configurations are evaluated:

| Configuration | Feature groups | Dimension |
|---|---|---:|
| A1 | YOLO + Geometry + Reference | 14 |
| A2 | YOLO + Geometry + Observer State + Reference | 24 |
| A3 | YOLO + Geometry + KF + Reference | 17 |
| A4 | YOLO + Geometry + Observer State + KF + Reference | 27 |

## Evaluation Metrics

Two position-error options are provided:

### Three-Dimensional RMSE

```text
sqrt(mean(||prediction - target||²))
```

Use:

```bash
--metric rmse3d
```

### Mean Euclidean Position Error

```text
mean(||prediction - target||)
```

Use:

```bash
--metric mean_euclidean
```

The second option is retained for compatibility with earlier internal scripts.

## Citation

If you use this repository, its source code, or the shared datasets in academic work, **please cite the associated study**:

> **Cooperative UAV Localization with Visual Ranging and Residual Learning**

A complete BibTeX entry will be added after publication.

Please also cite the original work associated with the external YOLOv7 distance-estimation implementation when using that component:

https://github.com/LOOKOUT-AI/YOLOv7_distance_heading/tree/distance_network

## Data Availability

The datasets and processed experiment files are publicly available at:

https://drive.google.com/drive/folders/1Kv5epGekZt7io7r9bolqL-2aVs6mDA1R?usp=sharing

## Acknowledgements

The YOLOv7-based visual ranging component builds upon the external implementation provided by LOOKOUT-AI.

## Contact

For questions, reproducibility issues, or collaboration requests, please open an issue in this repository.
