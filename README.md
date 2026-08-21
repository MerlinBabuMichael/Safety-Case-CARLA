# Safety Case: CARLA Autonomous Driving Perception System

A safety analysis and verification of a three-model perception stack for an autonomous
driving prototype in the CARLA simulator. This repository accompanies the *Introduction to
Machine Learning Safety* final report and contains the training, evaluation, and
verification code behind the safety case.

## Overview

The system uses three independent binary classifiers, each built on a frozen ImageNet
pretrained ResNet-18 backbone, to detect the presence of a **pedestrian**, a **vehicle**,
and a **traffic light** from a single forward-facing RGB camera. The notebook trains these
models and then evaluates them against the verifications that make up the safety case:
in-distribution accuracy, adversarial robustness, calibration, out-of-distribution
detection, and explainability.

The accompanying report presents the full STPA analysis (losses, hazards, unsafe control
actions, loss scenarios, and safety constraints), the verification results, the residual
risks, and the deployment recommendation.

## Repository structure

```
.
├── Safety_Case_CARLA.ipynb     # main notebook: training, evaluation, verification
├── README.md
├── README_kprojection.md
├── kprojection.py
└── .gitignore
```

Trained model weights are hosted separately on Google Drive (see [Trained models](#trained-models)).

The dataset is expected in the following layout (not included in the repo):

```
carla_dataset/
├── train/            # 7,200 images
├── validation/       # held-out split for checkpoint selection
├── test/             # 3,600 images (in-distribution)
├── test-fog/         # distribution-shift test sets
├── test-night/
└── test-town-01/
    └── rgb-front/    # images
        labels.csv    # has_pedestrian, has_vehicle, has_traffic_light
        ...
```

## Requirements

- Python 3.x with a CUDA-capable GPU (developed on Google Colab)
- PyTorch and torchvision
- scikit-learn
- pandas, numpy
- matplotlib
- Pillow (PIL)
- grad-cam (`pip install grad-cam`)

## How to run

The notebook is written for Google Colab and mounts Google Drive for the dataset and model
checkpoints. There are two ways to get the trained models:

**Option A - use the provided weights (no training needed).** Download the three trained
checkpoints from Google Drive (see [Trained models](#trained-models)), place them where the
notebook expects them, then run the evaluation and verification sections directly. This
reproduces the results in the report without retraining.

**Option B - train from scratch.** Run the training section to regenerate
`has_pedestrian_best.pth`, `has_vehicle_best.pth`, and `has_traffic_light_best.pth`, then
run the evaluation and verification sections.

Training uses BCEWithLogitsLoss and the Adam optimizer (learning rate 1e-4, 15 epochs,
batch size 32). Only the final classification head is trained; the backbone stays frozen.

## Trained models

The three trained checkpoints are hosted on Google Drive (they exceed GitHub's file-size
limit):

Download: https://drive.google.com/drive/folders/1h4pygA77hWbUV4fuYpi2M6DLVQD5y6YO?usp=sharing

```
carla_models/
├── has_pedestrian_best.pth            # pedestrian detector (used in the report)
├── has_vehicle_best.pth               # vehicle detector
├── has_traffic_light_best.pth         # traffic-light detector
├── has_pedestrian_backdoor_best.pth   # backdoored model from the data-poisoning experiment (not part of the main safety case)
└── checkpoints/                       # per-epoch snapshots (epoch 5/10/15); intermediate, not needed to reproduce results
```

Each is a checkpoint dictionary saved during training (best validation loss). Download the
folder, then load a model with the notebook's `load_binary_resnet18()` helper:

```python
model = load_binary_resnet18('carla_models/has_vehicle_best.pth', device)
```

In the notebook these are loaded from `/content/drive/MyDrive/carla_models/`; update the
path to point to wherever you place the downloaded folder. These are the exact weights
behind the results reported in the accompanying safety case.

## Verifications

| Verification | Method | Result |
|---|---|---|
| V-1 In-distribution recall | Per-class recall on the test set | Partial (pedestrian fails) |
| V-2 Robustness | FGSM recall drop at various epsilon | Not met |
| V-3 Calibration | Expected Calibration Error, with temperature scaling | Met |
| V-4 OOD detection | MSP baseline vs Mahalanobis distance (AUROC) | Met (Mahalanobis) |
| V-5 Safe fallback | System-level design argument | Partial |

Supporting analysis includes Grad-CAM explainability, which reveals shortcut learning in
the detectors.

## Key findings

- The pedestrian detector reaches only 0.0014 recall, driven by severe class imbalance and
  a frozen backbone, leaving pedestrian detection largely uncontrolled.
- The models are not robust to small adversarial perturbations (FGSM).
- Calibration is acceptable, and a feature-based OOD monitor (Mahalanobis distance)
  detects distribution shift far more reliably than raw softmax confidence.
- On this evidence, the system is **not recommended for deployment** in its current form.

## Note

This work was produced for the 'Introduction to Machine Learning Safety' Course. See the accompanying report for the full
safety case, traceability, and residual-risk analysis.
