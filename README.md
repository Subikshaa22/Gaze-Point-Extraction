# Gaze Estimation from Dynamic Attention Maps Using Probabilistic Models

## Overview

This repository contains the implementation for detecting gaze hotspot regions from dynamic attention heatmap videos generated using webcam-based eye-tracking tools such as GazeRecorder and RealEye.

The project benchmarks multiple deep learning architectures for gaze hotspot localization, including:

- RF-DETR
- Faster R-CNN
- Mask R-CNN
- Siamese U-Net

The models are evaluated using metrics such as mAP, IoU, training time, and inference runtime.

---

## Dataset

The dataset consists of annotated frames extracted from screen-recording videos containing dynamic gaze heatmaps.

Annotations were created using CVAT.

The dataset includes:
- Research papers
- Web pages
- Forms
- Tables
- Dark/light backgrounds
- Different hotspot intensities and transparencies

---

## Models

### Faster R-CNN
Two-stage object detector using Region Proposal Networks.

### Mask R-CNN
Segmentation-based detector adapted for gaze hotspot localization.

### RF-DETR
Transformer-based detector using receptive field attention.

### Siamese U-Net
Temporal segmentation architecture that processes consecutive frames jointly.

---

## Training

### Faster R-CNN

```bash
python faster_rcnn/train.py
```

### Mask R-CNN

```bash
python mask_rcnn/train.py
```

### RF-DETR

```bash
python rf_detr/train.py
```

### Siamese U-Net

```bash
python siamese_unet/train.py
```

---

## Inference

### Faster R-CNN

```bash
python faster_rcnn/infer.py
```

### Mask R-CNN

```bash
python mask_rcnn/infer.py
```

### RF-DETR

```bash
python rf_detr/infer.py
```

### Siamese U-Net

```bash
python siamese_unet/infer.py
```

---

## Results

| Model | mAP | Average IoU |
|---|---|---|
| Faster R-CNN | 0.13 | 0.20 |
| Siamese U-Net | 0.74 | 0.72 |
| RF-DETR | 0.9933 | 0.9682 |
| Mask R-CNN | 0.9027 | 0.9341 |


