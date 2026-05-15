import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
import torch.utils.data
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import logging
import random
import time
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# =========================
# DATASET
# =========================
class COCOHotspotDataset(Dataset):
    def __init__(self, image_dirs, annotation_paths):
        self.all_images = []
        self.all_targets = []
        self.image_to_path = {}

        for img_dir, ann_path in zip(image_dirs, annotation_paths):
            with open(ann_path, 'r') as f:
                coco_data = json.load(f)

            image_id_to_filename = {
                img['id']: img['file_name'] for img in coco_data['images']
            }

            image_annotations = defaultdict(list)
            for ann in coco_data['annotations']:
                x, y, w, h = ann['bbox']
                image_annotations[ann['image_id']].append([x, y, x+w, y+h])

            for image_id, filename in image_id_to_filename.items():
                path = os.path.join(img_dir, filename)
                if not os.path.exists(path):
                    continue

                self.all_images.append(image_id)
                self.all_targets.append(image_annotations[image_id])
                self.image_to_path[image_id] = path

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        image_id = self.all_images[idx]
        img = Image.open(self.image_to_path[image_id]).convert("RGB")

        img = torch.tensor(np.array(img) / 255.0, dtype=torch.float32).permute(2, 0, 1)

        boxes = self.all_targets[idx]
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4))

        target = {
            "boxes": boxes,
            "labels": torch.ones((len(boxes),), dtype=torch.int64)
        }

        return img, target

# =========================
# MODEL
# =========================
def get_model(num_classes=2):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

# =========================
# METRICS
# =========================
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)

def evaluate_metrics(model, data_loader, device):
    model.eval()

    ious = []
    all_precisions = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for pred, tgt in zip(outputs, targets):
                gt_boxes = tgt['boxes'].numpy()
                pred_boxes = pred['boxes'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()

                if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                    continue

                matched = 0
                for pb in pred_boxes:
                    best_iou = max([compute_iou(pb, gb) for gb in gt_boxes])
                    ious.append(best_iou)
                    if best_iou > 0.5:
                        matched += 1

                precision = matched / (len(pred_boxes) + 1e-6)
                all_precisions.append(precision)

    mean_iou = np.mean(ious) if ious else 0
    mAP = np.mean(all_precisions) if all_precisions else 0

    return mean_iou, mAP

# =========================
# TRAIN
# =========================
def train_one_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# =========================
# MAIN TRAIN LOOP
# =========================
def train_model(image_dirs, annotation_paths, epochs=25, batch_size=2):
    dataset = COCOHotspotDataset(image_dirs, annotation_paths)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model()
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    for epoch in range(epochs):
        start = time.time()

        loss = train_one_epoch(model, optimizer, loader, device)

        iou, mAP = evaluate_metrics(model, loader, device)

        elapsed = time.time() - start

        logging.info(
            f"Epoch {epoch+1:03d} | "
            f"Loss: {loss:.4f} | "
            f"IoU: {iou:.4f} | "
            f"mAP: {mAP:.4f} | "
            f"Time: {elapsed:.2f}s"
        )

    return model

# =========================
# ENTRY
# =========================
def main():
    image_dirs = [
        'frames'
    ]

    annotation_paths = [
        'annotations_result3.json'
    ]

    train_model(image_dirs, annotation_paths, epochs=25)

if __name__ == "__main__":
    main()