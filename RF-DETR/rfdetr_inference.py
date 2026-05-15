import sys
import transformers

# --- PATCHES ---
if not hasattr(transformers.utils, 'torch_int'):
    try:
        from transformers.pytorch_utils import torch_int
        transformers.utils.torch_int = torch_int
    except ImportError:
        transformers.utils.torch_int = int

from transformers.configuration_utils import PretrainedConfig
if not hasattr(PretrainedConfig, "_attn_implementation"):
    PretrainedConfig._attn_implementation = "eager"
# --- END PATCHES ---

import os
import cv2
import json
import torch
import numpy as np
import time
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from rfdetr import RFDETRMedium
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ===============================
# CONFIG
# ===============================
CHECKPOINT_PATH = r"C:\Users\VriddhiSubi\Downloads\Rf-detr\rfdetr_results_fresh\checkpoint_best_total.pth"

INPUT_FRAMES_DIR = r"C:\Users\VriddhiSubi\Downloads\dataset\RealEye1\frames"
ANNOTATION_PATH = r"C:\Users\VriddhiSubi\Downloads\dataset\RealEye1\annotations.json"

OUTPUT_DIR = r"C:\Users\VriddhiSubi\Downloads\Rf-detr\fresh_inference\RealEye1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRED_JSON_PATH = os.path.join(OUTPUT_DIR, "predictions.json")
PR_CURVE_PATH = os.path.join(OUTPUT_DIR, "pr_curve.png")

CONF_THRESHOLD = 0.4

# ===============================
# IoU FUNCTION
# ===============================
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)

# ===============================
# LOAD COCO GT
# ===============================
coco_gt = COCO(ANNOTATION_PATH)

filename_to_id = {
    img["file_name"]: img["id"]
    for img in coco_gt.dataset["images"]
}

# ===============================
# MODEL
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = RFDETRMedium(
    num_classes=1,
    device=device,
    pretrain_weights=CHECKPOINT_PATH
)

model.optimize_for_inference()
print("Model loaded")

# ===============================
# PROCESS FRAMES
# ===============================
frame_files = sorted([
    f for f in os.listdir(INPUT_FRAMES_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

predictions = []
all_ious = []

# ===============================
# TIMING START
# ===============================
start_time = time.time()
frame_times = []

for fname in tqdm(frame_files):
    if fname not in filename_to_id:
        continue

    frame_start = time.time()

    image_id = filename_to_id[fname]

    img_path = os.path.join(INPUT_FRAMES_DIR, fname)
    frame = cv2.imread(img_path)

    if frame is None:
        continue

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    detections = model.predict(image, threshold=CONF_THRESHOLD)

    boxes = detections.xyxy
    scores = detections.confidence

    # ===============================
    # GT BOXES
    # ===============================
    ann_ids = coco_gt.getAnnIds(imgIds=image_id)
    anns = coco_gt.loadAnns(ann_ids)

    gt_boxes = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        gt_boxes.append([x, y, x + w, y + h])

    # ===============================
    # IoU CALCULATION
    # ===============================
    image_ious = []

    for pbox in boxes:
        pbox = pbox.tolist()
        best_iou = 0

        for gt in gt_boxes:
            best_iou = max(best_iou, compute_iou(pbox, gt))

        if gt_boxes:
            image_ious.append(best_iou)

    if image_ious:
        all_ious.append(np.mean(image_ious))

    # ===============================
    # COCO PRED FORMAT
    # ===============================
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.tolist()

        w = x2 - x1
        h = y2 - y1

        pred = {
            "image_id": int(image_id),
            "category_id": 1,
            "bbox": [x1, y1, w, h],
            "score": float(score)
        }

        predictions.append(pred)

    frame_end = time.time()
    frame_times.append(frame_end - frame_start)

# ===============================
# SAVE PREDICTIONS
# ===============================
with open(PRED_JSON_PATH, "w") as f:
    json.dump(predictions, f)

print("Predictions saved at:", PRED_JSON_PATH)

# ===============================
# COCO EVALUATION
# ===============================
coco_dt = coco_gt.loadRes(PRED_JSON_PATH)

coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# ===============================
# EXTRACT METRICS
# ===============================
stats = coco_eval.stats

map_5095 = stats[0]
ap50 = stats[1]
ap75 = stats[2]

avg_iou = np.mean(all_ious) if all_ious else 0

print("\n===== FINAL METRICS =====")
print(f"mAP (0.5:0.95): {map_5095:.4f}")
print(f"AP50: {ap50:.4f}")
print(f"AP75: {ap75:.4f}")
print(f"Average IoU: {avg_iou:.4f}")

# ===============================
# PRECISION-RECALL CURVES
# ===============================
precision = coco_eval.eval['precision']
recall = coco_eval.params.recThrs
iou_thresholds = coco_eval.params.iouThrs

plt.figure()

for iou_target in [0.5, 0.75]:
    iou_idx = np.where(iou_thresholds == iou_target)[0][0]

    pr = precision[iou_idx, :, 0, 0, 2]

    valid = pr > -1
    pr_valid = pr[valid]
    recall_valid = recall[valid]

    plt.plot(recall_valid, pr_valid, label=f"IoU={iou_target}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()

plt.savefig(PR_CURVE_PATH)
print("PR curve saved at:", PR_CURVE_PATH)

# ===============================
# TIMING END
# ===============================
end_time = time.time()

total_time = end_time - start_time
avg_time = np.mean(frame_times) if frame_times else 0
fps = 1 / avg_time if avg_time > 0 else 0

print("\n===== PERFORMANCE =====")
print(f"Total time: {total_time:.2f} seconds")
print(f"Average time per frame: {avg_time:.4f} seconds")
print(f"FPS: {fps:.2f}")