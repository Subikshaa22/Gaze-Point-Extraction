import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from torchvision import transforms
from model import get_model

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# =========================
# START TIMER
# =========================
start_time = time.time()

# =========================
# SETTINGS
# =========================
MODEL_PATH = r"C:\Users\VriddhiSubi\Downloads\Mask_RCNN\checkpoints_new\best_model.pth"
IMAGE_PATH = r"C:\Users\VriddhiSubi\Downloads\dataset\RealEye1\frames"
ANNOTATION_PATH = r"C:\Users\VriddhiSubi\Downloads\dataset\RealEye1\annotations.json"

OUTPUT_FOLDER = "eval_outputs_RealEye1"
PLOTS_FOLDER = "eval_plots_RealEye1"

NUM_CLASSES = 2
CONF_THRESHOLD = 0.5

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# LOAD MODEL
# =========================
model = get_model(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()

# =========================
# COCO GT
# =========================
coco_gt = COCO(ANNOTATION_PATH)

# =========================
# FOLDERS
# =========================
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

transform = transforms.ToTensor()

# =========================
# STORAGE
# =========================
coco_results = []
ious = []

all_scores = []
all_tp = []
all_fp = []

# =========================
# IOU FUNCTION
# =========================
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

# =========================
# INFERENCE LOOP
# =========================
image_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith((".jpg", ".png"))]

for img_name in image_files:
    img_path = os.path.join(IMAGE_PATH, img_name)

    image = cv2.imread(img_path)
    original = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_tensor = transform(image_rgb).unsqueeze(0).to(device)

    # get image_id
    img_id = None
    for img in coco_gt.dataset["images"]:
        if img["file_name"] == img_name:
            img_id = img["id"]
            break
    if img_id is None:
        continue

    # ground truth
    ann_ids = coco_gt.getAnnIds(imgIds=img_id)
    anns = coco_gt.loadAnns(ann_ids)

    gt_boxes = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        gt_boxes.append([x, y, x+w, y+h])

    gt_boxes = np.array(gt_boxes)

    # inference
    with torch.no_grad():
        outputs = model(img_tensor)

    boxes = outputs[0]["boxes"].cpu().numpy()
    scores = outputs[0]["scores"].cpu().numpy()

    matched_gt = set()

    for i in range(len(boxes)):
        if scores[i] < CONF_THRESHOLD:
            continue

        pred_box = boxes[i]

        best_iou = 0
        best_gt_idx = -1

        for j, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        ious.append(best_iou)

        if best_iou >= 0.5 and best_gt_idx not in matched_gt:
            all_tp.append(1)
            all_fp.append(0)
            matched_gt.add(best_gt_idx)
        else:
            all_tp.append(0)
            all_fp.append(1)

        all_scores.append(scores[i])

        xmin, ymin, xmax, ymax = pred_box
        coco_results.append({
            "image_id": img_id,
            "category_id": 1,
            "bbox": [xmin, ymin, xmax-xmin, ymax-ymin],
            "score": float(scores[i])
        })

        xmin, ymin, xmax, ymax = pred_box.astype(int)
        cv2.rectangle(original, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(original, f"{scores[i]:.2f}", (xmin, ymin-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    save_path = os.path.join(OUTPUT_FOLDER, img_name)
    cv2.imwrite(save_path, original)

# =========================
# COCO EVAL
# =========================
coco_dt = coco_gt.loadRes(coco_results)
coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

mAP = coco_eval.stats[0]
AP50 = coco_eval.stats[1]
AP75 = coco_eval.stats[2]

# =========================
# IoU
# =========================
avg_iou = np.mean(ious)

# =========================
# TIME END
# =========================
end_time = time.time()
total_time = end_time - start_time  # seconds

# =========================
# FINAL OUTPUT
# =========================
print("\n========== FINAL METRICS ==========")
print(f"mAP (0.5:0.95): {mAP:.4f}")
print(f"AP50: {AP50:.4f}")
print(f"AP75: {AP75:.4f}")
print(f"Average IoU: {avg_iou:.4f}")
print("===================================")

print(f"\nTotal Execution Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

print("Results saved in:", OUTPUT_FOLDER)
print("Plots saved in:", PLOTS_FOLDER)