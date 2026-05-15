import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

FRAMES_DIR = "frames"
OUTPUT_DIR = "outputs"
MODEL_PATH = "model.pth"

CONFIDENCE_THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(num_classes=2):

    model = fasterrcnn_resnet50_fpn(pretrained=False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )

    return model


model = get_model()

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.to(DEVICE)

model.eval()

print(f"Model loaded on {DEVICE}")


files = sorted(os.listdir(FRAMES_DIR))

total_inference_time = 0
total_images = 0

overall_start = time.time()

for file_name in files:

    image_path = os.path.join(FRAMES_DIR, file_name)

    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        continue

    orig = image_bgr.copy()

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(image_rgb)

    image = torch.tensor(
        np.array(image) / 255.0,
        dtype=torch.float32
    ).permute(2, 0, 1)

    image = image.to(DEVICE)


    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()

    with torch.no_grad():

        outputs = model([image])

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()

    inference_time = end_time - start_time

    total_inference_time += inference_time
    total_images += 1


    output = outputs[0]

    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()


    for box, score in zip(boxes, scores):

        if score < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(
            orig,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            orig,
            f"{score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


    save_path = os.path.join(OUTPUT_DIR, file_name)

    cv2.imwrite(save_path, orig)

    print(
        f"{file_name} -> "
        f"Inference Time: {inference_time:.4f} sec"
    )


overall_end = time.time()

total_execution_time = overall_end - overall_start

avg_inference_time = total_inference_time / total_images

fps = 1 / avg_inference_time

print(f"Total Images Processed : {total_images}")
print(f"Average Inference Time : {avg_inference_time:.4f} sec/image")
print(f"Total Execution Time   : {total_execution_time:.4f} sec")

print(f"\nOutputs saved in: {OUTPUT_DIR}")
