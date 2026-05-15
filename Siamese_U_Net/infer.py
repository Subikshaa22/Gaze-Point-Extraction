import os
import cv2
import time
import torch
import numpy as np
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

class SiameseUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        return e1, e2, e3, b

    def forward(self, img1, img2):

        f1 = self.encode(img1)
        f2 = self.encode(img2)

        d1 = torch.abs(f1[0] - f2[0])
        d2 = torch.abs(f1[1] - f2[1])
        d3 = torch.abs(f1[2] - f2[2])
        db = torch.abs(f1[3] - f2[3])

        x = self.up3(db)
        x = torch.cat([x, d3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, d2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, d1], dim=1)
        x = self.dec1(x)

        return self.final(x)

def get_bbox(mask, threshold=0.5):

    mask = (mask > threshold).astype(np.uint8)

    ys, xs = np.where(mask == 1)

    if len(xs) == 0:
        return None

    return min(xs), min(ys), max(xs), max(ys)


def run():

    frames_dir = "frames"
    output_dir = "ouputs"

    os.makedirs(output_dir, exist_ok=True)

    model = SiameseUNet().to(DEVICE)

    model.load_state_dict(
        torch.load("smodel.pth", map_location=DEVICE)
    )

    model.eval()

    files = sorted(os.listdir(frames_dir))

    total_inference_time = 0
    total_frames = 0

    overall_start = time.time()

    for i in range(len(files) - 1):

        f1 = cv2.imread(os.path.join(frames_dir, files[i]))
        f2 = cv2.imread(os.path.join(frames_dir, files[i + 1]))

        orig = f2.copy()

        img1 = cv2.resize(f1, (256, 256)) / 255.0
        img2 = cv2.resize(f2, (256, 256)) / 255.0

        img1 = np.transpose(img1, (2, 0, 1))
        img2 = np.transpose(img2, (2, 0, 1))

        img1 = torch.tensor(
            img1,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        img2 = torch.tensor(
            img2,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        with torch.no_grad():
            pred = model(img1, img2)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()

        inference_time = end_time - start_time

        total_inference_time += inference_time
        total_frames += 1

        print(
            f"{files[i+1]} -> "
            f"Inference Time: {inference_time:.4f} sec"
        )

        pred = torch.sigmoid(pred)

        pred = pred[0][0].cpu().numpy()

        bbox = get_bbox(pred, threshold=0.5)

        if bbox:

            x1, y1, x2, y2 = bbox

            h, w = orig.shape[:2]

            x1 = int(x1 * w / 256)
            x2 = int(x2 * w / 256)

            y1 = int(y1 * h / 256)
            y2 = int(y2 * h / 256)

            cv2.rectangle(
                orig,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

        cv2.imwrite(
            os.path.join(output_dir, files[i + 1]),
            orig
        )

    overall_end = time.time()

    total_execution_time = overall_end - overall_start

    avg_time = total_inference_time / total_frames

    fps = 1 / avg_time

    print(f"Total Frames Processed : {total_frames}")
    print(f"Average Inference Time : {avg_time:.4f} sec/frame")
    print(f"Complete Execution Time: {total_execution_time:.4f} sec")

    print("Done!")

if __name__ == "__main__":
    run()


# ==============================
# Total Frames Processed : 325
# Average Inference Time : 0.2018 sec/frame
# Approx FPS             : 4.96
# Complete Execution Time: 89.8909 sec
# ==============================
