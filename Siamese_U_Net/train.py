import os
import cv2
import json
import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ChangeDataset(Dataset):
    def __init__(self, frames_dir, annotation_file, size=256):
        with open(annotation_file) as f:
            coco = json.load(f)

        self.frames_dir = frames_dir
        self.size = size
        self.images = sorted(coco["images"], key=lambda x: x["id"])

        self.bboxes = {}
        for ann in coco["annotations"]:
            self.bboxes.setdefault(ann["image_id"], []).append(ann["bbox"])

    def __len__(self):
        return len(self.images) - 1

    def __getitem__(self, idx):
        img1_info = self.images[idx]
        img2_info = self.images[idx + 1]

        img1 = cv2.imread(os.path.join(self.frames_dir, img1_info["file_name"]))
        img2 = cv2.imread(os.path.join(self.frames_dir, img2_info["file_name"]))

        h, w = img1.shape[:2]

        prev_mask = np.zeros((h, w), dtype=np.float32)
        curr_mask = np.zeros((h, w), dtype=np.float32)

        for bbox in self.bboxes.get(img1_info["id"], []):
            x, y, bw, bh = map(int, bbox)
            prev_mask[y:y+bh, x:x+bw] = 1

        for bbox in self.bboxes.get(img2_info["id"], []):
            x, y, bw, bh = map(int, bbox)
            curr_mask[y:y+bh, x:x+bw] = 1

        mask = (curr_mask - prev_mask) > 0
        mask = mask.astype(np.float32)

        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        mask = (mask > 0.1).astype(np.float32)

        img1 = cv2.resize(img1, (self.size, self.size)) / 255.0
        img2 = cv2.resize(img2, (self.size, self.size)) / 255.0
        mask = cv2.resize(mask, (self.size, self.size))

        img1 = np.transpose(img1, (2, 0, 1))
        img2 = np.transpose(img2, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        return (
            torch.tensor(img1, dtype=torch.float32),
            torch.tensor(img2, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )

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

bce = nn.BCEWithLogitsLoss()

def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (
        pred.sum() + target.sum() + smooth
    )

def loss_fn(pred, target):
    return bce(pred, target) + dice_loss(pred, target)

def compute_iou(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets).clamp(0, 1).sum(dim=(1, 2, 3))

    return ((intersection + 1e-6) / (union + 1e-6)).mean().item()


def compute_map(preds, targets, thresholds=np.arange(0.5, 1.0, 0.05)):
    preds = torch.sigmoid(preds)

    aps = []
    for t in thresholds:
        pred_bin = (preds > t).float()

        tp = (pred_bin * targets).sum(dim=(1, 2, 3))
        fp = (pred_bin * (1 - targets)).sum(dim=(1, 2, 3))
        fn = ((1 - pred_bin) * targets).sum(dim=(1, 2, 3))

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        ap = precision * recall
        aps.append(ap.mean().item())

    return np.mean(aps)


def train():
    dataset = ChangeDataset("frames", "annotations_result3.json", size=256)

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = SiameseUNet().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    scaler = torch.cuda.amp.GradScaler()

    epochs = 25

    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        total_loss = 0
        total_iou = 0
        total_map = 0

        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

        for img1, img2, mask in loop:
            img1, img2, mask = img1.to(DEVICE), img2.to(DEVICE), mask.to(DEVICE)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                preds = model(img1, img2)
                loss = loss_fn(preds, mask)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            # Metrics
            iou = compute_iou(preds.detach(), mask)
            m_ap = compute_map(preds.detach(), mask)

            total_loss += loss.item()
            total_iou += iou
            total_map += m_ap

            loop.set_postfix(loss=loss.item(), iou=iou, mAP=m_ap)

        scheduler.step()

        epoch_time = time.time() - start_time

        avg_loss = total_loss / len(loader)
        avg_iou = total_iou / len(loader)
        avg_map = total_map / len(loader)

        print(
            f"\nEpoch {epoch+1} Summary:"
            f"\nLoss: {avg_loss:.4f}"
            f"\nIoU: {avg_iou:.4f}"
            f"\nmAP: {avg_map:.4f}"
            f"\nTime: {epoch_time:.2f} sec\n"
        )

    torch.save(model.state_dict(), "siamese_unet_metrics.pth")
    print("Model saved!")

if __name__ == "__main__":
    train()


