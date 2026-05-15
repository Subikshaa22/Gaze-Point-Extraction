import torch
import numpy as np
from torchvision.datasets import CocoDetection
import torchvision.transforms as T


class CocoMaskDataset(CocoDetection):
    def __init__(self, img_folder, ann_file):
        super().__init__(img_folder, ann_file)

    def __getitem__(self, idx):

        img, targets = super().__getitem__(idx)
        w, h = img.size

        boxes = []
        labels = []
        masks = []

        for obj in targets:

            xmin, ymin, width, height = obj["bbox"]

            # Skip invalid width/height
            if width <= 0 or height <= 0:
                continue

            xmax = xmin + width
            ymax = ymin + height

            # Clamp to image boundaries
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)

            # Skip invalid boxes
            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj["category_id"])

            # Create rectangular mask
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 1
            masks.append(mask)

        # Skip empty images completely
        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self))

        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Faster mask stacking (fixes slow warning)
        masks = torch.tensor(np.array(masks), dtype=torch.uint8)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = torch.tensor([idx])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["area"] = area
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        img = T.ToTensor()(img)

        return img, target