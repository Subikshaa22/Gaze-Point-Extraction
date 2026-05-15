import torch
import time
import os
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from dataset import CocoMaskDataset
from model import get_model


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":

    IMG_FOLDER = r"C:\Users\VriddhiSubi\Downloads\dataset\Video3\frames"
    ANN_FILE = r"C:\Users\VriddhiSubi\Downloads\dataset\Video3\annotations_result3.json"

    NUM_CLASSES = 2
    BATCH_SIZE = 4
    NUM_EPOCHS = 25
    LR = 0.005
    NUM_WORKERS = 4

    # Folder to save models
    SAVE_DIR = "checkpoints_new"
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU Memory (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)

    dataset = CocoMaskDataset(IMG_FOLDER, ANN_FILE)

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )

    print("Total Images:", len(dataset))
    print("Batches per Epoch:", len(data_loader))
    print("Starting training...\n")

    model = get_model(NUM_CLASSES)
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=0.9,
        weight_decay=0.0005
    )

    scaler = GradScaler("cuda")

    best_loss = float("inf")

    # Total training timer
    total_start_time = time.time()

    for epoch in range(NUM_EPOCHS):

        model.train()
        total_loss = 0
        start_time = time.time()

        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for images, targets in progress_bar:

            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            with autocast("cuda"):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += losses.item()
            progress_bar.set_postfix(loss=losses.item())

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(data_loader)

        print(f"\nEpoch {epoch+1} Completed")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Time: {epoch_time:.2f} seconds\n")

        # Save latest checkpoint
        torch.save(
            model.state_dict(),
            os.path.join(SAVE_DIR, f"maskrcnn_epoch_{epoch+1}.pth")
        )

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                model.state_dict(),
                os.path.join(SAVE_DIR, "best_model.pth")
            )
            print(f"Best model updated (Loss: {best_loss:.4f})\n")

    # Total training time
    total_training_time = time.time() - total_start_time

    print("Training Finished.")
    print(f"Best Loss Achieved: {best_loss:.4f}")
    print(f"Total Training Time: {total_training_time:.2f} seconds")
    print(f"Total Training Time (minutes): {total_training_time/60:.2f} min")

    # Optional pretty format
    hours = total_training_time // 3600
    minutes = (total_training_time % 3600) // 60
    seconds = total_training_time % 60

    print(f"Total Training Time (formatted): {int(hours)}h {int(minutes)}m {seconds:.2f}s")