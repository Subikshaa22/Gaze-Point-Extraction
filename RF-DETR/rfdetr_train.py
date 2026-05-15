def main():
    import os
    import time
    import torch
    from rfdetr import RFDETRMedium

    BASE_DIR = r"C:\Users\VriddhiSubi\Downloads\Rf-detr"
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    OUTPUT_DIR = os.path.join(BASE_DIR, "rfdetr_results_fresh")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Dataset check
    for split in ["train", "valid"]:
        ann = os.path.join(DATASET_DIR, split, "_annotations.coco.json")
        if not os.path.exists(ann):
            raise FileNotFoundError(f"Missing {ann}")

    print("Dataset structure verified")

    model = RFDETRMedium(
        num_classes=1,
        device=device
    )

    print("RF-DETR Medium initialized")

    # ⏱️ Start timer
    start_time = time.time()

    model.train(
        dataset_dir=DATASET_DIR,
        epochs=25,
        batch_size=1,
        gradient_accumulation_steps=2,
        lr=1e-4,
        lr_backbone=1e-5,
        mixed_precision=True,
        workers=2,
        output_dir=OUTPUT_DIR,
        save_every=5,
        #resume=os.path.join(OUTPUT_DIR, "checkpoint.pth")
    )

    # ⏱️ End timer
    end_time = time.time()
    total_time = end_time - start_time

    # Convert to readable format
    hrs = int(total_time // 3600)
    mins = int((total_time % 3600) // 60)
    secs = int(total_time % 60)

    print("\nTRAINING COMPLETED")
    print(f"Total Training Time: {hrs}h {mins}m {secs}s")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()