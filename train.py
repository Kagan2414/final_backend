"""
SafeGuard - YOLOv8 Accident Detection Model Training Script

This script fine-tunes a YOLOv8 model to detect road accidents.

Dataset structure required:
    dataset/
    ├── train/
    │   ├── images/    (jpg/png images)
    │   └── labels/    (YOLO format .txt files)
    ├── val/
    │   ├── images/
    │   └── labels/
    └── data.yaml

YOLO label format (one line per object):
    <class_id> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1].

Usage:
    # Train with default settings
    python train.py

    # Train with custom parameters
    python train.py --epochs 100 --batch 16 --imgsz 640 --model yolov8s.pt

    # Resume training
    python train.py --resume
"""

import argparse
import os
import yaml
from pathlib import Path
from ultralytics import YOLO


DEFAULT_CLASSES = [
    "accident",
    "non-accident",
]

EXTENDED_CLASSES = [
    "accident",
    "collision",
    "fire",
    "smoke",
    "damaged_vehicle",
    "rollover",
    "pedestrian_incident",
    "non-accident",
]


def create_dataset_yaml(
    dataset_dir: str,
    classes: list[str],
    output_path: str = "dataset/data.yaml",
) -> str:
    """Create the data.yaml configuration file for YOLO training."""
    data = {
        "path": os.path.abspath(dataset_dir),
        "train": "train/images",
        "val": "val/images",
        "nc": len(classes),
        "names": classes,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"[SafeGuard] Created dataset config: {output_path}")
    print(f"[SafeGuard] Classes ({len(classes)}): {classes}")
    return output_path


def setup_dataset_structure(base_dir: str = "dataset"):
    """Create the required directory structure for YOLO training."""
    dirs = [
        f"{base_dir}/train/images",
        f"{base_dir}/train/labels",
        f"{base_dir}/val/images",
        f"{base_dir}/val/labels",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"[SafeGuard] Created: {d}")

    print(f"""
[SafeGuard] Dataset structure ready at ./{base_dir}/
    
Next steps:
  1. Place training images in {base_dir}/train/images/
  2. Place training labels in {base_dir}/train/labels/
  3. Place validation images in {base_dir}/val/images/
  4. Place validation labels in {base_dir}/val/labels/

Label format (YOLO .txt, one line per object):
  <class_id> <x_center> <y_center> <width> <height>
  All values normalized to [0, 1].

Classes: {DEFAULT_CLASSES}
  class_id 0 = accident
  class_id 1 = non-accident

You can find accident detection datasets on:
  - Roboflow: https://roboflow.com (search "accident detection")
  - Kaggle: https://kaggle.com (search "road accident dataset YOLO")
  - Universe.roboflow.com: pre-labeled datasets in YOLO format
""")


def train(
    model_name: str = "yolov8n.pt",
    data_yaml: str = "dataset/data.yaml",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    project: str = "runs/detect",
    name: str = "accident_model",
    resume: bool = False,
    device: str = "",
):
    """Train/fine-tune YOLOv8 on accident detection data."""

    if not Path(data_yaml).exists():
        print(f"[SafeGuard] Error: {data_yaml} not found.")
        print("[SafeGuard] Run with --setup to create dataset structure first.")
        return None

    print(f"[SafeGuard] Training Configuration:")
    print(f"  Base model:  {model_name}")
    print(f"  Dataset:     {data_yaml}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch size:  {batch_size}")
    print(f"  Image size:  {img_size}")
    print(f"  Output:      {project}/{name}")
    print()

    model = YOLO(model_name)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=project,
        name=name,
        exist_ok=True,
        resume=resume,
        device=device if device else None,
        patience=20,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        # Augmentation for accident detection
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    best_path = Path(project) / name / "weights" / "best.pt"
    print(f"\n[SafeGuard] Training complete!")
    print(f"[SafeGuard] Best model saved at: {best_path}")
    print(f"\n[SafeGuard] To use this model with the detection server:")
    print(f"  set YOLO_CUSTOM_MODEL_PATH={best_path}")
    print(f"  python main.py")

    return results


def validate(model_path: str, data_yaml: str = "dataset/data.yaml"):
    """Run validation on the trained model."""
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, verbose=True)
    print(f"\n[SafeGuard] Validation Results:")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    return metrics


def export_model(
    model_path: str,
    format: str = "onnx",
):
    """Export the trained model to different formats."""
    model = YOLO(model_path)
    path = model.export(format=format)
    print(f"[SafeGuard] Model exported to: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="SafeGuard YOLOv8 Accident Detection Training")
    parser.add_argument("--setup", action="store_true", help="Create dataset directory structure")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model (default: yolov8n.pt)")
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--device", default="", help="Device (cuda:0, cpu, etc.)")
    parser.add_argument("--validate", type=str, help="Validate model at given path")
    parser.add_argument("--export", type=str, help="Export model at given path")
    parser.add_argument("--export-format", default="onnx", help="Export format")
    parser.add_argument("--classes", default="default", choices=["default", "extended"],
                        help="Class set to use")
    args = parser.parse_args()

    if args.setup:
        classes = EXTENDED_CLASSES if args.classes == "extended" else DEFAULT_CLASSES
        setup_dataset_structure()
        create_dataset_yaml("dataset", classes)
        return

    if args.validate:
        validate(args.validate, args.data)
        return

    if args.export:
        export_model(args.export, args.export_format)
        return

    train(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        resume=args.resume,
        device=args.device,
    )


if __name__ == "__main__":
    main()
