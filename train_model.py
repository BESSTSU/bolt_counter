"""
train_model.py — Train / fine-tune YOLOv8 on your bolt/nut dataset
Then export to ONNX & Hailo HEF for Pi AI HAT

Dataset structure expected (YOLO format):
  dataset/
    images/
      train/  *.jpg
      val/    *.jpg
    labels/
      train/  *.txt
      val/    *.txt
    data.yaml
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


YAML_TEMPLATE = """path: {dataset_path}
train: images/train
val: images/val

nc: {num_classes}
names: {class_names}
"""


def create_yaml(dataset_path: str, class_names: list) -> str:
    yaml_content = YAML_TEMPLATE.format(
        dataset_path=os.path.abspath(dataset_path),
        num_classes=len(class_names),
        class_names=str(class_names),
    )
    yaml_path = os.path.join(dataset_path, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"✅ Created data.yaml: {yaml_path}")
    return yaml_path


def train(
    dataset_path: str,
    class_names: list,
    epochs: int = 10,
    imgsz: int = 640,
    batch: int = 16,
    base_model: str = "yolov8n.pt",   # nano = fastest on Pi
    output_name: str = "bolt_counter",
):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Install ultralytics first: pip install ultralytics")
        sys.exit(1)

    yaml_path = os.path.join(dataset_path, "data.yaml")
    model = YOLO(base_model)

    print(f"\n🚀 Starting training: {epochs} epochs, imgsz={imgsz}, batch={batch}")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=output_name,
        patience=20,
        save=True,
        plots=True,
        device="cpu",       # change to 0 for CUDA GPU
        # ── Dense object augmentation ──────────
        mosaic=1.0,         # mix 4 images → โมเดลเห็นของเยอะๆ ตอน train
        copy_paste=0.3,     # copy วัตถุไปวางใน scene อื่น
        degrees=15,         # rotate ±15°
        fliplr=0.5,         # flip horizontal
        flipud=0.1,         # flip vertical
        scale=0.4,          # zoom in/out
        hsv_h=0.015,        # hue shift
        hsv_s=0.5,          # saturation
        hsv_v=0.4,          # brightness
        overlap_mask=True,  # สำหรับ seg model
    )

    best_pt = Path(f"runs/detect/{output_name}/weights/best.pt")
    if best_pt.exists():
        Path("models").mkdir(exist_ok=True)
        shutil.copy(best_pt, "models/best.pt")
        print(f"\n✅ Model saved: models/best.pt")
        export_model("models/best.pt")
    else:
        print(f"⚠️  best.pt not found at {best_pt}")

    return results


def export_model(model_pt: str = "models/best.pt"):
    """Export .pt → ONNX → (optionally) Hailo HEF"""
    try:
        from ultralytics import YOLO
        model = YOLO(model_pt)
        onnx_path = model.export(format="onnx", opset=11, simplify=True)
        print(f"✅ ONNX exported: {onnx_path}")

        # Hailo export (requires hailo_model_zoo + Docker or Hailo SDK)
        print("\n📌 To convert ONNX → HEF for Hailo-8L:")
        print("   hailomz compile yolov8n --ckpt models/best.onnx \\")
        print("       --hw-arch hailo8l --calib-path /path/to/calib/images \\")
        print("       --output-dir models/")
        print("\n   Or use Hailo Model Zoo Docker:")
        print("   docker run --rm -v $(pwd):/workspace hailo/hailo_model_zoo:latest \\")
        print("       hailomz compile --ckpt /workspace/models/best.onnx \\")
        print("                       --hw-arch hailo8l \\")
        print("                       --output-dir /workspace/models")
    except Exception as e:
        print(f"Export error: {e}")


def validate(model_pt: str = "models/best.pt", dataset_yaml: str = "dataset/data.yaml"):
    from ultralytics import YOLO
    model = YOLO(model_pt)
    metrics = model.val(data=dataset_yaml)
    print(f"\n📊 Validation Results:")
    print(f"   mAP50    : {metrics.box.map50:.4f}")
    print(f"   mAP50-95 : {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall   : {metrics.box.mr:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 for bolt/nut detection")
    sub = parser.add_subparsers(dest="cmd")

    tr = sub.add_parser("train", help="Train model")
    tr.add_argument("--dataset", required=True)
    tr.add_argument("--classes", required=True, help="bolt,nut,washer,screw")
    tr.add_argument("--epochs", type=int, default=100)
    tr.add_argument("--imgsz", type=int, default=640)
    tr.add_argument("--batch", type=int, default=16)
    tr.add_argument("--base", default="yolov8n.pt")
    tr.add_argument("--name", default="bolt_counter")

    ex = sub.add_parser("export", help="Export .pt → ONNX + HEF")
    ex.add_argument("--model", default="models/best.pt")

    vl = sub.add_parser("validate", help="Run validation metrics")
    vl.add_argument("--model", default="models/best.pt")
    vl.add_argument("--yaml", default="dataset/data.yaml")

    args = parser.parse_args()

    if args.cmd == "train":
        train(
            dataset_path=args.dataset,
            class_names=args.classes.split(","),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            base_model=args.base,
            output_name=args.name,
        )
    elif args.cmd == "export":
        export_model(args.model)
    elif args.cmd == "validate":
        validate(args.model, args.yaml)
    else:
        parser.print_help()
