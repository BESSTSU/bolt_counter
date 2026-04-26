"""
download_dataset.py — ดึง Dataset จาก Roboflow
Usage: python download_dataset.py
       python download_dataset.py --workspace MY_WS --project MY_PROJ --version 1
"""

import argparse
import os
import shutil
from pathlib import Path


def download_roboflow(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    export_format: str = "yolov8",
    output_dir: str = "dataset",
    show_progress: bool = True,
):
    try:
        from roboflow import Roboflow
    except ImportError:
        print("❌ ยังไม่ได้ติดตั้ง: pip install roboflow")
        return None

    print(f"\n{'─'*50}")
    print(f"  🔗 Workspace : {workspace}")
    print(f"  📦 Project   : {project}")
    print(f"  🏷️  Version   : {version}")
    print(f"  📐 Format    : {export_format}")
    print(f"{'─'*50}")

    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    dataset = project_obj.version(version).download(
        export_format,
        location=output_dir,
        overwrite=True,
    )

    print(f"\n✅ Downloaded → {dataset.location}")
    _print_dataset_info(dataset.location)
    _fix_yaml_path(dataset.location)
    return dataset


def _print_dataset_info(dataset_dir: str):
    """แสดงจำนวนภาพใน train/val/test"""
    base = Path(dataset_dir)
    print(f"\n  📊 Dataset summary:")
    total = 0
    for split in ["train", "valid", "test"]:
        img_dir = base / split / "images"
        if img_dir.exists():
            count = len(list(img_dir.glob("*.*")))
            total += count
            print(f"     {split:<8}: {count:>5} images")

    # แสดง class names จาก data.yaml
    yaml_file = base / "data.yaml"
    if yaml_file.exists():
        import yaml
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])
        nc = data.get("nc", len(names))
        print(f"\n  🏷️  Classes ({nc}): {', '.join(names)}")
    print(f"\n  📁 Total images : {total}")


def _fix_yaml_path(dataset_dir: str):
    """
    แก้ path ใน data.yaml ให้เป็น absolute path
    เพื่อให้ train ได้จากทุก working directory
    """
    yaml_file = Path(dataset_dir) / "data.yaml"
    if not yaml_file.exists():
        return

    try:
        import yaml
        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        abs_path = str(Path(dataset_dir).resolve())
        data["path"] = abs_path

        # Roboflow ใช้ "valid" แต่ ultralytics ชอบ "val"
        if "valid" in str(data.get("val", "")):
            data["val"] = data["val"].replace("valid", "valid")  # keep as-is

        with open(yaml_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        print(f"\n  ✅ data.yaml patched: {yaml_file}")
    except Exception as e:
        print(f"  ⚠️  Could not patch yaml: {e}")


def check_dataset(dataset_dir: str = "dataset"):
    """ตรวจสอบ dataset ที่ดาวน์โหลดมาแล้ว"""
    base = Path(dataset_dir)
    if not base.exists():
        print(f"❌ ไม่พบโฟลเดอร์: {dataset_dir}")
        return

    print(f"\n{'═'*50}")
    print(f"  🔍 Checking dataset: {base.resolve()}")
    print(f"{'═'*50}")
    _print_dataset_info(dataset_dir)

    # ตรวจสอบ label format
    print(f"\n  🔬 Checking label format (sample):")
    for split in ["train", "valid"]:
        lbl_dir = base / split / "labels"
        if lbl_dir.exists():
            txts = list(lbl_dir.glob("*.txt"))[:3]
            for txt in txts:
                lines = txt.read_text().strip().splitlines()
                if lines:
                    sample = lines[0].split()
                    print(f"     {txt.name}: cls={sample[0]}, "
                          f"cx={float(sample[1]):.3f}, cy={float(sample[2]):.3f}")


# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Roboflow dataset")
    parser.add_argument("--api-key",   default=os.getenv("ROBOFLOW_API_KEY", ""),
                        help="Roboflow API Key (หรือ set env ROBOFLOW_API_KEY)")
    parser.add_argument("--workspace", default="", help="Workspace name/slug")
    parser.add_argument("--project",   default="", help="Project name/slug")
    parser.add_argument("--version",   type=int, default=1)
    parser.add_argument("--format",    default="yolov8",
                        choices=["yolov8", "yolov5", "coco", "voc"],
                        help="Export format (default: yolov8)")
    parser.add_argument("--output",    default="dataset")
    parser.add_argument("--check",     action="store_true",
                        help="แค่ตรวจสอบ dataset ที่มีอยู่แล้ว")
    args = parser.parse_args()

    if args.check:
        check_dataset(args.output)
    else:
        # ── ถ้าไม่ใส่ args ให้ถามแบบ interactive ──
        api_key   = args.api_key
        workspace = args.workspace
        project   = args.project
        version   = args.version

        if not api_key:
            print("\n📌 หา API Key ได้ที่: https://app.roboflow.com → Settings → API Keys")
            api_key = input("   Roboflow API Key: ").strip()
        if not workspace:
            workspace = input("   Workspace slug  : ").strip()
        if not project:
            project = input("   Project slug    : ").strip()
        if version == 1:
            v = input("   Version number  [1]: ").strip()
            version = int(v) if v else 1

        dataset = download_roboflow(
            api_key=api_key,
            workspace=workspace,
            project=project,
            version=version,
            export_format=args.format,
            output_dir=args.output,
        )

        if dataset:
            print(f"\n🎯 ขั้นตอนต่อไป:")
            print(f"   python train_model.py train \\")
            print(f"     --dataset {args.output} \\")
            print(f"     --classes bolt,nut,washer \\")
            print(f"     --epochs 100")
