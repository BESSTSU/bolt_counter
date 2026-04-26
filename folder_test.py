"""
folder_test.py — Batch image processing + accuracy report
Usage:  python folder_test.py --images /path/to/images [--labels /path/to/labels]
"""

import cv2
import argparse
import json
import time
import csv
from pathlib import Path
from typing import Dict, List, Optional
from detector import BoltCounter, DEFAULT_EXPECTED


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_yolo_label(label_path: Path, img_w: int, img_h: int, class_names: List[str]) -> dict:
    """Parse YOLO .txt label file → ground-truth counts"""
    counts: Dict[str, int] = {}
    if not label_path.exists():
        return counts
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            if cls_id < len(class_names):
                name = class_names[cls_id]
                counts[name] = counts.get(name, 0) + 1
    return counts


def run_folder_test(
    images_dir: str,
    labels_dir: Optional[str] = None,
    output_dir: str = "results",
    conf: float = 0.35,
    iou: float = 0.3,
    model: str = "runs/detect/bolt_counter8/weights/best.pt",
    hailo_model: str = "models/best.hef",
    use_hailo: bool = True,
    use_sahi: bool = False,
    sahi_slice: int = 320,
    expected: Optional[dict] = None,
    save_images: bool = True,
    class_names: Optional[List[str]] = None,
):
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "annotated").mkdir(exist_ok=True)

    counter = BoltCounter(
        model_path=model,
        hailo_model_path=hailo_model,
        conf_threshold=conf,
        iou_threshold=iou,
        expected_counts=expected or DEFAULT_EXPECTED,
        use_hailo=use_hailo,
        use_sahi=use_sahi,
        sahi_slice_size=sahi_slice,
    )
    names = class_names or list((expected or DEFAULT_EXPECTED).keys())

    img_files = sorted([p for p in images_path.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if not img_files:
        print(f"❌ No images found in {images_dir}")
        return

    print(f"\n{'─'*55}")
    print(f"  📂 Images found: {len(img_files)}")
    print(f"  🔍 Model: {model} | Backend: {'Hailo' if use_hailo else 'YOLOv8'}")
    print(f"{'─'*55}\n")

    summary_rows = []
    tp_total: Dict[str, int] = {}
    fp_total: Dict[str, int] = {}
    fn_total: Dict[str, int] = {}
    correct_images = 0
    total_inf_ms = []

    for idx, img_path in enumerate(img_files, 1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  ⚠️  Skip (unreadable): {img_path.name}")
            continue

        result = counter.process_frame(frame)
        total_inf_ms.append(result.inference_ms)

        # ground truth
        gt_counts: Dict[str, int] = {}
        if labels_dir:
            lbl = Path(labels_dir) / (img_path.stem + ".txt")
            gt_counts = load_yolo_label(lbl, frame.shape[1], frame.shape[0], names)

        # per-class TP / FP / FN
        all_classes = set(result.counts) | set(gt_counts)
        image_ok = True
        for cls in all_classes:
            pred = result.counts.get(cls, 0)
            gt = gt_counts.get(cls, 0)
            tp = min(pred, gt)
            fp = max(0, pred - gt)
            fn = max(0, gt - pred)
            tp_total[cls] = tp_total.get(cls, 0) + tp
            fp_total[cls] = fp_total.get(cls, 0) + fp
            fn_total[cls] = fn_total.get(cls, 0) + fn
            if pred != gt and gt > 0:
                image_ok = False

        if image_ok and gt_counts:
            correct_images += 1
        elif not gt_counts:
            correct_images += 1 if not result.alerts else 0

        row = {
            "image": img_path.name,
            "total_detected": result.total_objects,
            "inference_ms": result.inference_ms,
            "alerts": "; ".join(result.alerts) if result.alerts else "OK",
            **{f"pred_{c}": result.counts.get(c, 0) for c in names},
            **{f"gt_{c}": gt_counts.get(c, 0) for c in names},
        }
        summary_rows.append(row)

        # draw + save
        if save_images:
            overlay = counter.draw(frame, result)
            out_img = output_path / "annotated" / img_path.name
            cv2.imwrite(str(out_img), overlay)

        status = "✅" if not result.alerts else "⚠️ "
        print(f"  [{idx:03d}/{len(img_files)}] {status} {img_path.name:<30} "
              f"total={result.total_objects}  {result.inference_ms:.1f}ms  "
              + ("  ".join(f"{k}:{v}" for k, v in result.counts.items()) or "—"))

    # ── Accuracy metrics ───────────────────────────────────────
    print(f"\n{'═'*55}")
    print("  📊 ACCURACY REPORT")
    print(f"{'═'*55}")
    print(f"  Images processed  : {len(summary_rows)}")
    print(f"  Correct images    : {correct_images}/{len(summary_rows)} "
          f"({100*correct_images/max(len(summary_rows),1):.1f}%)")
    if total_inf_ms:
        avg_ms = sum(total_inf_ms) / len(total_inf_ms)
        print(f"  Avg inference     : {avg_ms:.1f} ms  ({1000/avg_ms:.1f} FPS equiv)")

    if tp_total:
        print(f"\n  {'Class':<12} {'Prec':>8} {'Recall':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
        print(f"  {'─'*58}")
        for cls in names:
            tp = tp_total.get(cls, 0)
            fp = fp_total.get(cls, 0)
            fn = fn_total.get(cls, 0)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            print(f"  {cls:<12} {prec:>8.3f} {rec:>8.3f} {f1:>8.3f} {tp:>6} {fp:>6} {fn:>6}")

    # ── Save CSV ──────────────────────────────────────────────
    csv_path = output_path / f"results_{int(time.time())}.csv"
    if summary_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n  💾 CSV saved: {csv_path}")

    # ── Save JSON summary ─────────────────────────────────────
    json_path = output_path / f"summary_{int(time.time())}.json"
    summary = {
        "total_images": len(summary_rows),
        "correct_images": correct_images,
        "accuracy": round(correct_images / max(len(summary_rows), 1), 4),
        "avg_inference_ms": round(sum(total_inf_ms) / max(len(total_inf_ms), 1), 2),
        "per_class": {
            cls: {
                "precision": round(tp_total.get(cls, 0) / max(tp_total.get(cls, 0) + fp_total.get(cls, 0), 1), 4),
                "recall": round(tp_total.get(cls, 0) / max(tp_total.get(cls, 0) + fn_total.get(cls, 0), 1), 4),
            }
            for cls in names
        },
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  📄 JSON summary saved: {json_path}")

    if save_images:
        print(f"  🖼️  Annotated images: {output_path/'annotated'}/")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Folder batch test + accuracy report")
    parser.add_argument("--images", required=True, help="Path to test images folder")
    parser.add_argument("--labels", default=None, help="Path to YOLO label .txt folder")
    parser.add_argument("--output", default="results", help="Output folder")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou",  type=float, default=0.3,
                        help="NMS IoU threshold (lower = keep more dense detections)")
    parser.add_argument("--sahi", action="store_true",
                        help="Use SAHI sliced inference for dense/small objects")
    parser.add_argument("--sahi-slice", type=int, default=320,
                        help="SAHI slice size in pixels (default 320)")
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--hailo-model", default="models/best.hef")
    parser.add_argument("--no-hailo", action="store_true")
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--expected", type=str, default=None,
                        help='JSON e.g. \'{"bolt":4,"nut":4}\'')
    parser.add_argument("--classes", type=str, default=None,
                        help='Comma-separated class names: bolt,nut,washer')
    args = parser.parse_args()

    expected = json.loads(args.expected) if args.expected else None
    classes = args.classes.split(",") if args.classes else None

    run_folder_test(
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.output,
        conf=args.conf,
        iou=args.iou,
        model=args.model,
        hailo_model=args.hailo_model,
        use_hailo=not args.no_hailo,
        use_sahi=args.sahi,
        sahi_slice=args.sahi_slice,
        expected=expected,
        save_images=not args.no_save_images,
        class_names=classes,
    )
