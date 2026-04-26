"""
webcam_live.py — Real-time webcam detection station
Usage:  python webcam_live.py [--camera 0] [--conf 0.45] [--no-hailo]
"""

import cv2
import argparse
import time
import json
from pathlib import Path
from detector import BoltCounter, DEFAULT_EXPECTED


def run_webcam(
    camera_id: int = 0,
    conf: float = 0.35,
    iou: float = 0.3,
    model: str = "models/best.pt",
    hailo_model: str = "models/best.hef",
    use_hailo: bool = True,
    use_sahi: bool = False,
    sahi_slice: int = 320,
    save_log: bool = True,
    expected: dict = None,
    fullscreen: bool = False,
):
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

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_id}")
        return

    win_name = "Bolt Counter — Station"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    log_entries = []
    fps_arr = []
    last_log_time = time.time()
    frame_count = 0

    print("▶  Webcam running. Keys: [q] quit  [s] snapshot  [r] reset log")

    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Frame capture failed")
            continue

        result = counter.process_frame(frame)
        overlay = counter.draw(frame, result)

        # FPS overlay
        fps_arr.append(1.0 / max(time.time() - t_start, 1e-5))
        if len(fps_arr) > 30:
            fps_arr.pop(0)
        fps = sum(fps_arr) / len(fps_arr)
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

        cv2.imshow(win_name, overlay)
        frame_count += 1

        # auto-log every 5s
        now = time.time()
        if save_log and (now - last_log_time) >= 5.0:
            log_entries.append(result.to_dict())
            last_log_time = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, overlay)
            print(f"📸 Saved {fname}")
        elif key == ord("r"):
            log_entries.clear()
            print("🔄 Log reset")

    cap.release()
    cv2.destroyAllWindows()

    if save_log and log_entries:
        log_path = f"logs/webcam_log_{int(time.time())}.json"
        Path("logs").mkdir(exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(log_entries, f, indent=2)
        print(f"📄 Log saved: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live webcam bolt/nut counter")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou",  type=float, default=0.3,
                        help="NMS IoU threshold (lower = better dense detection)")
    parser.add_argument("--sahi", action="store_true",
                        help="Enable SAHI sliced inference (accurate but slower)")
    parser.add_argument("--sahi-slice", type=int, default=320)
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--hailo-model", default="models/best.hef")
    parser.add_argument("--no-hailo", action="store_true")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--expected", type=str, default=None,
                        help='JSON string e.g. \'{"bolt":4,"nut":4}\'')
    args = parser.parse_args()

    expected = json.loads(args.expected) if args.expected else None

    run_webcam(
        camera_id=args.camera,
        conf=args.conf,
        iou=args.iou,
        model=args.model,
        hailo_model=args.hailo_model,
        use_hailo=not args.no_hailo,
        use_sahi=args.sahi,
        sahi_slice=args.sahi_slice,
        expected=expected,
        fullscreen=args.fullscreen,
    )
