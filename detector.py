"""
detector.py — Core Detection Engine
Supports: Hailo-8L NPU (Pi AI HAT) → YOLOv8 CPU fallback
Dense mode: SAHI sliced inference for tightly packed objects
"""

import cv2
import numpy as np
import time
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Detection result dataclass
# ──────────────────────────────────────────────
@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    center: Tuple[int, int] = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class FrameResult:
    detections: List[Detection]
    counts: Dict[str, int]
    inference_ms: float
    total_objects: int
    alerts: List[str]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        d = asdict(self)
        d["detections"] = [
            {
                "class": det.class_name,
                "conf": round(det.confidence, 3),
                "bbox": list(det.bbox),
            }
            for det in self.detections
        ]
        return d


# ──────────────────────────────────────────────
#  Expected count config (configurable per station)
# ──────────────────────────────────────────────
DEFAULT_EXPECTED = {
    "bolt": 4,
    "nut": 4,
    "washer": 8,
    "screw": 6,
}


# ──────────────────────────────────────────────
#  Hailo detector wrapper
# ──────────────────────────────────────────────
class HailoDetector:
    def __init__(self, model_path: str, labels: List[str], conf_threshold: float = 0.4):
        self.labels = labels
        self.conf_threshold = conf_threshold
        self._init_hailo(model_path)

    def _init_hailo(self, model_path: str):
        try:
            from hailo_platform import (
                HEF, VDevice, HailoStreamInterface,
                InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams,
                FormatType,
            )
            logger.info("✅ Hailo platform found — loading HEF model")
            self.target = VDevice()
            self.hef = HEF(model_path)
            configure_params = ConfigureParams.create_from_hef(
                self.hef, interface=HailoStreamInterface.PCIe
            )
            self.network_groups = self.target.configure(self.hef, configure_params)
            self.network_group = self.network_groups[0]
            self.network_group_params = self.network_group.create_params()
            self.input_vstreams_params = InputVStreamParams.make(
                self.network_group, format_type=FormatType.FLOAT32
            )
            self.output_vstreams_params = OutputVStreamParams.make(
                self.network_group, format_type=FormatType.FLOAT32
            )
            self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
            self.input_shape = self.input_vstream_info.shape  # (H, W, C)
            self.hailo_ready = True
            logger.info(f"  Model input shape: {self.input_shape}")
        except ImportError:
            logger.warning("⚠️  hailo_platform not installed — Hailo unavailable")
            self.hailo_ready = False
        except Exception as e:
            logger.warning(f"⚠️  Hailo init failed: {e}")
            self.hailo_ready = False

    def infer(self, frame: np.ndarray) -> List[Detection]:
        if not self.hailo_ready:
            return []
        from hailo_platform import InferVStreams
        h, w = self.input_shape[:2]
        resized = cv2.resize(frame, (w, h))
        img = resized.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)

        with InferVStreams(
            self.network_group, self.input_vstreams_params, self.output_vstreams_params
        ) as pipeline:
            input_data = {self.input_vstream_info.name: img}
            with self.network_group.activate(self.network_group_params):
                output = pipeline.infer(input_data)

        # parse YOLO output (adjust for your HEF post-processing)
        detections = []
        for key, val in output.items():
            predictions = val[0]  # remove batch dim
            for pred in predictions:
                if len(pred) < 6:
                    continue
                x1, y1, x2, y2, conf, cls_id = pred[:6]
                if conf < self.conf_threshold:
                    continue
                # scale back to original frame
                fh, fw = frame.shape[:2]
                bx1 = int(x1 / w * fw)
                by1 = int(y1 / h * fh)
                bx2 = int(x2 / w * fw)
                by2 = int(y2 / h * fh)
                cls_name = self.labels[int(cls_id)] if int(cls_id) < len(self.labels) else "unknown"
                detections.append(Detection(cls_name, float(conf), (bx1, by1, bx2, by2)))
        return detections


# ──────────────────────────────────────────────
#  YOLOv8 (ultralytics) CPU/GPU detector
#  Dense mode: iou=0.3 + optional SAHI slicing
# ──────────────────────────────────────────────
class YOLOv8Detector:
    def __init__(
        self,
        model_path: str,
        labels: List[str],
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.3,       # ← ต่ำลงสำหรับ dense objects
        use_sahi: bool = False,
        sahi_slice_size: int = 320,
        sahi_overlap: float = 0.2,
    ):
        self.labels = labels
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.use_sahi = use_sahi
        self.sahi_slice_size = sahi_slice_size
        self.sahi_overlap = sahi_overlap
        self.sahi_model = None
        self._init_model(model_path)

    def _init_model(self, model_path: str):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.ready = True
            logger.info(f"✅ YOLOv8 loaded: {model_path}")
            if self.use_sahi:
                self._init_sahi(model_path)
        except ImportError:
            logger.warning("⚠️  ultralytics not installed")
            self.ready = False
            self._init_opencv_dnn(model_path)

    def _init_sahi(self, model_path: str):
        try:
            from sahi import AutoDetectionModel
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=model_path,
                confidence_threshold=self.conf_threshold,
                device="cpu",
            )
            logger.info(
                f"✅ SAHI enabled — slice={self.sahi_slice_size}px "
                f"overlap={self.sahi_overlap}"
            )
        except ImportError:
            logger.warning("⚠️  sahi not installed (pip install sahi) — using normal inference")
            self.use_sahi = False
        except Exception as e:
            logger.warning(f"⚠️  SAHI init failed: {e} — using normal inference")
            self.use_sahi = False

    def _init_opencv_dnn(self, model_path: str):
        onnx_path = model_path.replace(".pt", ".onnx")
        if os.path.exists(onnx_path):
            self.net = cv2.dnn.readNetFromONNX(onnx_path)
            self.dnn_ready = True
            logger.info(f"✅ OpenCV DNN ONNX: {onnx_path}")
        else:
            self.dnn_ready = False
            logger.error("❌ No usable model found")

    # ── SAHI sliced inference ──────────────────
    def _infer_sahi(self, frame: np.ndarray) -> List[Detection]:
        from sahi.predict import get_sliced_prediction
        import PIL.Image
        pil_img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = get_sliced_prediction(
            image=pil_img,
            detection_model=self.sahi_model,
            slice_height=self.sahi_slice_size,
            slice_width=self.sahi_slice_size,
            overlap_height_ratio=self.sahi_overlap,
            overlap_width_ratio=self.sahi_overlap,
            postprocess_type="NMM",           # Non-Maximum Merging (ดีกว่า NMS สำหรับ dense)
            postprocess_match_threshold=0.3,
            verbose=0,
        )
        detections = []
        for pred in result.object_prediction_list:
            cls_id = pred.category.id
            cls_name = (
                self.labels[cls_id]
                if cls_id < len(self.labels)
                else pred.category.name
            )
            conf = pred.score.value
            b = pred.bbox
            x1, y1, x2, y2 = int(b.minx), int(b.miny), int(b.maxx), int(b.maxy)
            detections.append(Detection(cls_name, conf, (x1, y1, x2, y2)))
        return detections

    # ── Normal YOLOv8 inference ────────────────
    def _infer_yolo(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,     # ← key param for dense
            agnostic_nms=True,          # treat all classes same in NMS
            max_det=500,
            verbose=False,
        )
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                cls_name = (
                    self.labels[cls_id]
                    if cls_id < len(self.labels)
                    else r.names.get(cls_id, "unknown")
                )
                detections.append(Detection(cls_name, conf, (x1, y1, x2, y2)))
        return detections

    def infer(self, frame: np.ndarray) -> List[Detection]:
        if not self.ready:
            return []
        if self.use_sahi and self.sahi_model is not None:
            return self._infer_sahi(frame)
        return self._infer_yolo(frame)


# ──────────────────────────────────────────────
#  Main BoltCounter engine
# ──────────────────────────────────────────────
CLASS_COLORS = {
    "bolt":   (0, 200, 255),
    "nut":    (0, 255, 100),
    "washer": (255, 180, 0),
    "screw":  (200, 0, 255),
    "default":(100, 200, 255),
}

class BoltCounter:
    def __init__(
        self,
        model_path: str = "models/best.pt",
        hailo_model_path: str = "models/best.hef",
        labels: Optional[List[str]] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.3,
        expected_counts: Optional[Dict[str, int]] = None,
        use_hailo: bool = True,
        use_sahi: bool = False,
        sahi_slice_size: int = 320,
        sahi_overlap: float = 0.2,
    ):
        self.labels = labels or list(DEFAULT_EXPECTED.keys())
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.expected = expected_counts or DEFAULT_EXPECTED.copy()
        self.detector = None

        # Try Hailo first (Hailo doesn't support SAHI — uses fixed pipeline)
        if use_hailo and os.path.exists(hailo_model_path):
            hd = HailoDetector(hailo_model_path, self.labels, conf_threshold)
            if hd.hailo_ready:
                self.detector = hd
                self.backend = "hailo"
                logger.info("🚀 Using Hailo-8L NPU backend")

        if self.detector is None:
            self.detector = YOLOv8Detector(
                model_path,
                self.labels,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                use_sahi=use_sahi,
                sahi_slice_size=sahi_slice_size,
                sahi_overlap=sahi_overlap,
            )
            sahi_tag = " + SAHI" if use_sahi else ""
            self.backend = f"yolov8{sahi_tag}"
            logger.info(f"🖥️  Using YOLOv8 CPU backend{sahi_tag}")

    # ── process one frame ──────────────────────
    def process_frame(self, frame: np.ndarray) -> FrameResult:
        t0 = time.perf_counter()
        detections = self.detector.infer(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        counts: Dict[str, int] = {}
        for det in detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1

        alerts = self._check_alerts(counts)

        return FrameResult(
            detections=detections,
            counts=counts,
            inference_ms=round(elapsed_ms, 1),
            total_objects=len(detections),
            alerts=alerts,
        )

    # ── draw overlay on frame ──────────────────
    def draw(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        out = frame.copy()
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            color = CLASS_COLORS.get(det.class_name, CLASS_COLORS["default"])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        # ── count panel ──
        panel_w = 220
        panel = np.zeros((frame.shape[0], panel_w, 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)

        y = 30
        cv2.putText(panel, "PART COUNT", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 10
        cv2.line(panel, (5, y), (panel_w - 5, y), (60, 60, 60), 1)

        for cls, cnt in result.counts.items():
            y += 32
            exp = self.expected.get(cls, 0)
            ok = cnt == exp if exp > 0 else True
            color = (0, 220, 80) if ok else (0, 60, 255)
            cv2.putText(panel, f"{cls}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
            count_str = f"{cnt}" + (f"/{exp}" if exp > 0 else "")
            cv2.putText(panel, count_str, (panel_w - 70, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        y += 40
        cv2.line(panel, (5, y - 10), (panel_w - 5, y - 10), (60, 60, 60), 1)
        cv2.putText(panel, f"TOTAL: {result.total_objects}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        y += 30
        status = "✓ OK" if not result.alerts else "✗ ALERT"
        color = (0, 200, 80) if not result.alerts else (0, 60, 255)
        cv2.putText(panel, status, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        y += 30
        cv2.putText(panel, f"{result.inference_ms:.1f}ms  {self.backend}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)

        # ── alert banner ──
        if result.alerts:
            for i, alert in enumerate(result.alerts):
                cv2.putText(panel, alert[:26], (8, y + 25 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

        combined = np.hstack([out, panel])
        return combined

    # ── alert logic ───────────────────────────
    def _check_alerts(self, counts: Dict[str, int]) -> List[str]:
        alerts = []
        for cls, exp in self.expected.items():
            if exp <= 0:
                continue
            got = counts.get(cls, 0)
            if got != exp:
                diff = got - exp
                sign = "+" if diff > 0 else ""
                alerts.append(f"{cls}: {got}/{exp} ({sign}{diff})")
        return alerts
