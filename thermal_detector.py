
from __future__ import annotations
from pathlib import Path
from typing import Optional
import torch
import numpy as np
from ultralytics import YOLO


class ThermalPersonDetector:
    """YOLO person detector with persistent ByteTrack IDs.

    For best thermal performance, pass a thermal-trained YOLO weights file.
    The default COCO model is a convenient baseline, not a thermal-specialist model.
    """

    def __init__(
        self,
        weights_path: str = "yolo11n.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "auto",
        person_class_id: int = 0,
        tracker: str = "bytetrack.yaml",
    ):
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.conf = conf
        self.iou = iou
        self.person_class_id = person_class_id
        self.tracker = tracker
        if not Path(weights_path).exists() and weights_path.endswith(".pt"):
            # Ultralytics downloads official model weights on first use.
            pass
        print(f"Loading YOLO weights: {weights_path} ({self.device})")
        self.model = YOLO(weights_path)

    def track_frame(self, frame: np.ndarray) -> list[dict]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        detections = []
        if not results or results[0].boxes is None:
            return detections
        boxes = results[0].boxes
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0].detach().cpu().item())
            if cls_id != self.person_class_id:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].detach().cpu().numpy())
            score = float(box.conf[0].detach().cpu().item())
            track_id: Optional[int] = None
            if boxes.id is not None:
                track_id = int(boxes.id[i].detach().cpu().item())
            detections.append({
                "box": (x1, y1, x2, y2),
                "conf": score,
                "track_id": track_id,
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                "foot": ((x1 + x2) // 2, y2),
            })
        return detections
