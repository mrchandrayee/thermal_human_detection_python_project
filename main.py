
from __future__ import annotations
import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from thermal_detector import ThermalPersonDetector
from visualizer import VisualizerHUD, COLOR_BG, COLOR_TEXT, COLOR_ACCENT


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def load_config(path: str | None) -> dict:
    defaults = {
        "weights": "yolo11n.pt",
        "confidence": 0.25,
        "iou": 0.45,
        "device": "auto",
        "person_class_id": 0,
        "tracker": "bytetrack.yaml",
        "thermal_mode": "auto",
        "display": True,
        "save_output": True,
        "output": "output/thermal_detection.mp4",
        "trail_length": 38,
        "window_name": "Thermal Human Detection",
        "line_thickness": 2
    }
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f)
        defaults.update(user)
    return defaults


def prepare_thermal(frame: np.ndarray, mode: str) -> np.ndarray:
    """Prepare grayscale/thermal frames for YOLO while retaining a 3-channel image."""
    if mode == "off":
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if mode == "gray":
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # Auto mode preserves existing thermal contrast and normalizes low-dynamic-range feeds.
    p2, p98 = np.percentile(gray, (2, 98))
    if p98 > p2:
        norm = np.clip((gray.astype(np.float32) - p2) * (255.0 / (p98 - p2)), 0, 255).astype(np.uint8)
    else:
        norm = gray
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)


def main():
    parser = argparse.ArgumentParser(description="Thermal human detection and ByteTrack tracking")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--weights", default=None, help="YOLO weights (.pt); defaults to config")
    parser.add_argument("--config", default="config/settings.json")
    parser.add_argument("--output", default=None, help="Output MP4 path")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--thermal-mode", choices=["auto", "gray", "off"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config if Path(args.config).exists() else None)
    if args.weights:
        cfg["weights"] = args.weights
    if args.output:
        cfg["output"] = args.output
    if args.no_display:
        cfg["display"] = False
    if args.thermal_mode:
        cfg["thermal_mode"] = args.thermal_mode

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 120:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    output_path = Path(cfg["output"])
    writer = None
    if cfg["save_output"]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    detector = ThermalPersonDetector(
        weights_path=cfg["weights"],
        conf=float(cfg["confidence"]),
        iou=float(cfg["iou"]),
        device=cfg["device"],
        person_class_id=int(cfg["person_class_id"]),
        tracker=cfg["tracker"],
    )

    trails = defaultdict(lambda: deque(maxlen=int(cfg["trail_length"])))
    frame_count = 0
    started = time.time()
    last_fps_time = started
    fps_counter = 0
    display_fps = 0.0

    print("Thermal Human Detection started. Press Q or ESC to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            processed = prepare_thermal(frame, cfg["thermal_mode"])
            detections = detector.track_frame(processed)

            for det in detections:
                tid = det["track_id"]
                if tid is not None:
                    trails[tid].append(det["center"])

            fps_counter += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                display_fps = fps_counter / (now - last_fps_time)
                fps_counter = 0
                last_fps_time = now

            canvas = frame.copy()
            VisualizerHUD.draw_trails(canvas, trails)
            for det in detections:
                x1, y1, x2, y2 = det["box"]
                VisualizerHUD.draw_bbox(
                    canvas, x1, y1, x2, y2, det["conf"], det["track_id"],
                    thickness=int(cfg["line_thickness"])
                )
            VisualizerHUD.draw_dashboard(
                canvas, len(detections), len(trails), display_fps, frame_count
            )

            if writer is not None:
                writer.write(canvas)
            if cfg["display"]:
                cv2.imshow(cfg["window_name"], canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = max(time.time() - started, 1e-6)
    print(f"Processed {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS).")
    if writer is not None:
        print(f"Annotated video saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
