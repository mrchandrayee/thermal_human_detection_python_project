
from __future__ import annotations
import colorsys
import cv2
import numpy as np

COLOR_PRIMARY = (0, 230, 255)
COLOR_ALERT = (0, 100, 255)
COLOR_BG = (20, 20, 20)
COLOR_TEXT = (255, 255, 255)
COLOR_ACCENT = (0, 255, 200)


def get_track_color(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return COLOR_PRIMARY
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


class VisualizerHUD:
    @staticmethod
    def draw_trails(img: np.ndarray, track_trails: dict, max_trail_len: int = 38):
        for track_id, points in track_trails.items():
            pts = list(points)[-max_trail_len:]
            if len(pts) < 2:
                continue
            color = get_track_color(track_id)
            n = len(pts)
            for i in range(1, n):
                ratio = i / n
                faded = tuple(int(c * ratio) for c in color)
                thickness = max(1, int(ratio * 4))
                cv2.line(img, pts[i - 1], pts[i], faded, thickness, cv2.LINE_AA)
            cv2.circle(img, pts[-1], 5, color, -1, cv2.LINE_AA)
            cv2.circle(img, pts[-1], 9, tuple(c // 2 for c in color), 1, cv2.LINE_AA)

    @staticmethod
    def draw_bbox(img, x1, y1, x2, y2, conf, track_id=None, thickness=2):
        h, w = img.shape[:2]
        scale = max(0.6, min(w, h) / 720.0)
        thick = max(2, int(thickness * scale))
        thin = max(1, thick // 2)
        color = get_track_color(track_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thin, cv2.LINE_AA)
        corner_len = max(10, int(min(w, h) * 0.06), int(30 * scale))
        corners = [
            (x1, y1, 1, 1), (x2, y1, -1, 1),
            (x1, y2, 1, -1), (x2, y2, -1, -1),
        ]
        for x, y, dx, dy in corners:
            cv2.line(img, (x, y), (x + dx * corner_len, y), color, thick, cv2.LINE_AA)
            cv2.line(img, (x, y), (x, y + dy * corner_len), color, thick, cv2.LINE_AA)
        label = f"PERSON #{track_id if track_id is not None else '-'} | {conf*100:.0f}%"
        font_scale = max(0.42, 0.52 * scale)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        top = max(0, y1 - th - 10)
        cv2.rectangle(img, (x1, top), (min(w - 1, x1 + tw + 10), top + th + 8), color, -1)
        cv2.putText(img, label, (x1 + 5, top + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (10, 10, 10), 1, cv2.LINE_AA)

    @staticmethod
    def draw_dashboard(img, current_people, total_tracks, fps, frame_number):
        h, w = img.shape[:2]
        panel_h = max(42, int(h * 0.085))
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
        scale = max(0.42, min(w, h) / 900)
        cv2.putText(img, "THERMAL HUMAN DETECTION", (12, int(panel_h * 0.42)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, COLOR_ACCENT, 1, cv2.LINE_AA)
        status = f"PERSONS: {current_people}   TRACKS: {total_tracks}   FPS: {fps:.1f}"
        cv2.putText(img, status, (12, int(panel_h * 0.82)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale * 0.78, COLOR_TEXT, 1, cv2.LINE_AA)
