import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float = 1.0
    label: str = "vehicle"

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resize_keep_width(frame: np.ndarray, width: int = 1280) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def iou(a: Box, b: Box) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def contained_ratio(inner: Box, outer: Box) -> float:
    x1 = max(inner.x1, outer.x1)
    y1 = max(inner.y1, outer.y1)
    x2 = min(inner.x2, outer.x2)
    y2 = min(inner.y2, outer.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max(1, inner.area)


def classify_blue_source(light_box: Box, car_box: Box) -> str:
    cx = (light_box.x1 + light_box.x2) / 2
    cy = (light_box.y1 + light_box.y2) / 2
    rel_x = (cx - car_box.x1) / max(1, car_box.x2 - car_box.x1)
    rel_y = (cy - car_box.y1) / max(1, car_box.y2 - car_box.y1)
    if rel_y < 0.45:
        return "Body glare"
    if rel_x < 0.45:
        return "Left lamp"
    if rel_x > 0.55:
        return "Right lamp"
    return "Lamp/plate glare"


def draw_label(img: np.ndarray, text: str, org: Tuple[int, int], color=(0, 0, 255)) -> None:
    x, y = org
    y = max(20, y)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
