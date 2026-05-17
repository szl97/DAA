from typing import List

import cv2
import numpy as np

from utils import Box


VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck in COCO


class VehicleDetector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35):
        self.conf = conf
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights)
        except Exception:
            self.model = None

    def detect(self, frame: np.ndarray) -> List[Box]:
        if self.model is not None:
            return self._detect_yolo(frame)
        return self._detect_fallback(frame)

    def _detect_yolo(self, frame: np.ndarray) -> List[Box]:
        boxes: List[Box] = []
        results = self.model(frame, conf=self.conf, verbose=False)
        for r in results:
            for b in r.boxes:
                cls = int(b.cls[0])
                if cls not in VEHICLE_CLASS_IDS:
                    continue
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                score = float(b.conf[0])
                boxes.append(Box(x1, y1, x2, y2, score, "vehicle"))
        return boxes

    def _detect_fallback(self, frame: np.ndarray) -> List[Box]:
        # Traditional fallback for environments without YOLO: locate large low-position objects.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 160)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = frame.shape[:2]
        boxes: List[Box] = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area < 0.003 * w * h or bw < 35 or bh < 25:
                continue
            if y + bh < 0.25 * h:
                continue
            ratio = bw / max(1, bh)
            if 0.5 <= ratio <= 5.5:
                boxes.append(Box(x, y, x + bw, y + bh, 0.5, "vehicle_candidate"))
        return boxes
