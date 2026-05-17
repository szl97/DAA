from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from utils import Box, contained_ratio, classify_blue_source


@dataclass
class BlueRegion:
    box: Box
    area: float
    mean_v: float
    source: str = "环境蓝光干扰"
    matched_vehicle: int = -1


def preprocess(frame: np.ndarray) -> np.ndarray:
    # Denoise and suppress local illumination variation.
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    enhanced = cv2.merge([l2, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def segment_blue(frame: np.ndarray, adaptive: bool = True) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if adaptive:
        v_min = max(120, int(np.percentile(v, 75)))
        s_min = max(55, int(np.percentile(s, 55)))
    else:
        v_min, s_min = 140, 70
    lower = np.array([85, s_min, v_min], dtype=np.uint8)
    upper = np.array([135, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def extract_regions(mask: np.ndarray, min_area: int = 35) -> List[BlueRegion]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: List[BlueRegion] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 0 and h > 0:
            regions.append(BlueRegion(Box(x, y, x + w, y + h, 1.0, "blue"), area, 0.0))
    return regions


def associate_regions(regions: List[BlueRegion], vehicles: List[Box]) -> Tuple[List[BlueRegion], int]:
    blue_vehicle_ids = set()
    for region in regions:
        best_idx = -1
        best_ratio = 0.0
        for i, car in enumerate(vehicles):
            ratio = contained_ratio(region.box, car)
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = i
        if best_idx >= 0 and best_ratio >= 0.45:
            region.matched_vehicle = best_idx
            region.source = classify_blue_source(region.box, vehicles[best_idx])
            blue_vehicle_ids.add(best_idx)
        else:
            region.source = "路牌/路灯/路面反光等环境蓝光干扰"
    return regions, len(blue_vehicle_ids)
