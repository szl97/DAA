from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from utils import Box, contained_ratio, iou


@dataclass
class BlueRegion:
    box: Box
    area: float
    mean_v: float
    source: str = "车灯蓝光"
    matched_vehicle: int = -1


def preprocess(frame: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    enhanced = cv2.merge([l2, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def _clip_box(box: Box, width: int, height: int) -> Box:
    return Box(
        max(0, min(width - 1, box.x1)),
        max(0, min(height - 1, box.y1)),
        max(0, min(width, box.x2)),
        max(0, min(height, box.y2)),
        box.score,
        box.label,
    )


def _light_candidates(gray: np.ndarray) -> List[BlueRegion]:
    threshold = max(185, int(np.percentile(gray, 98.5)))
    bright = cv2.inRange(gray, threshold, 255)
    background = cv2.GaussianBlur(gray, (21, 21), 0)
    contrast = cv2.subtract(gray, background)
    peaks = cv2.inRange(contrast, 22, 255)
    mask = cv2.bitwise_and(bright, peaks)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: List[BlueRegion] = []
    h, w = gray.shape[:2]
    image_area = h * w
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4 or area > 0.003 * image_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 2 or bh < 2:
            continue
        aspect = bw / max(1, bh)
        fill = area / max(1, bw * bh)
        if aspect < 0.25 or aspect > 4.5 or fill < 0.15:
            continue
        box = Box(x, y, x + bw, y + bh, 1.0, "blue_light")
        regions.append(BlueRegion(box, area, float(gray[y:y + bh, x:x + bw].mean())))
    return regions


def _has_vehicle_body(gray: np.ndarray, box: Box) -> bool:
    roi = gray[box.y1:box.y2, box.x1:box.x2]
    if roi.size == 0:
        return False
    global_dark = np.percentile(gray, 55)
    dark_ratio = float((roi < global_dark).mean())
    edges = cv2.Canny(roi, 50, 150)
    edge_ratio = float((edges > 0).mean())
    return dark_ratio >= 0.04 and edge_ratio >= 0.07


def _pair_headlight_vehicles(gray: np.ndarray, lights: List[BlueRegion]) -> List[Box]:
    h, w = gray.shape[:2]
    vehicles: List[Box] = []
    centers = [r.box.center for r in lights]
    for i, a in enumerate(lights):
        ax, ay = centers[i]
        if ay < 0.18 * h or ax > 0.75 * w:
            continue
        aw, ah = a.box.x2 - a.box.x1, a.box.y2 - a.box.y1
        for j in range(i + 1, len(lights)):
            b = lights[j]
            bx, by = centers[j]
            if by < 0.18 * h or bx > 0.75 * w:
                continue
            bw, bh = b.box.x2 - b.box.x1, b.box.y2 - b.box.y1
            dx, dy = abs(bx - ax), abs(by - ay)
            if dx < 10 or dx > 70 or dy > max(5, 0.25 * dx):
                continue
            size_ratio = max(aw * ah, bw * bh) / max(1, min(aw * ah, bw * bh))
            if size_ratio > 4:
                continue
            veh_w = int(dx * 2.2)
            veh_h = int(max(22, dx * 0.9))
            cx, cy = (ax + bx) // 2, (ay + by) // 2
            if cy + veh_h * 0.55 < 0.2 * h:
                continue
            box = Box(cx - veh_w // 2, cy - int(0.45 * veh_h), cx + veh_w // 2, cy + int(0.55 * veh_h), 0.45, "lamp_pair_vehicle")
            box = _clip_box(box, w, h)
            if not _has_vehicle_body(gray, box):
                continue
            vehicles.append(box)
    return vehicles


def _merge_vehicles(vehicles: List[Box]) -> List[Box]:
    merged: List[Box] = []
    for box in sorted(vehicles, key=lambda b: b.score, reverse=True):
        if any(iou(box, kept) > 0.35 or contained_ratio(box, kept) > 0.7 for kept in merged):
            continue
        merged.append(box)
    return merged


def build_vehicle_candidates(frame: np.ndarray, detected: List[Box]) -> Tuple[List[Box], List[BlueRegion]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lights = _light_candidates(gray)
    lamp_pair_vehicles = _pair_headlight_vehicles(gray, lights)
    vehicles = _merge_vehicles(detected + lamp_pair_vehicles)
    return vehicles, lights


def _is_lamp_position(light: BlueRegion, vehicle: Box) -> bool:
    cx, cy = light.box.center
    rel_x = (cx - vehicle.x1) / max(1, vehicle.x2 - vehicle.x1)
    rel_y = (cy - vehicle.y1) / max(1, vehicle.y2 - vehicle.y1)
    if vehicle.label == "lamp_pair_vehicle":
        return 0.2 <= rel_y <= 0.8
    return rel_y >= 0.48 and (rel_x <= 0.38 or rel_x >= 0.62)


def _classify_vehicle_light(light: BlueRegion, vehicle: Box) -> str:
    cx, cy = light.box.center
    rel_x = (cx - vehicle.x1) / max(1, vehicle.x2 - vehicle.x1)
    rel_y = (cy - vehicle.y1) / max(1, vehicle.y2 - vehicle.y1)
    if vehicle.label == "lamp_pair_vehicle":
        return "车灯蓝光"
    if rel_y >= 0.48 and (rel_x <= 0.38 or rel_x >= 0.62):
        return "车灯蓝光"
    return "车身反光蓝光"


def extract_vehicle_blue_regions(frame: np.ndarray, vehicles: List[Box], lights: List[BlueRegion] | None = None) -> Tuple[List[BlueRegion], np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if lights is None:
        lights = _light_candidates(gray)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    regions: List[BlueRegion] = []
    matched = set()

    for vehicle_idx, vehicle in enumerate(vehicles):
        vehicle_regions: List[BlueRegion] = []
        for light_idx, light in enumerate(lights):
            if light_idx in matched:
                continue
            if contained_ratio(light.box, vehicle) < 0.55:
                continue
            source = _classify_vehicle_light(light, vehicle)
            if vehicle.label == "lamp_pair_vehicle" and source != "车灯蓝光":
                continue
            region = BlueRegion(light.box, light.area, light.mean_v, source, vehicle_idx)
            vehicle_regions.append(region)
        vehicle_regions = sorted(vehicle_regions, key=lambda r: r.mean_v * r.area, reverse=True)[:2]
        for region in vehicle_regions:
            matched.add(lights.index(next(l for l in lights if l.box == region.box)))
            regions.append(region)
            b = region.box
            mask[b.y1:b.y2, b.x1:b.x2] = 255

    return regions, mask


def segment_blue(frame: np.ndarray, adaptive: bool = True) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for region in _light_candidates(gray):
        b = region.box
        mask[b.y1:b.y2, b.x1:b.x2] = 255
    return mask


def extract_regions(mask: np.ndarray, min_area: int = 12) -> List[BlueRegion]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: List[BlueRegion] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
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
            blue_vehicle_ids.add(best_idx)
        else:
            region.source = "环境蓝光干扰"
    return regions, len(blue_vehicle_ids)
