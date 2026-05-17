import argparse
from pathlib import Path

import cv2

from blue_light import associate_regions, extract_regions, preprocess, segment_blue
from detector import VehicleDetector
from utils import draw_label, ensure_dir, resize_keep_width


def process_frame(frame, detector):
    frame = resize_keep_width(frame, 1280)
    enhanced = preprocess(frame)
    mask = segment_blue(enhanced, adaptive=True)
    vehicles = detector.detect(frame)
    regions = extract_regions(mask)
    regions, blue_count = associate_regions(regions, vehicles)

    vis = frame.copy()
    for i, car in enumerate(vehicles):
        has_blue = any(r.matched_vehicle == i for r in regions)
        color = (0, 0, 255) if has_blue else (0, 200, 0)
        label = "Blue Vehicle" if has_blue else "Vehicle"
        cv2.rectangle(vis, (car.x1, car.y1), (car.x2, car.y2), color, 2)
        draw_label(vis, label, (car.x1, car.y1 - 8), color)

    for r in regions:
        color = (255, 0, 0) if r.matched_vehicle >= 0 else (0, 165, 255)
        b = r.box
        cv2.rectangle(vis, (b.x1, b.y1), (b.x2, b.y2), color, 2)
        draw_label(vis, r.source, (b.x1, b.y2 + 18), color)

    summary = f"vehicles={len(vehicles)}, blue_vehicles={blue_count}, blue_regions={len(regions)}"
    draw_label(vis, summary, (20, 35), (0, 0, 255))
    return vis, mask, {
        "vehicles": len(vehicles),
        "blue_vehicles": blue_count,
        "blue_regions": len(regions),
        "filtered_interference": sum(1 for r in regions if r.matched_vehicle < 0),
    }


def process_image(input_path: Path, output_path: Path, detector):
    frame = cv2.imread(str(input_path))
    if frame is None:
        raise FileNotFoundError(f"cannot read image: {input_path}")
    vis, mask, stats = process_frame(frame, detector)
    ensure_dir(str(output_path.parent))
    cv2.imwrite(str(output_path), vis)
    cv2.imwrite(str(output_path.with_name(output_path.stem + "_mask.png")), mask)
    print(stats)


def process_video(input_path: Path, output_path: Path, detector):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    first_ok, first = cap.read()
    if not first_ok:
        raise RuntimeError("empty video")
    vis, _, stats = process_frame(first, detector)
    h, w = vis.shape[:2]
    ensure_dir(str(output_path.parent))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    writer.write(vis)
    total = 1
    blue_frames = 1 if stats["blue_vehicles"] > 0 else 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis, _, stats = process_frame(frame, detector)
        writer.write(vis)
        total += 1
        blue_frames += 1 if stats["blue_vehicles"] > 0 else 0
    cap.release()
    writer.release()
    print({"frames": total, "frames_with_blue_vehicle": blue_frames})


def main():
    parser = argparse.ArgumentParser(description="Highway blue-light vehicle detection")
    parser.add_argument("--input", default="input/highway_720p.jpg")
    parser.add_argument("--output", default=None)
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()

    input_path = Path(args.input)
    output = Path(args.output) if args.output else Path("output") / ("result.mp4" if input_path.suffix.lower() in {".mp4", ".avi", ".mov"} else "result.jpg")
    detector = VehicleDetector(args.weights, args.conf)

    if input_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        process_video(input_path, output, detector)
    else:
        process_image(input_path, output, detector)


if __name__ == "__main__":
    main()
