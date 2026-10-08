"""Detect and report vehicles that remain inside a configured parking zone."""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


DEFAULT_VEHICLE_LABELS = {
    "auto rickshaw",
    "autorickshaw",
    "bike",
    "bus",
    "car",
    "cng",
    "easybike",
    "leguna",
    "motorbike",
    "motorcycle",
    "pickup",
    "rickshaw",
    "tractor",
    "truck",
    "van",
}


@dataclass
class TrackState:
    """Keep the current parking episode and path for one ByteTrack ID."""

    last_seen: float
    entered_at: float | None = None
    inside: bool = False
    violated: bool = False
    zone_index: int | None = None
    trail: deque[tuple[int, int]] = field(default_factory=deque)


def find_video(root: Path) -> Path | None:
    """Return a likely project video, preserving the previous script's defaults."""
    candidates = [
        root / "Illegal Parking Detection.mp4",
        root / "video.mp4",
        root / "video2.mp4",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return next(iter(sorted(root.rglob("*.mp4"))), None)


def parse_zone(value: str) -> list[np.ndarray]:
    """Parse one or more pixel polygons separated by `|`."""
    try:
        polygons = []
        for polygon_text in value.split("|"):
            points = [tuple(int(part.strip()) for part in pair.split(","))
                      for pair in polygon_text.split(";")]
            if len(points) < 3 or any(len(point) != 2 for point in points):
                raise ValueError
            polygons.append(np.asarray(points, dtype=np.int32))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Each zone needs at least three x,y points, e.g. '100,100;400,100;400,400'."
        ) from exc
    return polygons


def parse_frame_size(value: str) -> tuple[int, int]:
    """Parse a reference image size written as WIDTHxHEIGHT."""
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Frame size must look like 1140x586.") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Frame width and height must be positive.")
    return width, height


def scale_zones(zones: list[np.ndarray], reference_size: tuple[int, int],
                frame_size: tuple[int, int]) -> list[np.ndarray]:
    """Scale ROI vertices from their reference image dimensions to the video frame."""
    reference_width, reference_height = reference_size
    frame_width, frame_height = frame_size
    scaled_zones = []
    for zone in zones:
        scaled = zone.astype(np.float64)
        scaled[:, 0] *= frame_width / reference_width
        scaled[:, 1] *= frame_height / reference_height
        scaled[:, 0] = np.clip(np.rint(scaled[:, 0]), 0, frame_width - 1)
        scaled[:, 1] = np.clip(np.rint(scaled[:, 1]), 0, frame_height - 1)
        scaled_zones.append(scaled.astype(np.int32))
    return scaled_zones


def select_zones(frame: np.ndarray) -> list[np.ndarray] | None:
    """Let the operator click and confirm multiple polygons on the first video frame."""
    window = "Select forbidden zones"
    points: list[tuple[int, int]] = []
    zones: list[np.ndarray] = []
    cv2.namedWindow(window)

    def on_mouse(event: int, x: int, y: int, *_: Any) -> None:
        """Add a clicked vertex or undo the latest vertex."""
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.setMouseCallback(window, on_mouse)
    while True:
        display = frame.copy()
        for index, zone in enumerate(zones):
            cv2.polylines(display, [zone], True, (40, 220, 40), 2, cv2.LINE_AA)
            cv2.putText(display, f"ZONE {index + 1}", tuple(int(value) for value in zone[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 220, 40), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(display, point, 5, (0, 220, 255), -1)
        if len(points) > 1:
            cv2.polylines(display, [np.asarray(points, dtype=np.int32)], len(points) >= 3, (0, 220, 255), 2)
        cv2.putText(display, "Left-click vertices | Right-click undo | Enter save zone | F finish | Esc cancel",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (10, 13) and len(points) >= 3:
            zones.append(np.asarray(points, dtype=np.int32))
            points.clear()
        elif key == ord("f") and (zones or len(points) >= 3):
            if len(points) >= 3:
                zones.append(np.asarray(points, dtype=np.int32))
            break
        if key == 27:
            break
    cv2.destroyWindow(window)
    return zones or None


def normalize_label(label: str) -> str:
    """Normalize class labels so common spelling and punctuation variants match."""
    return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()


def resolve_vehicle_classes(names: dict[int, str] | list[str], requested: str | None) -> list[int]:
    """Map user-selected or known vehicle labels to model class indices."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    normalized_names = {int(class_id): normalize_label(str(name)) for class_id, name in items}
    labels = (
        {normalize_label(name) for name in requested.split(",") if name.strip()}
        if requested
        else DEFAULT_VEHICLE_LABELS
    )
    matched = [class_id for class_id, name in normalized_names.items() if name in labels]
    if requested and not matched:
        raise ValueError(f"None of the requested class names exist in model labels: {sorted(normalized_names.values())}")
    if not matched:
        raise ValueError(
            "No vehicle labels matched automatically. Use --classes with labels from this model: "
            f"{', '.join(normalized_names.values())}"
        )
    return matched


def class_name(names: dict[int, str] | list[str], class_id: int) -> str:
    """Return a readable class label for either supported Ultralytics names format."""
    return str(names[class_id])


def point_is_inside(point: tuple[int, int], zone: np.ndarray) -> bool:
    """Check polygon membership with OpenCV, including points on the boundary."""
    return cv2.pointPolygonTest(zone, point, False) >= 0


def to_numpy(value: Any) -> np.ndarray:
    """Convert an Ultralytics tensor or array to a CPU NumPy array."""
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def append_violation_log(log_path: Path, timestamp: str, track_id: int, label: str,
                         duration: float, image_path: Path | None) -> None:
    """Append one violation event to a CSV file, adding its header on first use."""
    needs_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if needs_header:
            writer.writerow(["timestamp", "vehicle_id", "class", "duration_seconds", "image"])
        writer.writerow([timestamp, track_id, label, f"{duration:.2f}", str(image_path or "")])


def save_violation(frame: np.ndarray, box: tuple[int, int, int, int], output_dir: Path,
                   timestamp: str, track_id: int, label: str, duration: float) -> Path | None:
    """Save a clipped vehicle image and append its corresponding event to the CSV log."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    margin = 12
    x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
    x2, y2 = min(width, x2 + margin), min(height, y2 + margin)
    image_path: Path | None = None
    if x2 > x1 and y2 > y1:
        image_path = output_dir / f"{timestamp.replace(':', '-')}_id-{track_id}_{label.replace(' ', '-')}.jpg"
        cv2.imwrite(str(image_path), frame[y1:y2, x1:x2])
    append_violation_log(output_dir.parent / "violations.csv", timestamp, track_id,
                         label, duration, image_path)
    return image_path


def draw_text(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int],
              scale: float = 0.55) -> None:
    """Draw legible overlay text with a dark outline."""
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line options for model, source, ROI, and violation policy."""
    parser = argparse.ArgumentParser(description="YOLOv8 illegal-parking detection with a polygonal ROI.")
    parser.add_argument("--source", help="Video path or camera index (for example, 0). Defaults to a project MP4 or webcam.")
    parser.add_argument("--weights", help="Model weights. Defaults to the trained project model, then yolov8n.pt.")
    parser.add_argument(
        "--zone", type=parse_zone,
        help="One or more pixel polygons: 'x,y;x,y;x,y|x,y;x,y;x,y'.",
    )
    parser.add_argument(
        "--zone-reference-size", type=parse_frame_size,
        help="Scale --zone coordinates from a reference frame size such as 1140x586.",
    )
    parser.add_argument("--classes", help="Comma-separated model class names; defaults to known vehicle labels.")
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds continuously inside the zone before violation.")
    parser.add_argument("--lost-grace", type=float, default=1.5, help="Seconds to retain a track when detections briefly disappear.")
    parser.add_argument("--point", choices=("center", "bottom-center"), default="bottom-center",
                        help="Bounding-box point used for zone membership.")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum detection confidence.")
    parser.add_argument("--trail-length", type=int, default=30, help="Maximum number of path points per tracked vehicle.")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Directory for violation images and CSV log.")
    parser.add_argument("--save-violations", action=argparse.BooleanOptionalAction, default=True,
                        help="Save a vehicle crop and CSV event when a violation starts.")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True,
                        help="Display the annotated live video; press q to stop.")
    return parser


def resolve_source(value: str | None, root: Path) -> int | str:
    """Resolve an explicit source or use a project video, falling back to webcam 0."""
    if value is None:
        video = find_video(root)
        return str(video) if video else 0
    if value.isdigit():
        return int(value)
    source = Path(value).expanduser()
    return str(source if source.is_absolute() else (root / source).resolve())


def resolve_weights(value: str | None, root: Path) -> Path:
    """Choose explicit weights, trained project weights, or the bundled YOLOv8n model."""
    if value:
        weights = Path(value).expanduser()
        weights = weights if weights.is_absolute() else (root / weights).resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"Weights file not found: {weights}")
        return weights
    trained = root / "runs" / "detect" / "bangladesh-vehicles" / "weights" / "best.pt"
    if trained.is_file():
        return trained
    fallback = root / "yolov8n.pt"
    if not fallback.is_file():
        raise FileNotFoundError(f"Neither trained nor fallback weights were found: {trained}, {fallback}")
    print("Warning: trained Bangladesh weights not found; using generic YOLOv8n.", file=sys.stderr)
    return fallback


def run(args: argparse.Namespace, root: Path) -> None:
    """Capture frames, track vehicles, apply parking timers, and render/log events."""
    source = resolve_source(args.source, root)
    model = YOLO(str(resolve_weights(args.weights, root)))
    names = model.names
    vehicle_classes = resolve_vehicle_classes(names, args.classes)
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {'CUDA' if device == '0' else 'CPU'}; vehicle classes: "
          f"{', '.join(class_name(names, class_id) for class_id in vehicle_classes)}")

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    output_dir = args.output.expanduser()
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    violations_dir = output_dir / "violations"
    states: dict[int, TrackState] = {}
    violation_count = 0
    zones = args.zone
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_step = 1.0 / fps if fps > 0 else 1.0 / 30.0
    previous_video_time = -1.0
    window = "Illegal Parking Detection"
    zones_scaled = False

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            height, width = frame.shape[:2]
            if zones is None:
                if not args.show:
                    raise ValueError("Provide --zone when using --no-show.")
                zones = select_zones(frame)
                if zones is None:
                    raise RuntimeError("ROI selection was cancelled.")
            elif args.zone_reference_size is not None and not zones_scaled:
                zones = scale_zones(zones, args.zone_reference_size, (width, height))
                zones_scaled = True
            for zone in zones:
                if (np.any(zone[:, 0] < 0) or np.any(zone[:, 0] >= width)
                        or np.any(zone[:, 1] < 0) or np.any(zone[:, 1] >= height)):
                    raise ValueError(f"Zone points must fit within the current frame ({width}x{height}).")

            if isinstance(source, int):
                now = time.monotonic()
            else:
                video_time = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                now = video_time if math.isfinite(video_time) and video_time > previous_video_time else previous_video_time + frame_step
                previous_video_time = now
            frame_start = time.monotonic()
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=args.conf,
                                  iou=0.5, device=device, classes=vehicle_classes, verbose=False)
            result = next(iter(results), None)
            if result is None:
                continue
            annotated = frame.copy()
            active_ids: set[int] = set()
            boxes = result.boxes

            if boxes is not None and boxes.id is not None:
                coordinates = to_numpy(boxes.xyxy).astype(np.int32)
                track_ids = to_numpy(boxes.id).astype(int).tolist()
                class_ids = to_numpy(boxes.cls).astype(int).tolist()
                confidences = to_numpy(boxes.conf).tolist()
                for raw_box, track_id, class_id, confidence in zip(coordinates, track_ids, class_ids, confidences):
                    x1, y1, x2, y2 = (int(value) for value in raw_box)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    bottom_center = ((x1 + x2) // 2, y2)
                    point = center if args.point == "center" else bottom_center
                    matching_zone = next(
                        (index for index, zone in enumerate(zones) if point_is_inside(point, zone)),
                        None,
                    )
                    inside = matching_zone is not None
                    label = class_name(names, class_id)
                    active_ids.add(track_id)
                    state = states.setdefault(track_id, TrackState(last_seen=now))
                    state.last_seen = now
                    state.trail = state.trail if state.trail.maxlen == args.trail_length else deque(state.trail, maxlen=args.trail_length)
                    state.trail.append(center)

                    if inside and not state.inside:
                        state.entered_at = now
                        state.violated = False
                        state.zone_index = matching_zone
                    elif not inside and state.inside:
                        state.entered_at = None
                        state.violated = False
                        state.zone_index = None
                    state.inside = inside
                    if inside:
                        state.zone_index = matching_zone
                    duration = max(0.0, now - state.entered_at) if inside and state.entered_at is not None else 0.0
                    if inside and duration >= args.duration and not state.violated:
                        state.violated = True
                        violation_count += 1
                        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
                        image_path = None
                        if args.save_violations:
                            violations_dir.mkdir(parents=True, exist_ok=True)
                            image_path = save_violation(frame, (x1, y1, x2, y2), violations_dir,
                                                        timestamp, track_id, normalize_label(label).replace(" ", "-"), duration)
                        print(f"ILLEGAL PARKING: id={track_id}, class={label}, duration={duration:.1f}s"
                              f"{f', image={image_path}' if image_path else ''}")

                    color = (0, 0, 255) if state.violated else (0, 190, 255) if inside else (40, 220, 40)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    for previous, current in zip(state.trail, list(state.trail)[1:]):
                        cv2.line(annotated, previous, current, color, 2, cv2.LINE_AA)
                    title = f"ID {track_id} | {label} {confidence:.2f}"
                    draw_text(annotated, title, (x1, max(20, y1 - 8)), color)
                    if inside:
                        status = f"ILLEGAL PARKING {duration:.1f}s" if state.violated else f"PARKED {duration:.1f}/{args.duration:.0f}s"
                        draw_text(annotated, status, (x1, min(height - 8, y2 + 22)), color)

            for track_id in list(states):
                if track_id not in active_ids and now - states[track_id].last_seen > args.lost_grace:
                    del states[track_id]

            for zone_index, zone in enumerate(zones):
                zone_violated = any(
                    state.violated and state.zone_index == zone_index
                    for state in states.values()
                )
                zone_color = (0, 0, 255) if zone_violated else (40, 220, 40)
                cv2.polylines(annotated, [zone], True, zone_color, 3, cv2.LINE_AA)
                anchor = (int(zone[0][0]), int(zone[0][1]))
                draw_text(annotated, f"ZONE {zone_index + 1}", anchor, zone_color, 0.5)
            overlay = f"Parking violations: {violation_count} | {width}x{height} | {device.upper()}"
            draw_text(annotated, overlay, (14, 28), (255, 255, 255), 0.7)
            elapsed = time.monotonic() - frame_start
            if elapsed > 0:
                draw_text(annotated, f"{1.0 / elapsed:.1f} FPS", (14, 54), (255, 255, 255), 0.55)
            if args.show:
                cv2.imshow(window, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if args.show:
            cv2.destroyAllWindows()
    print(f"Finished. Total violation events: {violation_count}. Output: {output_dir}")


def main() -> None:
    """Validate options and start the detector from the project directory."""
    parser = build_parser()
    args = parser.parse_args()
    if args.duration <= 0 or args.lost_grace < 0 or args.trail_length < 1 or not 0 <= args.conf <= 1:
        parser.error("--duration must be positive, --lost-grace non-negative, --trail-length at least 1, and --conf between 0 and 1.")
    if not args.show and args.zone is None:
        parser.error("--no-show requires a polygon supplied with --zone.")
    try:
        run(args, Path(__file__).resolve().parent)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()    python detect.py --source "video.mp4" --duration 10