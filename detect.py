from pathlib import Path
import argparse
import sys

from ultralytics import YOLO


def find_video(root: Path):
    candidates = [
        root / "Illegal Parking Detection.mp4",
        root / "video.mp4",
        root / "video2.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for video in sorted(root.rglob("*.mp4")):
        return video

    return None


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Detect vehicles in a video.")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video path or camera index, e.g. 0 for webcam. Defaults to the first MP4 in the project folder.",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display the video window while processing.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    args = parser.parse_args()

    model_path = root / "runs" / "detect" / "train" / "weights" / "best.pt"
    if not model_path.exists():
        model_path = root / "yolov8n.pt"

    source = args.source if args.source else find_video(root)
    if source is None:
        source = 0

    if isinstance(source, str):
        source = Path(source).expanduser()
        if not source.is_absolute():
            source = (root / source).resolve()
        source = str(source)

    model = YOLO(str(model_path))

    try:
        model.track(
            source=source,
            show=args.show,
            save=True,
            conf=args.conf,
            iou=0.5,
            classes=[0, 1, 2],
            tracker="bytetrack.yaml",
            device="cpu",
            persist=True,
        )
    except Exception as exc:
        print(f"Tracker not available — running detection without tracking: {exc}", file=sys.stderr)
        model.predict(
            source=source,
            show=args.show,
            save=True,
            conf=args.conf,
            classes=[0, 1, 2],
            device="cpu",
        )

    print(f"Done. Results saved under: {root / 'runs'}")


if __name__ == "__main__":
    main()