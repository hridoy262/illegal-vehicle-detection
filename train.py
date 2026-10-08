import argparse
from pathlib import Path

from ultralytics import YOLO

if __name__ == '__main__':
    ROOT = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Train the Bangladesh vehicle detector.')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum training epochs.')
    parser.add_argument('--batch', type=int, default=8, help='Training batch size.')
    parser.add_argument('--imgsz', type=int, default=640, help='Training image size.')
    parser.add_argument('--fraction', type=float, default=1.0, help='Fraction of the training set to use.')
    parser.add_argument('--name', default='bangladesh-vehicles', help='Training run name.')
    parser.add_argument('--device', default=None, help='Device, e.g. 0 for GPU or cpu. Defaults to auto.')
    args = parser.parse_args()

    model = YOLO(str(ROOT / 'yolov8n.pt'))

    model.train(
        data=str(ROOT / 'data.yaml'),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=2,
        device=args.device,
        project=str(ROOT / 'runs' / 'detect'),
        name=args.name,
        exist_ok=True,
        val=True,
        fraction=args.fraction,
    )