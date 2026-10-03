from pathlib import Path

from ultralytics import YOLO

if __name__ == '__main__':
    ROOT = Path(__file__).resolve().parent
    model = YOLO(str(ROOT / 'yolov8n.pt'))

    model.train(
        data=str(ROOT / 'data.yaml'),
        epochs=2,
        imgsz=640,
        batch=8,
        workers=2,
        device='cpu',
        project=str(ROOT / 'runs' / 'detect'),
        name='train',
        exist_ok=True,
        val=True,
    )