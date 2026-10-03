# Vehicle Detection

Vehicle detection and tracking in video using Ultralytics YOLO.

## Setup

Use Python 3.10 or newer. Create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the YOLOv8n model file (`yolov8n.pt`) in the project root. Ultralytics also provides this pretrained model through its model download workflow.

## Dataset

The training configuration expects the YOLO-format dataset under `train/`, `valid/`, and `test/`, each with `images/` and `labels/` subdirectories. The configured dataset source is [Illegal Parking Detection v2](https://universe.roboflow.com/jiawengan/illegal-parking-ltiwv/dataset/2), published under CC BY 4.0. Download it separately and arrange the extracted files to match those paths; dataset files are intentionally not tracked in this repository.

## Train

With the dataset in place, run:

```powershell
python train.py
```

Training results are written under `runs/detect/train/` and are not tracked by Git.

## Detect or track a video

Pass a video path explicitly:

```powershell
python detect.py --source path\to\video.mp4 --no-show
```

Use `--show` to display the preview window. If `runs/detect/train/weights/best.pt` exists, the script uses that trained model; otherwise it uses `yolov8n.pt`. Inference outputs are saved under `runs/`.