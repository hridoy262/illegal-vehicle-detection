# Bangladesh Vehicle Detection

Detect and track vehicles in video using an Ultralytics YOLO model trained on the Bangladesh vehicle dataset.

## Setup

Use Python 3.10 or newer. Create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the YOLOv8n model file (`yolov8n.pt`) in the project root. Ultralytics also provides this pretrained model through its model download workflow.

## Dataset

The training configuration expects the YOLO-format dataset under `train/`, `valid/`, and `test/`, each with `images/` and `labels/` subdirectories. The dataset is [Vehicle Detection In Bangladesh v1](https://universe.roboflow.com/united-international-university-qbzdi/vehicle-detection-in-bangladesh/dataset/1), published under CC BY 4.0. Download the YOLOv8 export separately and arrange the extracted files to match those paths; dataset files are intentionally not tracked in this repository.

The dataset has 15 classes: bicycle, bike, boat, bus, car, cng, easybike, horsecart, launch, leguna, rickshaw, tractor, truck, van, and wheelbarrow. It does not define a separate `pickup` class.

## Train

For a short CPU trial that uses a small portion of the dataset:

```powershell
python train.py --epochs 2 --imgsz 416 --fraction 0.1 --batch 8 --name cpu-trial
```

This trial is only for checking that training works; it is not expected to produce the best detector. For better accuracy, train on the full dataset for more epochs, preferably on a CUDA GPU:

```powershell
python train.py --epochs 50 --fraction 1.0 --imgsz 640 --device 0
```

The script automatically selects an available device if `--device` is omitted; use `--device cpu` to force CPU training. Other options include `--batch`, `--imgsz`, `--fraction`, and `--name`. Training results are written under `runs/detect/<name>/` and are not tracked by Git. Full-dataset CPU training may take many hours.

## Detect or track a video

Pass a video path explicitly:

```powershell
python detect.py --source path\to\video.mp4 --no-show
```

Use `--show` to display the preview window. If `runs/detect/bangladesh-vehicles/weights/best.pt` exists, the script uses the trained 15-class model; otherwise it falls back to the generic `yolov8n.pt` model. Inference outputs are saved under `runs/`.