# YOLOv11 Real-Time Object Detection

A concise, practical webcam object detector using Ultralytics YOLOv11 and OpenCV.

## Prerequisites
- Python 3.8+
- A working webcam

## Setup
```bash
python -m venv my_env
source my_env/bin/activate  # Windows: my_env\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python realtime_object_detection.py
```
- A window opens showing detections with bounding boxes, class names, confidence, and FPS (top-left).
- Press `q` to quit.

## Notes
- The script uses the YOLOv11 nano detection model `yolo11n.pt` (downloads automatically on first run).
- If you have multiple cameras, try changing the index in `cv2.VideoCapture(0)` to `1`, `2`, etc.
- For performance on CPU-only machines, consider the nano/small models. For better accuracy, try larger models.
- More info: Ultralytics docs: https://docs.ultralytics.com/quickstart/
