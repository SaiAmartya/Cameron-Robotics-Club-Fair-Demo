import cv2
import time
from ultralytics import YOLO


def main() -> None:
    """Run YOLOv11 on the default webcam and display detections."""
    # Use YOLOv11 nano model for object detection per Ultralytics supported tasks
    model = YOLO("yolo11n.pt")  # downloads weights on first run if missing

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    window_name = "YOLOv11 Real-Time Object Detection (press 'q' to quit)"

    prev_time = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if prev_time else 0.0
            prev_time = current_time

            results = model(frame, verbose=False)  # infer on BGR frame
            annotated = results[0].plot()  # draws boxes with labels and confidences

            # Draw FPS at top-left
            cv2.putText(
                annotated,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


