import cv2
from ultralytics import YOLO


MODEL_PATH = "C:/Users/lenovo/Documents/GitHub/DHL_DEMO/models/detection/waybill_good.pt"   # change if needed
CAMERA_INDEX = 0
CONF_THRESHOLD = 0.4


def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Failed to open camera index {CAMERA_INDEX}")
        return

    print("Camera opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)

        annotated_frame = frame.copy()

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())

                class_name = model.names.get(cls_id, str(cls_id))

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    f"{class_name} {conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("YOLO Camera Test", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()