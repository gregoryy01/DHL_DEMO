import cv2
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
IMAGE_PATH = r"C:/Users/lenovo/Documents/GitHub/DHL_DEMO/ocrread1.png"
MODEL_PATH = r"C:/Users/lenovo/Documents/GitHub/DHL_DEMO/models/recognize/read_good.pt"
OUTPUT_PATH = r"C:/Users/lenovo/Documents/GitHub/DHL_DEMO/data/results/read_test_result.jpg"

CONF_THRESHOLD = 0.25
EXPECTED_LENGTH = 10   # set to None if you do not want length checking


def is_valid_number(number_str, expected_length=None):
    if not number_str:
        return False
    if not number_str.isdigit():
        return False
    if expected_length is not None and len(number_str) != expected_length:
        return False
    return True


def main():
    model = YOLO(MODEL_PATH)

    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Failed to load image: {IMAGE_PATH}")
        return

    display = image.copy()

    results = model.predict(source=image, conf=CONF_THRESHOLD, verbose=False)

    detections = []

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy()

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            score = float(confs[i])
            cls_id = int(clss[i])

            if isinstance(model.names, dict):
                digit_text = str(model.names.get(cls_id, cls_id))
            else:
                digit_text = str(model.names[cls_id]) if cls_id < len(model.names) else str(cls_id)

            detections.append((float(x1), float(y1), float(x2), float(y2), digit_text, score))

    # sort left to right
    detections.sort(key=lambda d: d[0])

    number_str = "".join(d[4] for d in detections)
    valid = is_valid_number(number_str, EXPECTED_LENGTH)

    # draw digit boxes
    for x1, y1, x2, y2, digit_text, score in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(
            display,
            f"{digit_text}:{score:.2f}",
            (x1, max(y1 - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # draw final prediction
    color = (0, 255, 0) if valid else (0, 165, 255)
    label = f"READ: {number_str if number_str else 'NO READ'}"
    cv2.putText(
        display,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    print("Predicted number:", number_str if number_str else "NO READ")
    print("Valid:", valid)
    print("Detections:")
    for det in detections:
        print(det)

    cv2.imwrite(OUTPUT_PATH, display)
    print(f"Saved output to: {OUTPUT_PATH}")

    cv2.imshow("Read Model Test", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()