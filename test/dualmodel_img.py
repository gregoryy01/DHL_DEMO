import cv2
from ultralytics import YOLO


# =========================
# CONFIG
# =========================
IMAGE_PATH = r"C:/Users/lenovo/Documents/GitHub/DHL_DEMO/waybill8.png"
WAYBILL_MODEL_PATH = r"C:/Users/lenovo/Documents/GitHub/DHL_DEMO/models/detection/waybill_good.pt"
READ_MODEL_PATH = r"C:/Users/lenovo/Documents/GitHub/DHL_DEMO/models/recognize/read100.pt"
OUTPUT_PATH = r"C:/Users/lenovo/Documents/GitHub/DHL_DEMO/data/results/result.jpg"

WAYBILL_CONF = 0.45
DIGIT_CONF = 0.25
ROI_PAD = 0
EXPECTED_LENGTH = 10   # set to None if you do not want length checking


# =========================
# LOAD MODELS
# =========================
waybill_model = YOLO(WAYBILL_MODEL_PATH)
read_model = YOLO(READ_MODEL_PATH)


def clamp(val, low, high):
    return max(low, min(val, high))


def crop_with_padding(frame, x1, y1, x2, y2, pad=0):
    h, w = frame.shape[:2]
    x1 = clamp(int(x1) - pad, 0, w - 1)
    y1 = clamp(int(y1) - pad, 0, h - 1)
    x2 = clamp(int(x2) + pad, 0, w - 1)
    y2 = clamp(int(y2) + pad, 0, h - 1)

    if x2 <= x1 or y2 <= y1:
        return None, x1, y1, x2, y2

    roi = frame[y1:y2, x1:x2]
    return roi, x1, y1, x2, y2


def read_digits_from_roi(roi, model, conf=0.25):
    results = model.predict(source=roi, conf=conf, verbose=False)

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

    detections.sort(key=lambda d: d[0])
    number_str = "".join(d[4] for d in detections)
    return number_str, detections


def is_valid_number(number_str, expected_length=None):
    if not number_str:
        return False
    if not number_str.isdigit():
        return False
    if expected_length is not None and len(number_str) != expected_length:
        return False
    return True


def draw_digit_boxes_on_frame(frame, roi_origin_x, roi_origin_y, digit_dets):
    for x1, y1, x2, y2, digit_text, score in digit_dets:
        gx1 = int(roi_origin_x + x1)
        gy1 = int(roi_origin_y + y1)
        gx2 = int(roi_origin_x + x2)
        gy2 = int(roi_origin_y + y2)

        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (255, 0, 0), 1)
        cv2.putText(
            frame,
            digit_text,
            (gx1, max(gy1 - 3, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )


def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Failed to load image: {IMAGE_PATH}")
        return

    display = image.copy()
    roi_preview = None

    results = waybill_model.predict(source=image, conf=WAYBILL_CONF, verbose=False)

    found_any = False

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            wb_score = float(confs[i])

            roi, rx1, ry1, rx2, ry2 = crop_with_padding(image, x1, y1, x2, y2, ROI_PAD)
            if roi is None or roi.size == 0:
                continue

            found_any = True
            roi_preview = roi.copy()

            number_str, digit_dets = read_digits_from_roi(roi, read_model, DIGIT_CONF)
            valid = is_valid_number(number_str, EXPECTED_LENGTH)

            color = (0, 255, 0) if valid else (0, 165, 255)

            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), color, 2)
            draw_digit_boxes_on_frame(display, rx1, ry1, digit_dets)

            if number_str:
                label = f"{number_str} | wb:{wb_score:.2f}"
            else:
                label = f"NO READ | wb:{wb_score:.2f}"

            cv2.putText(
                display,
                label,
                (rx1, max(ry1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

            print("Detected number:", number_str if number_str else "NO READ")

    if not found_any:
        cv2.putText(
            display,
            "No waybill detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        print("No waybill detected.")

    cv2.imwrite(OUTPUT_PATH, display)
    print(f"Saved result image to: {OUTPUT_PATH}")

    cv2.imshow("Waybill + Read Model", display)
    if roi_preview is not None:
        cv2.imshow("Detected ROI", roi_preview)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()