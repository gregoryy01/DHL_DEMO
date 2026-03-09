import cv2
from pyzbar.pyzbar import decode


VALID_LENGTHS = {10, 11, 16}


def extract_waybill(text: str):
    if not text:
        return None

    # keep digits only
    digits = "".join(ch for ch in text if ch.isdigit())

    # exact match
    if len(digits) in VALID_LENGTHS:
        return digits

    # search inside longer digit string
    for length in sorted(VALID_LENGTHS):
        for i in range(0, len(digits) - length + 1):
            sub = digits[i:i + length]
            if len(sub) in VALID_LENGTHS:
                return sub

    return None


def detect_waybills(frame):
    results = decode(frame)
    found = []

    for r in results:
        try:
            raw_text = r.data.decode("utf-8").strip()
        except Exception:
            continue

        waybill = extract_waybill(raw_text)
        if not waybill:
            continue

        x, y, w, h = r.rect.left, r.rect.top, r.rect.width, r.rect.height

        found.append({
            "waybill": waybill,
            "raw_text": raw_text,
            "type": r.type,
            "bbox": (x, y, w, h),
        })

    return found


def draw_results(frame, results):
    out = frame.copy()

    for item in results:
        x, y, w, h = item["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

        label = f"{item['waybill']} [{item['type']}]"
        cv2.putText(
            out,
            label,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    return out


def test_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    results = detect_waybills(frame)

    print("=== IMAGE RESULTS ===")
    if not results:
        print("No valid waybill barcode found")
    else:
        for i, item in enumerate(results, 1):
            print(f"[{i}] waybill   : {item['waybill']}")
            print(f"    raw_text  : {item['raw_text']}")
            print(f"    type      : {item['type']}")
            print(f"    bbox      : {item['bbox']}")

    vis = draw_results(frame, results)
    cv2.imshow("pyzbar waybill test", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    last_printed = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        results = detect_waybills(frame)
        vis = draw_results(frame, results)

        if results:
            best = results[0]
            if best["waybill"] != last_printed:
                last_printed = best["waybill"]
                print("Detected:")
                print("  waybill  :", best["waybill"])
                print("  raw_text :", best["raw_text"])
                print("  type     :", best["type"])
                print("  bbox     :", best["bbox"])

        cv2.imshow("pyzbar live test", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # ===== choose one =====

    # 1) Test with image
    #test_image("waybillex.jpg")

    # 2) Test with camera
    test_camera(0)