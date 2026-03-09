import os
import sys
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.detection.waybill_reader import WaybillDetector

img = cv2.imread("waybill1.png")
if img is None:
    raise FileNotFoundError("Could not read test_label.jpg")

detector = WaybillDetector(
    tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

result = detector.detect(img)
print("RESULT:", result)

debug = detector.debug_draw(img, result)
cv2.imshow("Detected", debug)

if result and result.get("text_box") is not None:
    tx, ty, tw, th = result["text_box"]
    text_roi = img[ty:ty + th, tx:tx + tw]
    cv2.imshow("Text Below Barcode", text_roi)

if result and result.get("barcode_box") is not None:
    bx, by, bw, bh = result["barcode_box"]
    barcode_roi = img[by:by + bh, bx:bx + bw]
    cv2.imshow("Chosen Barcode ROI", barcode_roi)

cv2.waitKey(0)
cv2.destroyAllWindows()