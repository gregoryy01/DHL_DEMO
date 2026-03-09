import cv2
from app.detection.ocr_reader import OCRReader
from app.detection.parser import WaybillParser

img = cv2.imread("test_label.jpg")

ocr = OCRReader(tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
parser = WaybillParser()

raw = ocr.read(img)
parsed = parser.parse(raw)

print("RAW:", raw)
print("PARSED:", parsed)