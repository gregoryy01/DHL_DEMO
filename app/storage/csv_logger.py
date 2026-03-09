import csv
from pathlib import Path
from datetime import datetime


class CSVLogger:
    def __init__(self, csv_path="data/results/waybill_log.csv"):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file()

    def _ensure_file(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "barcode_text",
                    "ocr_text",
                    "image_path",
                    "status"
                ])

    def log(self, barcode_text, ocr_text, image_path=""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = self._build_status(barcode_text, ocr_text)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                barcode_text,
                ocr_text,
                image_path,
                status
            ])

    def _build_status(self, barcode_text, ocr_text):
        barcode_found = bool(barcode_text) and barcode_text != "BARCODE NOT FOUND"
        ocr_found = bool(ocr_text) and ocr_text != "OCR NOT FOUND"

        if barcode_found and ocr_found:
            return "barcode+ocr"
        if barcode_found:
            return "barcode_only"
        if ocr_found:
            return "ocr_only"
        return "no_result"