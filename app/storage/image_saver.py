from pathlib import Path
from datetime import datetime
import cv2
import re


class ImageSaver:
    def __init__(self, base_dir="data/captures"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _clean_text(self, text, max_len=30):
        if not text:
            return "none"

        text = str(text).strip().upper()
        text = re.sub(r"[^A-Z0-9_-]", "", text)

        if not text:
            return "none"

        return text[:max_len]

    def save(self, frame, barcode_text="", ocr_text=""):
        now = datetime.now()

        day_folder = self.base_dir / now.strftime("%Y-%m-%d")
        day_folder.mkdir(parents=True, exist_ok=True)

        barcode_part = self._clean_text(barcode_text)
        ocr_part = self._clean_text(ocr_text)

        filename = f"img_{now.strftime('%Y%m%d_%H%M%S')}_{barcode_part}_{ocr_part}.jpg"
        path = day_folder / filename

        ok = cv2.imwrite(str(path), frame)
        if not ok:
            raise RuntimeError(f"Failed to save image: {path}")

        return str(path)