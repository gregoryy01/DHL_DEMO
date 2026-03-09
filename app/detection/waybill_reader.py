import re
import cv2
import easyocr
import numpy as np


class WaybillReader:
    def __init__(self, langs=None, min_text_conf=0.25):
        if langs is None:
            langs = ["en"]

        self.reader = easyocr.Reader(langs, gpu=False)
        self.min_text_conf = min_text_conf

    def preprocess(self, roi):
        """
        Preprocess cropped ROI before OCR.
        """
        if roi is None or roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # enlarge small text regions
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        # reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # adaptive threshold often helps printed labels
        proc = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10
        )
        return proc

    def clean_text(self, text):
        """
        Keep only alphanumeric text, remove spaces/symbol noise.
        """
        if not text:
            return ""

        text = text.upper().strip()
        text = re.sub(r"[^A-Z0-9]", "", text)
        return text

    def read(self, roi):
        """
        Returns best OCR text from ROI.
        """
        proc = self.preprocess(roi)
        if proc is None:
            return ""

        results = self.reader.readtext(proc)

        if not results:
            return ""

        candidates = []
        for item in results:
            # EasyOCR returns: (bbox, text, confidence)
            _, text, conf = item
            if conf >= self.min_text_conf:
                cleaned = self.clean_text(text)
                if cleaned:
                    candidates.append((cleaned, conf))

        if not candidates:
            return ""

        # sort by confidence descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[0][0]