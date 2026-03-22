import re
import cv2
from pyzbar.pyzbar import decode


class BarcodeReader:
    def __init__(self):
        pass

    def read(self, frame, preferred_text=None, expected_digits_len=10, return_all=False):
        """
        Best barcode picker.

        Args:
            frame: OpenCV image
            preferred_text: OCR result / expected waybill text, e.g. "1149007930"
            expected_digits_len: preferred digit length for waybill barcode
            return_all: if True, also return sorted candidates

        Returns:
            if return_all=False:
                (best_text, best_bbox)

            if return_all=True:
                (best_text, best_bbox, candidates)
        """
        candidates = self.read_all(
            frame,
            preferred_text=preferred_text,
            expected_digits_len=expected_digits_len,
        )

        if not candidates:
            if return_all:
                return None, None, []
            return None, None

        best = candidates[0]

        if return_all:
            return best["data"], best["bbox"], candidates
        return best["data"], best["bbox"]

    def read_all(self, frame, preferred_text=None, expected_digits_len=10):
        """
        Returns all barcode candidates sorted from best to worst.
        """
        if frame is None:
            return []

        h, w = frame.shape[:2]
        preferred_digits = self._only_digits(preferred_text)

        variants = self._build_variants(frame)

        # Deduplicate by raw decoded text
        merged = {}

        for variant_name, img in variants:
            try:
                barcodes = decode(img)
            except Exception as e:
                print(f"[Barcode decode error on {variant_name}] {e}")
                continue

            for b in barcodes:
                raw_text = self._safe_decode(b.data)
                if not raw_text:
                    continue

                x, y, bw, bh = b.rect
                bbox = (int(x), int(y), int(bw), int(bh))

                candidate = self._build_candidate(
                    raw_text=raw_text,
                    bbox=bbox,
                    barcode_type=getattr(b, "type", "UNKNOWN"),
                    preferred_digits=preferred_digits,
                    expected_digits_len=expected_digits_len,
                    frame_w=w,
                    frame_h=h,
                    variant_name=variant_name,
                )

                key = raw_text.strip()

                # Keep the best-scoring version of the same decoded text
                if key not in merged or candidate["score"] > merged[key]["score"]:
                    merged[key] = candidate

        candidates = sorted(merged.values(), key=lambda c: c["score"], reverse=True)
        return candidates

    def _build_candidate(
        self,
        raw_text,
        bbox,
        barcode_type,
        preferred_digits,
        expected_digits_len,
        frame_w,
        frame_h,
        variant_name,
    ):
        x, y, w, h = bbox
        digits = self._only_digits(raw_text)

        score = 0
        reasons = []

        # Strongest rule: exact digit match with OCR
        if preferred_digits and digits == preferred_digits:
            score += 1000
            reasons.append("exact OCR digit match")

        # Slightly weaker: raw text exact match with OCR text
        if preferred_digits and raw_text.strip() == preferred_digits:
            score += 200
            reasons.append("exact OCR raw match")

        # Prefer expected numeric length, e.g. 10 digits
        if expected_digits_len is not None:
            if len(digits) == expected_digits_len:
                score += 300
                reasons.append(f"{expected_digits_len} digits")
            else:
                diff = abs(len(digits) - expected_digits_len)
                penalty = min(diff * 40, 200)
                score -= penalty
                reasons.append(f"length penalty -{penalty}")

        # Prefer pure numeric raw barcode
        if raw_text.isdigit():
            score += 150
            reasons.append("raw is numeric")

        # Prefer digit-only content after cleanup
        if digits and digits == raw_text:
            score += 80
            reasons.append("clean digit-only text")

        # Penalize letters
        if any(ch.isalpha() for ch in raw_text):
            score -= 200
            reasons.append("contains letters")

        # Penalize special symbols often seen in shipment barcodes
        if "+" in raw_text:
            score -= 120
            reasons.append("contains +")
        if "-" in raw_text:
            score -= 40
            reasons.append("contains -")
        if "/" in raw_text:
            score -= 40
            reasons.append("contains /")

        # Position penalty: lower barcodes are less likely to be the waybill one
        # Small penalty only, so this helps but does not dominate
        y_ratio = y / max(frame_h, 1)
        pos_penalty = int(y_ratio * 120)
        score -= pos_penalty
        reasons.append(f"vertical penalty -{pos_penalty}")

        # Area penalty: very large barcode blocks are often the wrong barcode here
        area = w * h
        frame_area = max(frame_w * frame_h, 1)
        area_ratio = area / frame_area
        area_penalty = int(area_ratio * 250)
        score -= area_penalty
        reasons.append(f"area penalty -{area_penalty}")

        return {
            "data": raw_text,
            "digits": digits,
            "bbox": bbox,
            "type": barcode_type,
            "score": score,
            "variant": variant_name,
            "reasons": reasons,
        }

    def _build_variants(self, frame):
        variants = [("original", frame)]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variants.append(("gray", gray))

        # Binary threshold
        _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("thresh", th1))

        # Inverse threshold
        _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        variants.append(("thresh_inv", th2))

        return variants

    def _safe_decode(self, data):
        try:
            return data.decode("utf-8", errors="ignore").strip()
        except Exception:
            try:
                return str(data).strip()
            except Exception:
                return ""

    def _only_digits(self, text):
        if not text:
            return ""
        return re.sub(r"\D", "", str(text))