from pyzbar.pyzbar import decode
import cv2


class BarcodeReader:
    def __init__(self):
        pass

    def read(self, frame):
        if frame is None:
            return None, None

        # Try several variants for better barcode detection
        variants = self._build_variants(frame)

        found = []
        for img in variants:
            barcodes = decode(img)
            for b in barcodes:
                try:
                    text = b.data.decode("utf-8").strip()
                except Exception:
                    text = str(b.data)

                x, y, w, h = b.rect
                if text:
                    found.append((text, (x, y, w, h)))

            if found:
                break

        if not found:
            return None, None

        # Pick the largest barcode box
        best = max(found, key=lambda item: item[1][2] * item[1][3])
        return best

    def _build_variants(self, frame):
        variants = [frame]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variants.append(gray)

        # binary threshold
        _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(th1)

        # inverse threshold
        _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        variants.append(th2)

        return variants