import cv2
from ultralytics import YOLO


class WaybillReader:
    def __init__(
        self,
        waybill_model_path: str,
        read_model_path: str,
        waybill_conf: float = 0.45,
        digit_conf: float = 0.25,
        roi_pad: int = 0,
        expected_length: int | None = 10,
    ):
        self.waybill_model = YOLO(waybill_model_path)
        self.read_model = YOLO(read_model_path)

        self.waybill_conf = waybill_conf
        self.digit_conf = digit_conf
        self.roi_pad = roi_pad
        self.expected_length = expected_length

    @staticmethod
    def clamp(val: int, low: int, high: int) -> int:
        return max(low, min(val, high))

    def crop_with_padding(self, frame, x1, y1, x2, y2, pad=0):
        h, w = frame.shape[:2]
        x1 = self.clamp(int(x1) - pad, 0, w - 1)
        y1 = self.clamp(int(y1) - pad, 0, h - 1)
        x2 = self.clamp(int(x2) + pad, 0, w - 1)
        y2 = self.clamp(int(y2) + pad, 0, h - 1)

        if x2 <= x1 or y2 <= y1:
            return None, x1, y1, x2, y2

        roi = frame[y1:y2, x1:x2]
        return roi, x1, y1, x2, y2

    def read_digits_from_roi(self, roi):
        results = self.read_model.predict(
            source=roi,
            conf=self.digit_conf,
            verbose=False,
        )

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

                if isinstance(self.read_model.names, dict):
                    digit_text = str(self.read_model.names.get(cls_id, cls_id))
                else:
                    digit_text = (
                        str(self.read_model.names[cls_id])
                        if cls_id < len(self.read_model.names)
                        else str(cls_id)
                    )

                detections.append(
                    (float(x1), float(y1), float(x2), float(y2), digit_text, score)
                )

        detections.sort(key=lambda d: d[0])
        number_str = "".join(d[4] for d in detections)
        return number_str, detections

    def is_valid_number(self, number_str: str) -> bool:
        if not number_str:
            return False
        if not number_str.isdigit():
            return False
        if self.expected_length is not None and len(number_str) != self.expected_length:
            return False
        return True

    @staticmethod
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

    def process_frame(self, frame):
        """
        Returns:
            {
                "frame": annotated_frame,
                "number": best_number_or_none,
                "valid": bool,
                "found": bool,
                "roi": best_roi_or_none
            }
        """
        display = frame.copy()
        roi_preview = None

        best_number = None
        best_valid = False
        best_score = -1.0
        found_any = False

        results = self.waybill_model.predict(
            source=frame,
            conf=self.waybill_conf,
            verbose=False,
        )

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                wb_score = float(confs[i])

                roi, rx1, ry1, rx2, ry2 = self.crop_with_padding(
                    frame, x1, y1, x2, y2, self.roi_pad
                )
                if roi is None or roi.size == 0:
                    continue

                found_any = True

                number_str, digit_dets = self.read_digits_from_roi(roi)
                valid = self.is_valid_number(number_str)

                color = (0, 255, 0) if valid else (0, 165, 255)

                cv2.rectangle(display, (rx1, ry1), (rx2, ry2), color, 2)
                self.draw_digit_boxes_on_frame(display, rx1, ry1, digit_dets)

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

                if wb_score > best_score:
                    best_score = wb_score
                    best_number = number_str if number_str else None
                    best_valid = valid
                    roi_preview = roi.copy()

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

        return {
            "frame": display,
            "number": best_number,
            "valid": best_valid,
            "found": found_any,
            "roi": roi_preview,
        }