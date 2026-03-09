import time
from pathlib import Path
import cv2
from tkinter import filedialog

from app.storage.csv_logger import CSVLogger
from app.storage.image_saver import ImageSaver
from app.detection.barcode_reader import BarcodeReader
from app.detection.waybill_detector import WaybillDetector
from app.detection.waybill_reader import WaybillReader


class AppController:
    def __init__(self, gui):
        self.gui = gui

        self.cap = None
        self.current_frame = None
        self.current_camera_index = 0
        self.camera_running = False

        self.csv_logger = CSVLogger()
        self.image_saver = ImageSaver()
        self.barcode_reader = BarcodeReader()
        self.waybill_detector = WaybillDetector(
            model_path="models/detection/waybill_good.pt",
            conf=0.40
        )
        self.waybill_reader = WaybillReader()

        self.last_frame_time = time.time()
        self.fps = 0.0

        self.last_logged_barcode = None
        self.last_logged_ocr = None

        self.last_result = {
            "barcode": "BARCODE NOT FOUND",
            "ocr": "OCR NOT FOUND",
            "barcode_bbox": None,
            "waybill_bbox": None,
            "waybill_conf": None,
            "raw_texts": []
        }

    # ----------------------------
    # Camera management
    # ----------------------------
    def scan_cameras(self, max_index=5):
        camera_dict = {}

        for idx in range(max_index + 1):
            cap = cv2.VideoCapture(idx)
            if cap is not None and cap.isOpened():
                camera_dict[f"Camera {idx}"] = idx
                cap.release()

        self.gui.set_camera_options(camera_dict)

        if camera_dict:
            self.gui.update_status("CAMERAS FOUND", "lightgreen")
        else:
            self.gui.update_status("NO CAMERA", "tomato")

    def open_camera(self):
        # compatibility button from UI
        self.start_camera()

    def start_camera(self):
        self.stop_camera()

        camera_index = self.gui.get_selected_camera_index()
        self.current_camera_index = camera_index

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.cap = None
            self.camera_running = False
            self.gui.update_status("CAMERA FAILED", "tomato")
            return

        self.camera_running = True
        self.last_frame_time = time.time()
        self.gui.update_status(f"CAMERA {camera_index} ON", "lightgreen")

    def close_camera(self):
        # compatibility alias
        self.stop_camera()

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.camera_running = False
        self.gui.update_status("CAMERA OFF", "lightgray")
        self.gui.update_fps(0.0)

    # ----------------------------
    # Main loop
    # ----------------------------
    def loop(self):
        if not self.camera_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.gui.update_status("FRAME READ FAIL", "tomato")
            return

        self.current_frame = frame.copy()

        # live preview only, no OCR trigger here
        preview = frame.copy()
        self._update_fps()
        self._draw_fps(preview)

        self.gui.show_image(preview)
        self.gui.update_fps(self.fps)

    # ----------------------------
    # Trigger/manual inspection
    # ----------------------------
    def trigger_inspection(self):
        if self.current_frame is None:
            self.gui.update_status("NO FRAME", "tomato")
            return

        frame = self.current_frame.copy()
        processed_frame, result = self.process_frame(frame)

        self.last_result = result

        self.gui.show_image(processed_frame)
        self.gui.update_fps(self.fps)
        self._push_result_to_gui(result)
        self.gui.update_status("READ DONE", "lightgreen")

        self._save_and_log_if_useful(processed_frame, result)

    def run_image_ocr(self):
        file_path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return

        frame = cv2.imread(file_path)
        if frame is None:
            self.gui.update_status("IMAGE LOAD FAIL", "tomato")
            return

        self.current_frame = frame.copy()

        processed_frame, result = self.process_frame(frame)
        self.last_result = result

        self.gui.show_image(processed_frame)
        self._push_result_to_gui(result)
        self.gui.update_status(f"IMAGE OCR DONE", "lightgreen")

        self._save_and_log_if_useful(processed_frame, result)

    # ----------------------------
    # Core pipeline
    # ----------------------------
    def process_frame(self, frame):
        result = {
            "barcode": "BARCODE NOT FOUND",
            "ocr": "OCR NOT FOUND",
            "barcode_bbox": None,
            "waybill_bbox": None,
            "waybill_conf": None,
            "raw_texts": []
        }

        # barcode
        barcode_data, barcode_bbox = self._safe_read_barcode(frame)
        if barcode_data:
            result["barcode"] = barcode_data
            result["barcode_bbox"] = barcode_bbox
            result["raw_texts"].append(f"BARCODE: {barcode_data}")

        # YOLO waybill detection
        detections = self._safe_detect_waybill(frame)
        best_detection = self._select_best_detection(detections)

        if best_detection is not None:
            x1, y1, x2, y2 = best_detection["bbox"]
            result["waybill_bbox"] = (x1, y1, x2, y2)
            result["waybill_conf"] = best_detection["confidence"]

            roi = self._crop_roi(frame, x1, y1, x2, y2)
            if roi is not None:
                ocr_text = self.waybill_reader.read(roi)
                if ocr_text:
                    result["ocr"] = ocr_text
                result["raw_texts"].append(f"OCR: {result['ocr']}")

        if not result["raw_texts"]:
            result["raw_texts"].append("NO RESULT")

        self._draw_result_overlay(frame, result)
        self._draw_fps(frame)

        return frame, result

    # ----------------------------
    # Helpers
    # ----------------------------
    def _safe_read_barcode(self, frame):
        try:
            return self.barcode_reader.read(frame)
        except Exception as e:
            print(f"[BarcodeReader Error] {e}")
            return None, None

    def _safe_detect_waybill(self, frame):
        try:
            return self.waybill_detector.detect(frame)
        except Exception as e:
            print(f"[WaybillDetector Error] {e}")
            return []

    def _select_best_detection(self, detections):
        if not detections:
            return None
        return max(detections, key=lambda d: d["confidence"])

    def _crop_roi(self, frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame[y1:y2, x1:x2]
        if roi is None or roi.size == 0:
            return None

        return roi

    def _draw_result_overlay(self, frame, result):
        # barcode bbox
        if result["barcode_bbox"] is not None:
            x, y, w, h = result["barcode_bbox"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"BARCODE: {result['barcode']}",
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        # waybill bbox
        if result["waybill_bbox"] is not None:
            x1, y1, x2, y2 = result["waybill_bbox"]
            conf = result["waybill_conf"] if result["waybill_conf"] is not None else 0.0

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"WAYBILL {conf:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"OCR: {result['ocr']}",
                (x1, min(frame.shape[0] - 10, y2 + 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    def _push_result_to_gui(self, result):
        self.gui.update_waybill_list([result["ocr"]])
        self.gui.update_barcode_list([result["barcode"]])
        self.gui.update_raw_texts(result["raw_texts"])

    def _save_and_log_if_useful(self, frame, result):
        barcode = result["barcode"]
        ocr = result["ocr"]

        useful_barcode = barcode != "BARCODE NOT FOUND"
        useful_ocr = ocr != "OCR NOT FOUND"

        if not useful_barcode and not useful_ocr:
            return

        if barcode == self.last_logged_barcode and ocr == self.last_logged_ocr:
            return

        self.last_logged_barcode = barcode
        self.last_logged_ocr = ocr

        saved_path = ""
        try:
            saved_path = self.image_saver.save(frame, barcode, ocr)
        except Exception as e:
            print(f"[ImageSaver Error] {e}")

        try:
            self.csv_logger.log(barcode, ocr, saved_path)
        except Exception as e:
            print(f"[CSVLogger Error] {e}")

    def _update_fps(self):
        now = time.time()
        dt = now - self.last_frame_time
        if dt > 0:
            self.fps = 1.0 / dt
        self.last_frame_time = now

    def _draw_fps(self, frame):
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )