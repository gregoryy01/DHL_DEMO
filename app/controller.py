import time
import cv2
from tkinter import filedialog

from app.storage.csv_logger import CSVLogger
from app.storage.image_saver import ImageSaver
from app.detection.barcode_reader import BarcodeReader
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

        self.waybill_reader = WaybillReader(
            waybill_model_path="models/detection/waybill_good.pt",
            read_model_path="models/recognize/read100.pt",
            waybill_conf=0.45,
            digit_conf=0.25,
            roi_pad=0,
            expected_length=10,
        )

        self.last_frame_time = time.time()
        self.fps = 0.0

        self.last_logged_barcode = None
        self.last_logged_ocr = None

        self.last_result = {
            "barcode": "BARCODE NOT FOUND",
            "ocr": "OCR NOT FOUND",
            "barcode_bbox": None,
            "barcode_candidates": [],
            "waybill_bbox": None,
            "waybill_conf": None,
            "raw_texts": [],
            "roi": None,
            "waybill_found": False,
            "ocr_valid": False,
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
        self.gui.update_status("IMAGE OCR DONE", "lightgreen")

        self._save_and_log_if_useful(processed_frame, result)

    # ----------------------------
    # Core pipeline
    # ----------------------------
    def process_frame(self, frame):
        result = {
            "barcode": "BARCODE NOT FOUND",
            "ocr": "OCR NOT FOUND",
            "barcode_bbox": None,
            "barcode_candidates": [],
            "waybill_bbox": None,
            "waybill_conf": None,
            "raw_texts": [],
            "roi": None,
            "waybill_found": False,
            "ocr_valid": False,
        }

        # 1) Waybill detection + digit reading first
        try:
            wb = self.waybill_reader.process_frame(frame)
        except Exception as e:
            print(f"[WaybillReader Error] {e}")
            wb = {
                "frame": frame.copy(),
                "number": None,
                "valid": False,
                "found": False,
                "roi": None,
            }

        annotated_frame = wb["frame"]
        result["waybill_found"] = bool(wb.get("found", False))
        result["ocr_valid"] = bool(wb.get("valid", False))
        result["roi"] = wb.get("roi", None)

        if wb.get("number"):
            result["ocr"] = wb["number"]

        # 2) Barcode reading using OCR as preference
        preferred_text = result["ocr"] if result["ocr"] != "OCR NOT FOUND" else None

        barcode_data, barcode_bbox, barcode_candidates = self._safe_read_barcode(
            frame,
            preferred_text=preferred_text,
            expected_digits_len=10,
            return_all=True,
        )

        if barcode_data:
            result["barcode"] = barcode_data
            result["barcode_bbox"] = barcode_bbox

        result["barcode_candidates"] = barcode_candidates

        # 3) Draw barcode overlay on top of annotated waybill frame
        self._draw_barcode_overlay(annotated_frame, result)

        # 4) Build raw texts for GUI
        if result["barcode"] != "BARCODE NOT FOUND":
            result["raw_texts"].append(f"BARCODE: {result['barcode']}")
        else:
            result["raw_texts"].append("BARCODE: NOT FOUND")

        if result["waybill_found"]:
            if result["ocr"] != "OCR NOT FOUND":
                result["raw_texts"].append(f"OCR: {result['ocr']}")
            else:
                result["raw_texts"].append("OCR: WAYBILL FOUND BUT NO READ")
        else:
            result["raw_texts"].append("OCR: NO WAYBILL DETECTED")

        if barcode_candidates:
            result["raw_texts"].append("BARCODE CANDIDATES:")
            for c in barcode_candidates[:5]:
                result["raw_texts"].append(
                    f"- {c['data']} | score={c['score']} | digits={c['digits']}"
                )

        self._draw_fps(annotated_frame)
        return annotated_frame, result

    # ----------------------------
    # Helpers
    # ----------------------------
    def _safe_read_barcode(self, frame, preferred_text=None, expected_digits_len=10, return_all=False):
        try:
            return self.barcode_reader.read(
                frame,
                preferred_text=preferred_text,
                expected_digits_len=expected_digits_len,
                return_all=return_all,
            )
        except Exception as e:
            print(f"[BarcodeReader Error] {e}")
            if return_all:
                return None, None, []
            return None, None

    def _draw_barcode_overlay(self, frame, result):
        if result["barcode_bbox"] is None:
            return

        x, y, w, h = result["barcode_bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"BARCODE: {result['barcode']}",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
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
            2,
        )