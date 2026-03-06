import cv2


class AppController:
    def __init__(self, gui):
        self.gui = gui
        self.cap = None
        self.current_frame = None

    def open_image(self):
        self.gui.update_status("OPEN IMAGE", "lightblue")

    def run_image_ocr(self):
        self.gui.update_status("RUN OCR", "khaki")

        # temporary dummy result
        self.gui.update_waybill_list(["1234567890"])
        self.gui.update_barcode_list(["1234567890"])
        self.gui.update_raw_texts([
            "DHL EXPRESS",
            "WAYBILL: 1234567890"
        ])

    def scan_cameras(self):
        camera_map = {}

        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_map[f"Camera {i}"] = i
                cap.release()

        self.gui.set_camera_options(camera_map)
        self.gui.update_status("SCAN DONE", "lightgreen")

    def start_camera(self):
        cam_index = self.gui.get_selected_camera_index()

        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(cam_index)

        if self.cap.isOpened():
            self.gui.update_status("CAMERA ON", "lightgreen")
        else:
            self.gui.update_status("CAMERA FAIL", "tomato")

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.gui.update_status("CAMERA OFF", "lightgray")

    def trigger_inspection(self):
        if self.current_frame is None:
            self.gui.update_status("NO FRAME", "tomato")
            return

        # later:
        # 1. save image
        # 2. run barcode
        # 3. fallback OCR
        # 4. save CSV
        self.gui.update_status("TRIGGERED", "orange")

        # dummy output for now
        self.gui.update_waybill_list(["WB123456789"])
        self.gui.update_barcode_list(["WB123456789"])
        self.gui.update_raw_texts([
            "WAYBILL NO: WB123456789",
            "BARCODE: WB123456789"
        ])

    def loop(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.gui.show_image(frame)