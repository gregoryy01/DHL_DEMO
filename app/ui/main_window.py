import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

from app.controller import AppController
from config.settings import SETTINGS


class OCRGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Waybill Reader System")
        self.root.geometry("1200x700")

        self.controller = AppController(self)

        self.photo_ref = None
        self._camera_map = {}

        self._build_ui()
        self._bind_keys()

        self.controller.scan_cameras()
        self._start_loop()

    def _build_ui(self):
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, pady=5)

        tk.Button(
            top,
            text="Open Camera",
            width=15,
            command=self.controller.open_camera
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            top,
            text="Run OCR Image",
            width=15,
            command=self.controller.run_image_ocr
        ).pack(side=tk.LEFT, padx=5)

        self.camera_var = tk.StringVar()
        self.camera_dropdown = ttk.Combobox(
            top,
            textvariable=self.camera_var,
            width=30,
            state="readonly"
        )
        self.camera_dropdown.pack(side=tk.LEFT, padx=5)

        tk.Button(
            top,
            text="Scan Camera",
            width=12,
            command=self.controller.scan_cameras
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            top,
            text="Start Camera",
            width=15,
            command=self.controller.start_camera
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            top,
            text="Stop Camera",
            width=15,
            command=self.controller.stop_camera
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            top,
            text="Trigger Read [Space]",
            width=18,
            bg="#ffd966",
            command=self.controller.trigger_inspection
        ).pack(side=tk.LEFT, padx=20)

        self.status_var = tk.StringVar(value="IDLE")
        self.status_label = tk.Label(
            top,
            textvariable=self.status_var,
            width=16,
            font=("Segoe UI", 12, "bold"),
            relief="groove",
            bd=2,
            bg="lightgray"
        )
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.fps_var = tk.StringVar(value="FPS: 0.0")
        self.fps_label = tk.Label(
            top,
            textvariable=self.fps_var,
            width=12,
            font=("Segoe UI", 11, "bold"),
            relief="groove",
            bd=2,
            bg="white"
        )
        self.fps_label.pack(side=tk.LEFT, padx=5)

        main = tk.Frame(self.root)
        main.pack(expand=True, fill=tk.BOTH)

        left = tk.Frame(main)
        left.pack(side=tk.LEFT, padx=10, pady=10)

        self.canvas = tk.Canvas(
            left,
            width=SETTINGS.display.width,
            height=SETTINGS.display.height,
            bg="black"
        )
        self.canvas.pack()

        right = tk.Frame(main)
        right.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        tk.Label(
            right,
            text="Detected Waybill Number:",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w")
        self.waybill_list = tk.Listbox(right, height=5)
        self.waybill_list.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            right,
            text="Detected Barcode Data:",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w")
        self.barcode_list = tk.Listbox(right, height=5)
        self.barcode_list.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            right,
            text="OCR Raw Text:",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w")
        self.text_box = tk.Text(right, height=15)
        self.text_box.pack(fill=tk.BOTH, expand=True)

    def _bind_keys(self):
        self.root.bind("<space>", self._on_space_trigger)
        self.root.bind("<Escape>", self._on_escape)

    def _on_space_trigger(self, event=None):
        self.controller.trigger_inspection()

    def _on_escape(self, event=None):
        self.controller.stop_camera()

    def show_image(self, img_bgr):
        try:
            if img_bgr is None:
                return

            img_resized = cv2.resize(
                img_bgr,
                (SETTINGS.display.width, SETTINGS.display.height)
            )

            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            self.photo_ref = ImageTk.PhotoImage(pil_img)

            self.canvas.delete("all")
            self.canvas.create_image(
                SETTINGS.display.width // 2,
                SETTINGS.display.height // 2,
                image=self.photo_ref,
                anchor="center"
            )
        except Exception as e:
            print("[GUI ERROR] show_image:", e)

    def update_waybill_list(self, items):
        self.waybill_list.delete(0, tk.END)
        for item in items:
            self.waybill_list.insert(tk.END, item)

    def update_barcode_list(self, items):
        self.barcode_list.delete(0, tk.END)
        for item in items:
            self.barcode_list.insert(tk.END, item)

    def update_raw_texts(self, texts):
        self.text_box.delete("1.0", tk.END)
        for text in texts:
            self.text_box.insert(tk.END, text + "\n")

    def clear_results(self):
        self.update_waybill_list([])
        self.update_barcode_list([])
        self.update_raw_texts([])

    def update_status(self, text, color="lightgray"):
        self.status_var.set(text)
        self.status_label.config(bg=color)

    def update_fps(self, fps_value):
        self.fps_var.set(f"FPS: {fps_value:.1f}")

    def set_camera_options(self, camera_dict):
        self._camera_map = camera_dict or {}
        values = list(self._camera_map.keys())
        self.camera_dropdown["values"] = values

        if values:
            self.camera_dropdown.current(0)
            self.camera_var.set(values[0])
        else:
            self.camera_var.set("")

    def get_selected_camera_index(self):
        selected_name = self.camera_var.get()
        return self._camera_map.get(selected_name, 0)

    def _start_loop(self):
        self.controller.loop()
        self.root.after(30, self._start_loop)