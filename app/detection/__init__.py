class AppController:
    def __init__(self, gui):
        self.gui = gui
        self.cap = None
        self.current_frame = None

        self.waybill_reader = WaybillReader(
            waybill_model_path=r"C:/Users/lenovo/Documents/GitHub/DHL_DEMO/models/detection/waybill_good.pt",
            read_model_path=r"C:/Users/lenovo/Documents/GitHub/DHL_DEMO/models/recognize/read100.pt",
            waybill_conf=0.45,
            digit_conf=0.25,
            roi_pad=0,
            expected_length=10,
        )

        self.last_waybill = None