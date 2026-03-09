from ultralytics import YOLO


class WaybillDetector:
    def __init__(self, model_path="models/detection/waybill_good.pt", conf=0.4):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                detections.append({
                    "bbox": xyxy.tolist(),
                    "class_id": cls_id,
                    "confidence": conf
                })

        return detections