# src/detector.py
import os
from typing import List, Dict, Any

from ultralytics import YOLO


class CricketDetector:
    """
    Wrapper around a YOLO model for cricket object detection.

    Usage:
        detector = CricketDetector("models/yolov8_cricket.pt")
        detections = detector.detect(frame, conf=0.4)
        # detections => list of dicts: {"bbox": (x1,y1,x2,y2), "label": "player", "confidence": 0.87}
    """

    def __init__(self, model_path: str):
        # Basic existence & sanity checks
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # if file is extremely small it's likely not a valid checkpoint (helps catch HTML/text files renamed .pt)
        try:
            size_bytes = os.path.getsize(model_path)
            if size_bytes < 1024:
                raise ValueError(
                    f"Model file '{model_path}' is suspiciously small ({size_bytes} bytes). "
                    "Please replace with a valid YOLO .pt checkpoint."
                )
        except OSError:
            # fallback (should rarely happen)
            pass

        # Attempt to load the model and give a clear error on failure
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO model from '{model_path}'. "
                "Make sure the file is a valid YOLOv8 PyTorch checkpoint (.pt) and not a text/HTML file. "
                f"Original error: {e}"
            ) from e

    def detect(self, frame, conf: float = 0.4) -> List[Dict[str, Any]]:
        """
        Run detection on a single BGR frame.

        Returns a list of detections:
            [{"bbox": (x1,y1,x2,y2), "label": str, "confidence": float}, ...]
        """
        results = self.model(frame, conf=conf, verbose=False)
        detections: List[Dict[str, Any]] = []

        for r in results:
            # r.boxes contains detected boxes for this frame
            for box in r.boxes:
                # xyxy may be a tensor-like object; convert to Python numbers
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                cls = int(box.cls[0])
                # labelling fallback if name missing
                label = self.model.names.get(cls, str(cls)) if hasattr(self.model, "names") else str(cls)
                # confidence (if available)
                confidence = float(box.conf[0]) if hasattr(box, "conf") else None

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "label": label,
                    "confidence": confidence
                })

        return detections
