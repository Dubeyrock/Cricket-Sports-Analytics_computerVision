from ultralytics import YOLO

class CricketTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def track(self, video_path, save_path):
        self.model.track(
            source=video_path,
            tracker="bytetrack.yaml",
            persist=True,
            save=True,
            project=save_path,
            name="tracking"
        )
