# simple_tracker.py
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch

class SimpleCricketTracker:
    def __init__(self):
        # Use YOLOv8 for detection
        self.model = YOLO('yolov8s.pt')
        print("Model loaded successfully")
        
        # Create output directory
        Path('simple_outputs').mkdir(exist_ok=True)
    
    def process_video(self, video_path):
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            return
        
        print(f"Processing: {video_path.name}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Reduce resolution for faster processing
        output_width = 1280
        output_height = 720
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = f'simple_outputs/annotated_{video_path.stem}.mp4'
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        
        frame_count = 0
        max_frames = 100  # Process only 100 frames for testing
        
        print(f"Video: {width}x{height}, {fps} FPS")
        print(f"Processing {max_frames} frames...")
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for processing
            frame_resized = cv2.resize(frame, (output_width, output_height))
            
            # Run YOLO detection
            results = self.model(frame_resized, conf=0.3, classes=[0])  # Only detect persons
            
            # Draw detections
            annotated_frame = frame_resized.copy()
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"Person: {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Write frame to output
            out.write(annotated_frame)
            
            frame_count += 1
            
            if frame_count % 10 == 0:
                print(f"  Processed frame {frame_count}")
        
        cap.release()
        out.release()
        
        print(f"✓ Completed: {video_path.name}")
        print(f"✓ Output saved to: {output_path}")
        print(f"✓ Total frames processed: {frame_count}")
        
        return output_path

def main():
    tracker = SimpleCricketTracker()
    
    # Find videos
    video_dir = Path('data/raw_videos')
    if not video_dir.exists():
        print("Directory not found: data/raw_videos")
        return
    
    video_files = list(video_dir.glob('*.mp4'))
    
    if not video_files:
        print("No videos found. Please place videos in data/raw_videos/")
        return
    
    print(f"Found {len(video_files)} video(s)")
    
    # Process the first video
    if video_files:
        tracker.process_video(video_files[0])

if __name__ == "__main__":
    main()