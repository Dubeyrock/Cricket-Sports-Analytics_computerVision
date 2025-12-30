# quick_test.py
import cv2
import numpy as np
from pathlib import Path

def test_single_video(video_path):
    """Test processing on a single video"""
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return
    
    print(f"Testing with: {video_path.name}")
    
    # Just extract some frames and save them
    cap = cv2.VideoCapture(str(video_path))
    
    # Create output directory
    output_dir = Path('test_outputs')
    output_dir.mkdir(exist_ok=True)
    
    frame_count = 0
    save_interval = 30  # Save every 30th frame
    
    while cap.isOpened() and frame_count < 100:  # Just 100 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % save_interval == 0:
            # Save the frame
            output_path = output_dir / f"{video_path.stem}_frame_{frame_count:04d}.jpg"
            cv2.imwrite(str(output_path), frame)
            print(f"  Saved frame {frame_count} to {output_path}")
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames from {video_path.name}")

def main():
    # Find all videos
    video_dir = Path('data/raw_videos')
    if not video_dir.exists():
        print(f"Directory not found: {video_dir}")
        return
    
    video_files = list(video_dir.glob('*.mp4'))
    
    if not video_files:
        print("No videos found in data/raw_videos/")
        return
    
    print(f"Found {len(video_files)} video(s)")
    
    # Test with the first video
    if video_files:
        test_single_video(video_files[0])

if __name__ == "__main__":
    main()