# create_test_video.py
import cv2
import numpy as np
import os
from pathlib import Path

# Create directories
Path("data/raw_videos").mkdir(parents=True, exist_ok=True)
Path("data/videos").mkdir(parents=True, exist_ok=True)

# Create a sample cricket video
video_path = "data/raw_videos/cricket_sample.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, 30.0, (1280, 720))

for i in range(300):  # 10 seconds at 30 fps
    # Create green field background
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (0, 100, 0)  # Green
    
    # Draw pitch
    cv2.rectangle(frame, (540, 160), (740, 560), (139, 69, 19), -1)
    
    # Draw moving players (team A - red)
    cv2.rectangle(frame, 
                  (200 + i, 300), 
                  (240 + i, 400), 
                  (0, 0, 255), -1)  # Moving player
    
    cv2.rectangle(frame, 
                  (400, 200 + int(50 * np.sin(i * 0.1))), 
                  (440, 300 + int(50 * np.sin(i * 0.1))), 
                  (0, 0, 255), -1)  # Bouncing player
    
    # Draw moving players (team B - blue)
    cv2.rectangle(frame, 
                  (1000 - i, 300), 
                  (1040 - i, 400), 
                  (255, 0, 0), -1)  # Moving player
    
    cv2.rectangle(frame, 
                  (800, 200 + int(50 * np.cos(i * 0.1))), 
                  (840, 300 + int(50 * np.cos(i * 0.1))), 
                  (255, 0, 0), -1)  # Bouncing player
    
    # Draw ball (green circle)
    ball_x = 640 + int(200 * np.sin(i * 0.2))
    ball_y = 360 + int(100 * np.cos(i * 0.2))
    cv2.circle(frame, (ball_x, ball_y), 15, (0, 255, 0), -1)
    
    # Draw umpire (yellow)
    cv2.rectangle(frame, 
                  (640, 100), 
                  (680, 200), 
                  (0, 255, 255), -1)
    
    out.write(frame)

out.release()
print(f"Created sample video: {video_path}")

# Create symbolic link
import shutil
shutil.copy2(video_path, "data/videos/cricket_sample.mp4")
print("Copied to data/videos/")

print("\nNow run: python main.py --video_path data/videos/cricket_sample.mp4")