# test_simple.py
import cv2
import numpy as np
from pathlib import Path

print("Simple video processing test...")

# Check for videos
video_dir = Path('data/raw_videos')
if video_dir.exists():
    videos = list(video_dir.glob('*.mp4'))
    if videos:
        print(f"Found {len(videos)} video(s)")
        
        # Test with first video
        video_path = videos[0]
        print(f"\nTesting with: {video_path.name}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if cap.isOpened():
            # Read first frame
            ret, frame = cap.read()
            
            if ret:
                print(f"Frame shape: {frame.shape}")
                print(f"Frame type: {frame.dtype}")
                
                # Create output directory
                Path('test_output').mkdir(exist_ok=True)
                
                # Save first frame
                cv2.imwrite('test_output/first_frame.jpg', frame)
                print("Saved first frame to test_output/first_frame.jpg")
                
                # Process a few more frames
                frames_to_process = 10
                print(f"\nProcessing {frames_to_process} frames...")
                
                for i in range(frames_to_process):
                    ret, frame = cap.read()
                    if ret:
                        # Simple detection: find faces/people (just for testing)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                        # Use Haar cascade for face detection (simplest option)
                        face_cascade = cv2.CascadeClassifier(
                            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                        )
                        
                        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                        
                        # Draw rectangles around faces
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        cv2.imwrite(f'test_output/frame_{i:03d}.jpg', frame)
                        print(f"  Frame {i}: Found {len(faces)} faces")
                    else:
                        break
                
                print("\n✅ Test completed successfully!")
                print("Check 'test_output' folder for results")
            else:
                print("❌ Could not read frame from video")
        else:
            print("❌ Could not open video")
        
        cap.release()
    else:
        print("❌ No MP4 videos found in data/raw_videos/")
else:
    print("❌ Directory data/raw_videos/ not found")