# setup.py
import os
import sys
from pathlib import Path
import subprocess
import zipfile
import requests

def create_directories():
    """Create necessary directories"""
    directories = [
        'config',
        'data/videos',
        'data/raw_videos',  # For original videos
        'outputs',
        'assets',
        'src'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_config_file():
    """Create configuration file"""
    config_content = """# Model parameters
model_path: yolov8x.pt
conf_threshold: 0.5
iou_threshold: 0.3

# Tracking parameters
max_age: 30
min_hits: 3
max_cosine_distance: 0.2

# Field parameters
field_template: assets/cricket_field_template.jpg
raw_videos_dir: data/raw_videos
field_length: 20.12
field_width: 3.05

# Video processing
output_fps: 30
output_resolution: [1280, 720]
tactical_map_size: [800, 600]

# Colors for visualization
team_colors:
  team_a: [255, 0, 0]
  team_b: [0, 0, 255]
  ball: [0, 255, 0]
  umpire: [255, 255, 0]
"""
    
    with open('config/params.yaml', 'w') as f:
        f.write(config_content)
    print("✓ Created config/params.yaml")

def create_field_template():
    """Create a cricket field template"""
    import cv2
    import numpy as np
    
    # Create a green cricket field
    field = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Fill with green color
    field[:] = (34, 139, 34)  # Green
    
    # Draw pitch (brown rectangle in center)
    pitch_width = 80
    pitch_height = 400
    pitch_x = (800 - pitch_width) // 2
    pitch_y = (600 - pitch_height) // 2
    
    # Draw pitch
    cv2.rectangle(field, (pitch_x, pitch_y), (pitch_x + pitch_width, pitch_y + pitch_height), (101, 67, 33), -1)
    
    # Draw crease lines
    # Bowling crease
    cv2.rectangle(field, (pitch_x - 20, pitch_y), (pitch_x + pitch_width + 20, pitch_y + 10), (255, 255, 255), 2)
    cv2.rectangle(field, (pitch_x - 20, pitch_y + pitch_height - 10), 
                  (pitch_x + pitch_width + 20, pitch_y + pitch_height), (255, 255, 255), 2)
    
    # Popping crease (closer to stumps)
    cv2.rectangle(field, (pitch_x - 40, pitch_y + 50), (pitch_x + pitch_width + 40, pitch_y + 60), (255, 255, 255), 2)
    cv2.rectangle(field, (pitch_x - 40, pitch_y + pitch_height - 60), 
                  (pitch_x + pitch_width + 40, pitch_y + pitch_height - 50), (255, 255, 255), 2)
    
    # Draw stumps
    stump_width = 5
    stump_height = 30
    stump_x = pitch_x + pitch_width // 2 - stump_width // 2
    
    # Stumps at both ends
    cv2.rectangle(field, (stump_x, pitch_y - stump_height), (stump_x + stump_width, pitch_y), (139, 69, 19), -1)
    cv2.rectangle(field, (stump_x, pitch_y + pitch_height), (stump_x + stump_width, pitch_y + pitch_height + stump_height), (139, 69, 19), -1)
    
    # Draw boundary (circle)
    cv2.circle(field, (400, 300), 250, (255, 255, 255), 2)
    
    # Save the template
    cv2.imwrite('assets/cricket_field_template.jpg', field)
    print("✓ Created cricket field template at assets/cricket_field_template.jpg")
    
    return field

def download_sample_videos():
    """Download sample cricket videos if no videos exist"""
    video_dir = Path('data/raw_videos')
    
    # Check if we already have videos
    video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi')) + list(video_dir.glob('*.mov'))
    
    if len(video_files) > 0:
        print(f"✓ Found {len(video_files)} video(s) in data/raw_videos/")
        
        # Create symbolic links or copy to videos directory
        target_dir = Path('data/videos')
        target_dir.mkdir(exist_ok=True)
        
        for video in video_files:
            target_path = target_dir / video.name
            if not target_path.exists():
                try:
                    # Try to create symlink (works on Unix), fallback to copy
                    import platform
                    if platform.system() != 'Windows':
                        os.symlink(video.absolute(), target_path)
                    else:
                        # Windows - copy the file
                        import shutil
                        shutil.copy2(video, target_path)
                    print(f"  Linked/Copied: {video.name}")
                except:
                    # Just use the raw_videos directory directly
                    pass
        return
    
    print("\n⚠  No cricket videos found in data/raw_videos/")
    print("Please download cricket videos from:")
    print("Google Drive: https://drive.google.com/drive/folders/1kFXWTAGk_rM_z_gFEI34yF4GsUqyFQYx")
    print("\nAfter downloading, place the videos in: data/raw_videos/")
    print("Then run this setup again.")
    
    # Create a sample video for testing (optional)
    create_test_video()

def create_test_video():
    """Create a simple test video if no videos are available"""
    print("\nCreating a test video for debugging...")
    
    import cv2
    import numpy as np
    
    # Create a simple test video with moving rectangles
    video_path = 'data/videos/test_cricket.mp4'
    Path('data/videos').mkdir(exist_ok=True, parents=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
    
    for i in range(100):  # 100 frames
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (34, 139, 34)  # Green background
        
        # Draw a moving "player"
        player_x = 300 + int(100 * np.sin(i * 0.1))
        player_y = 200 + int(50 * np.cos(i * 0.1))
        cv2.rectangle(frame, (player_x, player_y), (player_x + 40, player_y + 80), (255, 0, 0), -1)
        
        # Draw a moving "ball"
        ball_x = 200 + i * 3
        ball_y = 250 + int(30 * np.sin(i * 0.2))
        cv2.circle(frame, (ball_x % 600, ball_y), 10, (0, 255, 0), -1)
        
        out.write(frame)
    
    out.release()
    print(f"✓ Created test video: {video_path}")

def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    
    dependencies = [
        'torch',
        'torchvision',
        'ultralytics',
        'opencv-python',
        'deep-sort-realtime',
        'numpy',
        'scikit-learn',
        'scipy',
        'pandas',
        'matplotlib',
        'seaborn',
        'moviepy',
        'pyyaml',
        'tqdm'
    ]
    
    print("Run: pip install " + " ".join(dependencies))
    print("\nOr create a requirements.txt file and run: pip install -r requirements.txt")

def main():
    print("=" * 60)
    print("CRICKET SPORTS ANALYTICS - SETUP")
    print("=" * 60)
    
    # Step 1: Create directories
    print("\n1. Creating directory structure...")
    create_directories()
    
    # Step 2: Create config file
    print("\n2. Creating configuration file...")
    create_config_file()
    
    # Step 3: Create field template
    print("\n3. Creating cricket field template...")
    create_field_template()
    
    # Step 4: Setup videos
    print("\n4. Setting up video files...")
    download_sample_videos()
    
    # Step 5: Install dependencies
    print("\n5. Dependency installation...")
    install_dependencies()
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Download cricket videos from:")
    print("   https://drive.google.com/drive/folders/1kFXWTAGk_rM_z_gFEI34yF4GsUqyFQYx")
    print("2. Place videos in: data/raw_videos/")
    print("3. Run: python main.py")
    print("\nFor testing, a sample video has been created at: data/videos/test_cricket.mp4")
    print("You can test with: python main.py --video_path data/videos/test_cricket.mp4")

if __name__ == "__main__":
    main()