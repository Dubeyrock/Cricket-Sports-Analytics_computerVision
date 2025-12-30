# Cricket Sports Analytics

A computer vision pipeline for tracking cricket entities and generating Hawk-Eye style tactical maps.

## Features

- **Multi-entity Detection**: Players, ball, umpire detection using YOLOv8
- **Multi-object Tracking**: Stable ID tracking with Kalman filter
- **Team Classification**: Automatic team identification based on jersey colors
- **2D Projection**: Top-down tactical map projection using homography
- **Visualization**: Annotated video + synchronized tactical map

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/cricket_sports_analytics.git
cd cricket_sports_analytics

# Install dependencies
pip install -r requirements.txt

# Download pre-trained model
wget -P models/ https://your-model-url/yolov8_cricket.pt