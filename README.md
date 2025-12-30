# Cricket Sports Analytics

🏏 End-to-end Cricket Analytics Dashboard using YOLOv8 & Streamlit. Detects players & ball, generates tactical pitch maps, ball heatmaps, match statistics, and downloadable annotated videos — built with production-ready computer vision pipelines.

## 🚀 Features

### 🎥 Video Analytics
- Player & ball detection using **YOLOv8**
- Annotated output video (bounding boxes + labels)
- Side-by-side **Original vs Annotated** preview

### 🗺 Tactical Map Intelligence
- Real-time projection of players & ball onto a **cricket pitch**
- Ball **trajectory visualization**
- Dedicated **tactical map video**

### 🔥 Heatmap & Shot Analysis
- Ball movement heatmap
- Ball touch count
- Visual understanding of shot distribution

### 📊 Match Statistics (Recruiter Favorite)
- Total frames processed
- Total detections
- Runtime & FPS
- Ball touches

### 🧠 Tracking & Data Export
- Object tracking (simple tracker, ByteTrack-ready)
- Tracks CSV export (frame-wise positions)
- Downloadable annotated videos, tactical videos & heatmaps

### 🎨 Professional UI (Streamlit)
- Sidebar-based controls
- Organized tabs:
  - Overview
  - Preview
  - Tactical Map
  - Heatmap & Trajectory
  - Tracks & CSV
  - Downloads
- Session-based caching (no reprocessing)

---

## 🖥 Demo



## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/cricket_sports_analytics.git
cd cricket_sports_analytics

# Install dependencies
pip install -r requirements.txt

# Download pre-trained model

wget -P models/ https://your-model-url/yolov8_cricket.pt
