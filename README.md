# Cricket Analytics Dashboard | Computer Vision Project

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

<img width="1536" height="1024" alt="images" src="https://github.com/user-attachments/assets/7a046a63-e135-4b2a-aefb-5b5bfd4a2f13" />




### 🧪 How It Works

-User selects or uploads a cricket video

-YOLOv8 detects players and ball

-Objects are projected to a tactical pitch

-Ball trajectory & heatmap are generated

-Outputs are visualized & made downloadable

### 📦 Outputs

🎞 Annotated video (*_annotated.mp4)

🗺 Tactical map video (*_tactical.mp4)

🔥 Ball heatmap (*_heatmap.png)

📄 Tracks CSV (*_tracks.csv)

### 🧠 Tech Stack

-Python

-YOLOv8 (Ultralytics)

-OpenCV

-Streamlit

-NumPy / Pandas

### 💡 Future Improvements

-ByteTrack integration (persistent IDs)

-Team classification using jersey color clustering

-Shot type classification

-Player speed & distance metrics

-Match report PDF export

-Cloud deployment (Streamlit Cloud / AWS)

### 👨‍💻 Author

Shivam Dubey
B.Tech CSE (2024)
AI / ML / Data Science Enthusiast



