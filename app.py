# app.py — Feature-complete: heatmap, stats panel, tabs, team classification
import streamlit as st
from pathlib import Path
import tempfile
import time
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from src.detector import CricketDetector
from src.tactical_map import TacticalMap
from src.utils import draw_bbox

# ---------------- Streamlit config ----------------
st.set_page_config(page_title="🏏 Cricket Analytics", layout="wide", initial_sidebar_state="expanded")

# ---------------- Paths ----------------
ROOT = Path(".")
DATA_DIR = ROOT / "data" / "raw_videos"
OUT_ANN = ROOT / "data" / "outputs" / "annotated"
OUT_TAC = ROOT / "data" / "outputs" / "tactical_map"
HEAT_DIR = ROOT / "data" / "outputs" / "heatmaps"
TRACKS_DIR = OUT_ANN

OUT_ANN.mkdir(parents=True, exist_ok=True)
OUT_TAC.mkdir(parents=True, exist_ok=True)
HEAT_DIR.mkdir(parents=True, exist_ok=True)
TRACKS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Sidebar ----------------
st.sidebar.title("⚙ Run Settings")
conf_thresh = st.sidebar.slider("Confidence", 0.1, 0.9, 0.35, 0.05)
frame_skip = st.sidebar.number_input("Process every Nth frame", 1, 30, 1)
resize_short = st.sidebar.number_input("Resize short side (px)", 0, 1280, 720)
force_reprocess = st.sidebar.checkbox("Force reprocess (overwrite outputs)", value=False)

st.sidebar.markdown("---")
st.sidebar.info("For CPU demos: increase frame-skip and lower resize_short for speed.")

# ---------------- Video input ----------------
st.sidebar.header("🎥 Videos")
available_videos = sorted([str(p) for p in DATA_DIR.glob("*.mp4")])
selected_videos = st.sidebar.multiselect("Select videos to process", available_videos)

uploaded = st.sidebar.file_uploader("Or upload a video (single)", type=["mp4", "mov", "avi"])
if uploaded:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded.read()); tmp.flush()
    selected_videos.append(tmp.name)
    st.sidebar.success(f"Uploaded: {Path(tmp.name).name}")

# ---------------- Detector loader ----------------
@st.cache_resource
def load_detector():
    # You can change to your custom model path if available
    return CricketDetector("yolov8n.pt")

# ---------------- Team classifier (simple color-based) ----------------
def classify_team(frame: np.ndarray, bbox: Tuple[int,int,int,int]) -> str:
    x1,y1,x2,y2 = bbox
    # Clip
    x1 = max(0, min(frame.shape[1]-1, x1))
    x2 = max(0, min(frame.shape[1]-1, x2))
    y1 = max(0, min(frame.shape[0]-1, y1))
    y2 = max(0, min(frame.shape[0]-1, y2))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "Unknown"
    mean = crop.reshape(-1,3).mean(axis=0)  # BGR
    # heuristic: if red channel > blue channel -> Team A else Team B
    return "Team A" if mean[2] > mean[0] else "Team B"

# ---------------- Heatmap util ----------------
def make_heatmap(points: List[Tuple[int,int]], out_path: Path, shape: Tuple[int,int]) -> str:
    # shape: (width, height)
    width, height = shape
    canvas = np.zeros((height, width), dtype=np.float32)
    for x,y in points:
        if 0 <= x < width and 0 <= y < height:
            canvas[y, x] += 1.0
    if canvas.max() <= 0:
        # save blank heatmap
        empty = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.imwrite(str(out_path), empty)
        return str(out_path)
    canvas = cv2.GaussianBlur(canvas, (0,0), sigmaX=6, sigmaY=6)
    norm = np.uint8(255 * (canvas / canvas.max()))
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_path), color)
    return str(out_path)

# ---------------- Simple tracker (demo) ----------------
class SimpleTracker:
    def __init__(self):
        self.next_id = 1
    def update(self, detections: List[Dict], frame_idx: int):
        # naive: assign incremental ids
        out = []
        for d in detections:
            d2 = d.copy()
            d2["id"] = self.next_id
            self.next_id += 1
            out.append(d2)
        return out

# ---------------- Core processing function ----------------
def process_single_video(video_path: str, detector: CricketDetector, tactical: TacticalMap) -> Dict:
    """
    Runs detection+simple tracking and writes annotated + tactical videos,
    returns metadata including heatmap path, tracks csv and stats.
    """
    base = Path(video_path).stem
    ann_out = OUT_ANN / f"{base}_annotated.mp4"
    tac_out = OUT_TAC / f"{base}_tactical.mp4"
    heat_out = HEAT_DIR / f"{base}_heatmap.png"
    tracks_out = TRACKS_DIR / f"{base}_tracks.csv"

    # force overwrite if requested
    if force_reprocess:
        for p in [ann_out, tac_out, heat_out, tracks_out]:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = 25.0

    # H264 avc1 for browser compatibility
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out_ann = cv2.VideoWriter(str(ann_out), fourcc, fps, (w, h))
    out_tac = cv2.VideoWriter(str(tac_out), fourcc, fps, (tactical.width, tactical.height))

    if not out_ann.isOpened() or not out_tac.isOpened():
        raise RuntimeError("VideoWriter failed to open. Check codec/FPS.")

    tracker = SimpleTracker()
    frames = 0
    total_dets = 0
    ball_points = []
    tracks_rows = []

    start = time.time()
    # main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        orig = frame.copy()

        # frame skip to speed up
        if frames % frame_skip != 0:
            out_ann.write(orig.astype("uint8"))
            out_tac.write(tactical.draw_pitch().astype("uint8"))
            continue

        # optional resize for detection speed (we detect on original here for simplicity)
        det_frame = frame
        # run detection
        dets = detector.detect(det_frame, conf=conf_thresh)  # returns list of {"bbox":(), "label": str, "confidence":float}

        total_dets += len(dets)

        # convert dets format for tracker: ensure bbox ints
        dets_int = []
        for d in dets:
            x1,y1,x2,y2 = map(int, d["bbox"])
            dets_int.append({"bbox": (x1,y1,x2,y2), "label": d.get("label","obj"), "confidence": d.get("confidence")})

        tracked = tracker.update(dets_int, frames)
        pitch = tactical.draw_pitch()

        for t in tracked:
            x1,y1,x2,y2 = t["bbox"]
            label = t.get("label","obj")
            obj_id = t.get("id", None)

            # Team classification for players
            label_text = label
            if label == "player":
                team = classify_team(frame, (x1,y1,x2,y2))
                label_text = f"{team}_{obj_id}"
            elif obj_id is not None:
                label_text = f"{label}_{obj_id}"

            draw_bbox(orig, (x1,y1,x2,y2), label_text)

            cx = (x1 + x2)//2
            cy = (y1 + y2)//2
            tx = int(cx * tactical.width / w)
            ty = int(cy * tactical.height / h)

            color = (255,255,255) if label == "player" else (0,0,255)
            tactical.draw_entity(pitch, tx, ty, color)

            if label == "ball":
                ball_points.append((tx, ty))

            tracks_rows.append({"frame": frames, "id": obj_id, "label": label, "cx": cx, "cy": cy})

        # draw ball trajectory if tactical supports (or draw directly)
        if hasattr(tactical, "draw_trajectory"):
            try:
                tactical.draw_trajectory(pitch, ball_points[-50:], max_len=50)
            except TypeError:
                # fallback signature compatibility
                tactical.draw_trajectory(pitch, ball_points[-50:])

        else:
            # fallback: draw small polyline
            if len(ball_points) >= 2:
                pts = np.array(ball_points[-50:], dtype=np.int32)
                cv2.polylines(pitch, [pts], False, (0,0,255), 2, lineType=cv2.LINE_AA)

        out_ann.write(orig.astype("uint8"))
        out_tac.write(pitch.astype("uint8"))

    cap.release()
    out_ann.release()
    out_tac.release()

    runtime = round(time.time() - start, 2)

    # save tracks CSV
    if tracks_rows:
        df_tracks = pd.DataFrame(tracks_rows)
        df_tracks.to_csv(tracks_out, index=False)
    else:
        df_tracks = pd.DataFrame([])

    # heatmap (tactical coords width,height)
    heatmap_path = None
    if ball_points:
        heatmap_path = make_heatmap(ball_points, Path(heat_out), (tactical.width, tactical.height))
    # pack results
    result = {
        "video": str(video_path),
        "annotated": str(ann_out),
        "tactical": str(tac_out),
        "heatmap": heatmap_path,
        "tracks_csv": str(tracks_out) if not df_tracks.empty else None,
        "frames": frames,
        "detections": total_dets,
        "runtime": runtime,
        "ball_touches": len(ball_points),
        "tracks_df": df_tracks
    }
    return result

# ---------------- UI ----------------
st.title("🏏 Cricket Analytics Dashboard")
st.markdown("""
**About / Overview**  
This demo runs YOLO-based detection on cricket videos, projects detections onto a tactical pitch, tracks objects (simple demo tracker), draws ball trajectories and heatmaps, and exports tracks CSV for analysis. Use the sidebar to pick videos and tune parameters.
""")

if st.button("▶ Run Analysis"):
    if not selected_videos:
        st.warning("No videos selected — pick one in the sidebar")
    else:
        detector = load_detector()
        tactical = TacticalMap()
        all_results = []

        for vid in selected_videos:
            st.subheader(Path(vid).name)
            with st.spinner(f"Processing {Path(vid).name} — this can take a while"):
                res = process_single_video(vid, detector, tactical)
            st.success(f"Finished: {Path(vid).name} (frames={res['frames']}, dets={res['detections']})")
            all_results.append(res)

            # ---------------- Tabs (ordered) ----------------
            tabs = st.tabs(["Overview", "Preview", "Tactical Map", "Heatmap & Trajectory", "Tracks & CSV", "Downloads"])

            # Overview / Stats
            with tabs[0]:
                st.subheader("📊 Match Statistics")
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Total Frames", res["frames"])
                c2.metric("Total Detections", res["detections"])
                c3.metric("Ball Touches", res["ball_touches"])
                fps_val = round(res["frames"]/res["runtime"], 2) if res["runtime"]>0 else 0.0
                c4.metric("FPS (approx)", fps_val)
                st.write(f"Processing time: {res['runtime']} s")

            # Preview (Annotated vs Original)
            with tabs[1]:
                st.subheader("Preview — Annotated vs Original")
                left, right = st.columns(2)
                with left:
                    st.caption("Annotated")
                    if Path(res["annotated"]).exists():
                        st.video(res["annotated"])
                    else:
                        st.text("Annotated video not found")
                with right:
                    st.caption("Original")
                    st.video(res["video"])

            # Tactical Map video
            with tabs[2]:
                st.subheader("Tactical Map (video)")
                if Path(res["tactical"]).exists():
                    st.video(res["tactical"])
                else:
                    st.text("Tactical video not found")

            # Heatmap & Trajectory
            with tabs[3]:
                st.subheader("Heatmap & Ball Trajectory")
                st.metric("Ball touches", res["ball_touches"])
                if res.get("heatmap") and Path(res["heatmap"]).exists():
                    st.image(res["heatmap"], caption="Ball movement heatmap", width=700)
                else:
                    st.text("No heatmap generated (no ball detections)")

            # Tracks & CSV
            with tabs[4]:
                st.subheader("Tracks & CSV")
                if res.get("tracks_csv") and Path(res["tracks_csv"]).exists():
                    df = res["tracks_df"]
                    st.dataframe(df.head(200))
                    with open(res["tracks_csv"], "rb") as f:
                        st.download_button("Download tracks CSV", f.read(), file_name=Path(res["tracks_csv"]).name, mime="text/csv")
                else:
                    st.text("No tracks CSV available (no detections)")

            # Downloads
            with tabs[5]:
                st.subheader("Downloads")
                if Path(res["annotated"]).exists():
                    with open(res["annotated"], "rb") as f:
                        st.download_button("Download Annotated Video", f.read(), file_name=Path(res["annotated"]).name, mime="video/mp4")
                if Path(res["tactical"]).exists():
                    with open(res["tactical"], "rb") as f:
                        st.download_button("Download Tactical Video", f.read(), file_name=Path(res["tactical"]).name, mime="video/mp4")
                if res.get("heatmap") and Path(res["heatmap"]).exists():
                    with open(res["heatmap"], "rb") as f:
                        st.download_button("Download Heatmap", f.read(), file_name=Path(res["heatmap"]).name, mime="image/png")

        # Keep results in session if you want further UI interactions
        st.session_state["last_results"] = all_results

# end of file
