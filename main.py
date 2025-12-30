# main.py
import cv2
import os
from src.detector import CricketDetector
from src.tactical_map import TacticalMap
from src.utils import draw_bbox

VIDEOS = [
    "data/raw_videos/Video1 - Copy.mp4",
    "data/raw_videos/Video2 - Copy.mp4",
    "data/raw_videos/Video3 - Copy.mp4",
    "data/raw_videos/Video4.mp4",
    "data/raw_videos/Video5.mp4",
    "data/raw_videos/video6.mp4",
    "data/raw_videos/Video7.mp4"
]

# Use a local model file or a vanilla yolov8n (will auto-download)
MODEL_PATH = "yolov8n.pt"

# Instantiate detector (detector will check model existence / validity)
detector = CricketDetector(MODEL_PATH)
tactical = TacticalMap()

os.makedirs("data/outputs/annotated", exist_ok=True)
os.makedirs("data/outputs/tactical_map", exist_ok=True)

for idx, video_path in enumerate(VIDEOS):
    print(f"[INFO] Starting: {video_path}")

    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}  —  skipping.")
        continue

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}  —  skipping.")
        cap.release()
        continue

    # Read the first frame to reliably obtain frame size (handles some codec quirks)
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"[ERROR] Cannot read first frame of {video_path}  —  skipping.")
        cap.release()
        continue

    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or fps != fps:  # check NaN
        fps = 25
        print(f"[WARN] Invalid FPS read — falling back to {fps} FPS")

    annotated_out_path = f"data/outputs/annotated/video{idx+1}_annotated.mp4"
    tactical_out_path = f"data/outputs/tactical_map/video{idx+1}_tactical.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(annotated_out_path, fourcc, fps, (w, h))
    tactical_writer = cv2.VideoWriter(tactical_out_path, fourcc, fps, (tactical.width, tactical.height))

    frame_idx = 0
    try:
        # Process the first frame and then continue with the rest
        while True:
            if frame is None:
                # read next frame
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[INFO] End of video or cannot read frame")
                    break

            frame_idx += 1

            # Run detection
            detections = detector.detect(frame, conf=0.35)
            print(f"[INFO] {os.path.basename(video_path)} - frame {frame_idx} - detections: {len(detections)}")

            # Tactical map for this frame
            pitch = tactical.draw_pitch()

            # Draw detections and project centers to tactical map
            for det in detections:
                bbox = det.get("bbox")
                label = det.get("label", "obj")
                conf = det.get("confidence")
                draw_bbox(frame, bbox, f"{label} {conf:.2f}" if conf is not None else label)

                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # simple linear projection to tactical map (placeholder)
                tx = int(cx * tactical.width / w)
                ty = int(cy * tactical.height / h)

                if label == "player":
                    tactical.draw_entity(pitch, tx, ty, (255, 255, 255))
                elif label == "ball":
                    tactical.draw_entity(pitch, tx, ty, (0, 0, 255))
                else:  # umpire / other
                    tactical.draw_entity(pitch, tx, ty, (255, 0, 0))

            # Write outputs
            out_video.write(frame)
            tactical_writer.write(pitch)

            # Display (if running on a machine with a display)
            cv2.imshow("Cricket Analytics - Annotated", frame)
            cv2.imshow("Cricket Analytics - Tactical Map", pitch)

            # read next frame for next iteration
            ret, frame = cap.read()
            if cv2.waitKey(30) & 0xFF == 27:  # ESC to break
                print("[INFO] Interrupted by user")
                break

    except Exception as e:
        print(f"[ERROR] Exception while processing {video_path}: {e}")

    finally:
        cap.release()
        out_video.release()
        tactical_writer.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Finished processing {video_path}")
        print(f"       -> Annotated saved to: {annotated_out_path}")
        print(f"       -> Tactical map saved to: {tactical_out_path}")

print("[INFO] All done.")
