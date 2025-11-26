# streamlit_app.py
import streamlit as st
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import time
import torch
import tempfile
import os

st.set_page_config(page_title="Detective-AI - Weapon Detector", layout="wide")

# ---------------- Settings (fixed recommended) ----------------
MODEL_PATHS = [
    Path("models/weapon_best.pt"),
    Path("runs/detect/weapon_s_model/weights/best.pt"),
    Path("runs/detect/weapon_s_model2/weights/best.pt"),
]
IMG_SIZE = 416           # recommended for 6GB VRAM
CONF_THR = 0.25
IOU_THR = 0.45
WEAPON_CLASS_INDEX = 1   # according to your data.yaml: 0=person, 1=weapon
MAX_FPS = 12

# ---------------- Model loader ----------------
@st.cache_resource(show_spinner=False)
def load_model():
    model_file = None
    for p in MODEL_PATHS:
        if p.exists():
            model_file = str(p)
            break
    if model_file is None:
        st.warning(
            "No trained model file found in expected paths. Using 'yolov8n.pt' as fallback (not trained for weapons). "
            "Place your trained 'best.pt' in models/weapon_best.pt or runs/detect/.../weights/"
        )
        model = YOLO("yolov8n.pt")
        return model, None
    
    model = YOLO(model_file)
    try:
        if torch.cuda.is_available():
            model.to(0)
    except Exception as e:
        st.warning(f"Could not move model to GPU: {e}")
    return model, model_file

model, model_path = load_model()

# ---------------- session state initialization ----------------
if "running" not in st.session_state: 
    st.session_state.running = False
if "events_log" not in st.session_state: 
    st.session_state.events_log = ""
if "video_temp_path" not in st.session_state: 
    st.session_state.video_temp_path = None
if "uploaded_name" not in st.session_state: 
    st.session_state.uploaded_name = None
if "frame_idx" not in st.session_state: 
    st.session_state.frame_idx = 0
if "last_frame_time" not in st.session_state: 
    st.session_state.last_frame_time = 0.0
if "current_frame" not in st.session_state:
    st.session_state.current_frame = None
if "last_alert" not in st.session_state:
    st.session_state.last_alert = "ℹ️ Click 'Start Scan' to begin weapon detection"

# ---------------- UI layout ----------------
st.title("🔴 Detective-AI — Live Weapon Detector")
col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("### Alerts & Controls")
    st.caption("Model: " + (os.path.basename(model_path) if model_path else "fallback yolov8n"))
    start_button = st.button("Start Scan", type="primary", use_container_width=True)
    stop_button = st.button("Stop Scan", type="secondary", use_container_width=True)
    
    # Clear logs button
    if st.button("Clear Event Log", use_container_width=True):
        st.session_state.events_log = ""
        st.session_state.last_alert = "ℹ️ Event log cleared"
        st.rerun()
    
    st.write("---")
    st.markdown("**Input source**")
    source_type = st.selectbox("Source:", ["Webcam (live)", "Upload video"])
    uploaded_file = None
    if source_type == "Upload video":
        uploaded_file = st.file_uploader("Upload video file (MP4, AVI, MOV, MKV)", type=["mp4", "avi", "mov", "mkv"])
    st.write("---")
    st.markdown("**Fixed settings (recommended)**")
    st.write(f"- Image size: {IMG_SIZE}")
    st.write(f"- Confidence threshold: {CONF_THR}")
    st.write(f"- NMS IoU threshold: {IOU_THR}")
    st.write(f"- Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    st.write("---")
    st.markdown("**Detected events**")
    events_out = st.empty()

with col1:
    frame_placeholder = st.empty()
    info = st.empty()
    alert_box = st.empty()

# Save uploaded file to temp and set session_state.video_temp_path
if uploaded_file is not None:
    # write to a temporary file only once
    if st.session_state.get("uploaded_name") != uploaded_file.name:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        tfile.write(uploaded_file.read())
        tfile.flush()
        tfile.close()
        st.session_state.video_temp_path = tfile.name
        st.session_state.uploaded_name = uploaded_file.name
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        st.session_state.frame_idx = 0

# ---------------- helper: process a single frame ----------------
def process_frame_and_display(frame_bgr):
    """Run detection on frame_bgr (BGR), draw boxes on BGR, display in Streamlit."""
    if frame_bgr is None or frame_bgr.size == 0:
        return False
    
    # Store current frame for display when stopped
    st.session_state.current_frame = frame_bgr.copy()
    
    # run detection on RGB image (ultralytics accepts numpy RGB)
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    try:
        results = model.predict(
            source=img_rgb,
            imgsz=IMG_SIZE,
            conf=CONF_THR,
            iou=IOU_THR,
            device=0 if torch.cuda.is_available() else "cpu",
            verbose=False
        )
    except Exception as e:
        st.error(f"Detection error: {e}")
        return False

    display_bgr = frame_bgr.copy()
    alert_msgs = []

    if results and len(results):
        r = results[0]
        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes):
            # convert tensors to numpy safely
            try:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clses = r.boxes.cls.cpu().numpy()
            except Exception:
                xyxy = np.array(r.boxes.xyxy)
                confs = np.array(r.boxes.conf)
                clses = np.array(r.boxes.cls)

            for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clses):
                cls = int(cls)
                if cls == WEAPON_CLASS_INDEX:
                    # BGR red
                    color = (0, 0, 255)
                    cv2.rectangle(display_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    cv2.putText(display_bgr, f"WEAPON {conf:.2f}", (int(x1), max(20, int(y1)-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    alert_msgs.append(f"Weapon detected (conf {conf:.2f})")
                else:
                    color = (0, 255, 0)
                    cv2.rectangle(display_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = model.names.get(cls, f"cls{cls}") if hasattr(model, 'names') else f"cls{cls}"
                    cv2.putText(display_bgr, f"{label} {conf:.2f}", (int(x1), max(20, int(y1)-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # convert to RGB for PIL/Streamlit
    display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(display_rgb)
    frame_placeholder.image(img_pil, use_container_width=True)

    # Alerts / Event log
    if alert_msgs:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"{timestamp} - {alert_msgs[0]}\n"
        st.session_state.events_log += log_entry
        st.session_state.last_alert = f"🔴 ALERT: {alert_msgs[0]}"
        alert_box.markdown(f"### 🔴 ALERT: {alert_msgs[0]}")
    else:
        st.session_state.last_alert = "✅ No weapon detected"
        alert_box.markdown("### ✅ No weapon detected")
    
    return True

# ---------------- Display persistent event log ----------------
def display_event_log():
    """Display the event log in the sidebar - persists even when stopped"""
    if st.session_state.events_log:
        events_out.text_area(
            "Event Log", 
            st.session_state.events_log, 
            height=300, 
            disabled=True,
            key=f"event_log_{len(st.session_state.events_log)}"  # Unique key to force update
        )
    else:
        events_out.text_area(
            "Event Log", 
            "No events yet. Start scanning to detect weapons.", 
            height=300, 
            disabled=True,
            key="event_log_empty"
        )

# Display event log (always visible)
display_event_log()

# ---------------- Main Start/Stop logic ----------------
if start_button:
    if st.session_state.running:
        info.info("⚠️ Already running")
    else:
        if source_type == "Upload video" and st.session_state.video_temp_path is None:
            info.error("❌ Please upload a video file first")
        else:
            st.session_state.running = True
            st.session_state.frame_idx = 0
            st.session_state.last_frame_time = 0.0
            info.success("✅ Scan started")
            time.sleep(0.3)
            st.rerun()

if stop_button:
    if st.session_state.running:
        st.session_state.running = False
        info.info("⏸️ Scanner stopped")
        # Keep the last frame and alert displayed
        alert_box.markdown(f"### {st.session_state.last_alert}")
        # Don't clear the frame - it stays visible
    else:
        info.info("ℹ️ Scanner already stopped")

# ---------------- Processing loop ----------------
if st.session_state.running:
    if source_type == "Webcam (live)":
        # Webcam mode
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            info.error("❌ Unable to open webcam. Please check if it's connected and not in use by another application.")
            st.session_state.running = False
        else:
            # Set webcam properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                info.warning("⚠️ Failed to grab frame from webcam. Retrying...")
                time.sleep(0.1)
            else:
                # Throttle by MAX_FPS
                now = time.time()
                elapsed = now - st.session_state.last_frame_time
                if elapsed < 1.0 / MAX_FPS:
                    time.sleep(1.0 / MAX_FPS - elapsed)
                
                st.session_state.last_frame_time = time.time()
                process_frame_and_display(frame)
    
    else:
        # Video file mode
        if st.session_state.video_temp_path is None:
            info.warning("⚠️ Please upload a video file first.")
            st.session_state.running = False
        else:
            video_path = st.session_state.video_temp_path
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                info.error(f"❌ Unable to open video file: {os.path.basename(video_path)}")
                st.session_state.running = False
            else:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Display progress
                progress_pct = (st.session_state.frame_idx / total_frames * 100) if total_frames > 0 else 0
                info.info(f"📹 Processing frame {st.session_state.frame_idx + 1}/{total_frames} ({progress_pct:.1f}%) | FPS: {fps:.1f}")
                
                # Seek and read frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_idx)
                ret, frame = cap.read()
                cap.release()
                
                if not ret or frame is None:
                    info.success(f"✅ Video processing complete! Processed {st.session_state.frame_idx} frames.")
                    st.session_state.running = False
                    # Keep the last frame displayed
                    alert_box.markdown(f"### {st.session_state.last_alert}")
                else:
                    success = process_frame_and_display(frame)
                    if success:
                        st.session_state.frame_idx += 1

    # Continue processing
    if st.session_state.running:
        time.sleep(0.01)
        st.rerun()

else:
    # Not running - show last frame or placeholder
    if st.session_state.current_frame is not None:
        # Display the last captured frame
        display_rgb = cv2.cvtColor(st.session_state.current_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(display_rgb)
        frame_placeholder.image(img_pil, use_container_width=True)
        # Display last alert status
        alert_box.markdown(f"### {st.session_state.last_alert}")
    else:
        # Show placeholder if no frame has been captured yet
        placeholder_img = Image.new("RGB", (640, 360), (30, 30, 30))
        frame_placeholder.image(placeholder_img, use_container_width=True)
        alert_box.info("ℹ️ Click 'Start Scan' to begin weapon detection")