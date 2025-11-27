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
import threading

st.set_page_config(page_title="Detective-AI - Weapon Detector", layout="wide")

# ---------------- Settings ----------------
MODEL_PATHS = [
    Path("models/weapon_best.pt"),
    Path("runs/detect/weapon_s_model/weights/best.pt"),
    Path("runs/detect/weapon_s_model2/weights/best.pt"),
]
IMG_SIZE = 416
CONF_THR = 0.25
IOU_THR = 0.45
WEAPON_CLASS_INDEX = 1
WEBCAM_FPS = 30  # Target FPS for webcam
VIDEO_FPS = 30   # Target FPS for video playback

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
            "No trained model file found. Using 'yolov8n.pt' as fallback. "
            "Place your trained model in models/weapon_best.pt"
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

# ---------------- Session State ----------------
if "running" not in st.session_state:
    st.session_state.running = False
if "events_log" not in st.session_state:
    st.session_state.events_log = ""
if "video_temp_path" not in st.session_state:
    st.session_state.video_temp_path = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None
if "last_alert" not in st.session_state:
    st.session_state.last_alert = "ℹ️ Click 'Start Scan' to begin weapon detection"
if "detection_count" not in st.session_state:
    st.session_state.detection_count = 0
if "video_cap" not in st.session_state:
    st.session_state.video_cap = None
if "video_total_frames" not in st.session_state:
    st.session_state.video_total_frames = 0
if "video_current_frame" not in st.session_state:
    st.session_state.video_current_frame = 0

# ---------------- UI Layout ----------------
st.title("🔴 Detective-AI — Live Weapon Detector")
col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("### Alerts & Controls")
    st.caption("Model: " + (os.path.basename(model_path) if model_path else "fallback yolov8n"))
    
    col_start, col_stop = st.columns(2)
    with col_start:
        start_button = st.button("▶️ Start", type="primary", use_container_width=True)
    with col_stop:
        stop_button = st.button("⏹️ Stop", type="secondary", use_container_width=True)
    
    if st.button("🗑️ Clear Log", use_container_width=True):
        st.session_state.events_log = ""
        st.session_state.detection_count = 0
        st.session_state.last_alert = "ℹ️ Event log cleared"
        st.rerun()
    
    st.write("---")
    st.markdown("**Input Source**")
    source_type = st.selectbox("Source:", ["Webcam (live)", "Upload video"])
    
    uploaded_file = None
    if source_type == "Upload video":
        uploaded_file = st.file_uploader(
            "Upload video file", 
            type=["mp4", "avi", "mov", "mkv"],
            help="Max 200MB recommended"
        )
    
    st.write("---")
    st.markdown("**Statistics**")
    stats_placeholder = st.empty()
    
    st.write("---")
    st.markdown("**Settings**")
    st.write(f"- Image size: {IMG_SIZE}")
    st.write(f"- Confidence: {CONF_THR}")
    st.write(f"- IoU threshold: {IOU_THR}")
    st.write(f"- Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    st.write("---")
    st.markdown("**Event Log**")
    events_out = st.empty()

with col1:
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    alert_placeholder = st.empty()

# ---------------- File Upload Handler ----------------
if uploaded_file is not None:
    if st.session_state.get("uploaded_name") != uploaded_file.name:
        # Close existing video capture
        if st.session_state.video_cap is not None:
            st.session_state.video_cap.release()
            st.session_state.video_cap = None
        
        # Save new file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        tfile.write(uploaded_file.read())
        tfile.flush()
        tfile.close()
        st.session_state.video_temp_path = tfile.name
        st.session_state.uploaded_name = uploaded_file.name
        st.session_state.video_current_frame = 0
        st.success(f"✅ Uploaded: {uploaded_file.name}")

# ---------------- Detection Function ----------------
def detect_objects(frame_bgr):
    """Run YOLO detection on frame and return annotated frame + detection info"""
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
        return frame_bgr, False, 0
    
    display_bgr = frame_bgr.copy()
    weapon_detected = False
    weapon_count = 0
    
    if results and len(results):
        r = results[0]
        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes):
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
                    weapon_detected = True
                    weapon_count += 1
                    color = (0, 0, 255)  # Red for weapons
                    thickness = 3
                    cv2.rectangle(display_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    label = f"WEAPON {conf:.2f}"
                    
                    # Add background to text
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(display_bgr, (int(x1), int(y1)-label_h-10), (int(x1)+label_w, int(y1)), color, -1)
                    cv2.putText(display_bgr, label, (int(x1), int(y1)-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    color = (0, 255, 0)  # Green for other objects
                    cv2.rectangle(display_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = model.names.get(cls, f"cls{cls}") if hasattr(model, 'names') else f"cls{cls}"
                    cv2.putText(display_bgr, f"{label} {conf:.2f}", (int(x1), max(20, int(y1)-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    return display_bgr, weapon_detected, weapon_count

# ---------------- Display Event Log ----------------
def display_event_log():
    if st.session_state.events_log:
        # Show last 10 events
        lines = st.session_state.events_log.strip().split('\n')
        recent_events = '\n'.join(lines[-15:])  # Last 15 events
        events_out.text_area(
            "Recent Detections",
            recent_events,
            height=250,
            disabled=True
        )
    else:
        events_out.info("No weapons detected yet")

# ---------------- Statistics Display ----------------
def update_stats(fps=0, frame_num=0, total_frames=0):
    stats_text = f"""
    **Detections:** {st.session_state.detection_count}
    **FPS:** {fps:.1f}
    """
    if total_frames > 0:
        progress = (frame_num / total_frames) * 100
        stats_text += f"\n**Progress:** {frame_num}/{total_frames} ({progress:.1f}%)"
    stats_placeholder.markdown(stats_text)

# Always display event log
display_event_log()

# ---------------- Start/Stop Logic ----------------
if start_button:
    if st.session_state.running:
        status_placeholder.warning("⚠️ Already running")
    else:
        if source_type == "Upload video" and st.session_state.video_temp_path is None:
            status_placeholder.error("❌ Please upload a video file first")
        else:
            st.session_state.running = True
            st.session_state.video_current_frame = 0
            status_placeholder.success("✅ Scanning started...")
            time.sleep(0.3)
            st.rerun()

if stop_button:
    if st.session_state.running:
        st.session_state.running = False
        if st.session_state.video_cap is not None:
            st.session_state.video_cap.release()
            st.session_state.video_cap = None
        status_placeholder.info("⏸️ Scanning stopped")
    else:
        status_placeholder.info("ℹ️ Not running")

# ---------------- Main Processing Loop ----------------
if st.session_state.running:
    
    if source_type == "Webcam (live)":
        # === WEBCAM MODE ===
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, WEBCAM_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for real-time
        
        if not cap.isOpened():
            status_placeholder.error("❌ Cannot access webcam")
            st.session_state.running = False
        else:
            frame_count = 0
            start_time = time.time()
            
            # Read and process frame
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                # Detect objects
                annotated_frame, weapon_found, weapon_count = detect_objects(frame)
                
                # Display frame
                display_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(display_rgb, use_container_width=True)
                
                # Update alert
                if weapon_found:
                    st.session_state.detection_count += weapon_count
                    timestamp = time.strftime('%H:%M:%S')
                    log_entry = f"🔴 {timestamp} - {weapon_count} weapon(s) detected\n"
                    st.session_state.events_log += log_entry
                    st.session_state.last_alert = f"🔴 WEAPON DETECTED! (Count: {weapon_count})"
                    alert_placeholder.error(f"🚨 WEAPON DETECTED! Total: {st.session_state.detection_count}")
                    display_event_log()  # Update log immediately
                else:
                    st.session_state.last_alert = "✅ No weapon detected"
                    alert_placeholder.success("✅ No threats detected")
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                update_stats(fps=fps)
                
                status_placeholder.success(f"📹 Live Webcam - FPS: {fps:.1f}")
            
            # Continue loop
            time.sleep(1.0 / WEBCAM_FPS)
            st.rerun()
    
    else:
        # === VIDEO FILE MODE ===
        if st.session_state.video_temp_path is None:
            status_placeholder.error("❌ No video uploaded")
            st.session_state.running = False
        else:
            # Initialize video capture once
            if st.session_state.video_cap is None:
                st.session_state.video_cap = cv2.VideoCapture(st.session_state.video_temp_path)
                st.session_state.video_total_frames = int(st.session_state.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                st.session_state.video_current_frame = 0
            
            cap = st.session_state.video_cap
            
            if not cap.isOpened():
                status_placeholder.error("❌ Cannot open video file")
                st.session_state.running = False
            else:
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                frame_delay = 1.0 / VIDEO_FPS if VIDEO_FPS > 0 else 0
                
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # Detect objects
                    annotated_frame, weapon_found, weapon_count = detect_objects(frame)
                    
                    # Display frame
                    display_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(display_rgb, use_container_width=True)
                    
                    # Update alert
                    if weapon_found:
                        st.session_state.detection_count += weapon_count
                        timestamp = time.strftime('%H:%M:%S')
                        frame_time = st.session_state.video_current_frame / video_fps if video_fps > 0 else 0
                        log_entry = f"🔴 {timestamp} [Frame {st.session_state.video_current_frame}, {frame_time:.1f}s] - {weapon_count} weapon(s)\n"
                        st.session_state.events_log += log_entry
                        alert_placeholder.error(f"🚨 WEAPON DETECTED! Total: {st.session_state.detection_count}")
                        display_event_log()  # Update log
                    else:
                        alert_placeholder.success("✅ No threats detected")
                    
                    # Update stats
                    st.session_state.video_current_frame += 1
                    update_stats(fps=VIDEO_FPS, frame_num=st.session_state.video_current_frame, 
                                total_frames=st.session_state.video_total_frames)
                    
                    status_placeholder.info(f"📹 Processing video... Frame {st.session_state.video_current_frame}/{st.session_state.video_total_frames}")
                    
                    # Control playback speed
                    time.sleep(frame_delay)
                    st.rerun()
                else:
                    # Video ended
                    cap.release()
                    st.session_state.video_cap = None
                    st.session_state.running = False
                    status_placeholder.success(f"✅ Video complete! Processed {st.session_state.video_current_frame} frames")
                    alert_placeholder.info(f"📊 Total detections: {st.session_state.detection_count}")

else:
    # Not running - show placeholder
    placeholder_img = Image.new("RGB", (1280, 720), (30, 30, 30))
    frame_placeholder.image(placeholder_img, use_container_width=True)
    alert_placeholder.info(st.session_state.last_alert)
    update_stats()