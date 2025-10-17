import streamlit as st
import cv2
import numpy as np
import time
from model.attention_model import AttentionDetector
from model.fusion_model import FusionModel

st.set_page_config(page_title="Autism Learning", layout="wide")
st.title("Autism Learning")

# --- Sidebar: student selection & controls ---
st.sidebar.header("Controls")
student_id = st.sidebar.selectbox("Select Student", [1, 2, 3])
start_cam = st.sidebar.button("Start Camera")
stop_cam = st.sidebar.button("Stop Camera")

if "running" not in st.session_state:
    st.session_state.running = False

if start_cam:
    st.session_state.running = True
if stop_cam:
    st.session_state.running = False

# --- Placeholders ---
frame_window = st.image([])
status_placeholder = st.sidebar.empty()
mode_placeholder = st.sidebar.empty()

# --- Initialize models ---
detector = AttentionDetector()
fusion = FusionModel(student_id)

# --- Start webcam if running ---
if st.session_state.running:
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        st.error("⚠️ Cannot access webcam. Allow camera permissions.")
        st.stop()

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera disconnected.")
            break

        # Analyze attention
        att, overlay = detector.analyze_frame(frame)
        engagement, mode = fusion.combine(att)

        # Display video + metrics
        frame_window.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        status_placeholder.metric("Engagement", f"{engagement:.2f}")
        mode_placeholder.text(f"Mode: {mode}")

        # Slight delay for stability
        time.sleep(0.1)

    cap.release()
    frame_window.empty()
