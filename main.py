import streamlit as st
import cv2
import numpy as np
import time
from model.attention_model import AttentionDetector
from model.fusion_model import FusionModel

st.set_page_config(page_title="NeuroLearn Live", layout="wide")
st.title("🎥 Live Adaptive Learning – NeuroLearn AI")

# Sidebar controls
start = st.sidebar.button("Start Camera")
stop = st.sidebar.button("Stop Camera")

FRAME_WINDOW = st.image([])
status_placeholder = st.sidebar.empty()

detector = AttentionDetector()
fusion = FusionModel(student_id=1)

if "running" not in st.session_state:
    st.session_state.running = False

if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

cap = None

if st.session_state.running:
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        st.error("⚠️ Cannot access webcam. Please allow camera permissions.")
        st.stop()

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera disconnected.")
            break

        # Process attention level
        att = detector.analyze_frame(frame)
        engagement, mode = fusion.combine(att)

        # Display live frame and info
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        status_placeholder.metric("Engagement", f"{engagement:.2f}")
        # st.sidebar.write(f"Mode: {mode}")

        time.sleep(0.1)

    cap.release()
    FRAME_WINDOW.empty()
