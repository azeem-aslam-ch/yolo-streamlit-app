import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# Set Streamlit layout
st.set_page_config(page_title="YOLOv8 Realtime + Batch", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📹 YOLOv8 Webcam & Batch Detection</h1>", unsafe_allow_html=True)

# Load model
model = YOLO("best.pt")

# Sidebar options
mode = st.sidebar.radio("Choose Mode", ["📷 Webcam Detection", "🖼️ Upload Multiple Images"])

# Webcam Detection Mode
if mode == "📷 Webcam Detection":
    st.markdown("### Live Webcam Feed (press Stop to end)")
    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame.")
            break
        results = model(frame, conf=0.5)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            text = f"{label.upper()} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    cap.release()

# Multi-image Batch Detection Mode
elif mode == "🖼️ Upload Multiple Images":
    uploaded_files = st.sidebar.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            st.markdown(f"### 🖼️ {uploaded_file.name}")
            st.image(image, width=400)

            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            results = model(img_bgr, conf=0.5)[0]

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                text = f"{label.upper()} {conf:.2f}"
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.putText(img_bgr, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            result_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(result_img, caption="Detection Result", use_column_width=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Made with ❤️ by Azeem Aslam | Powered by YOLOv8 + Streamlit</p>", unsafe_allow_html=True)
