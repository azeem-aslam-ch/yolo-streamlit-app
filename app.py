import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import io
import matplotlib.pyplot as plt
import random

# Set page layout
st.set_page_config(page_title="YOLOv8 Detection App", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🔍 YOLOv8 Object Detection</h1>", unsafe_allow_html=True)

# Load YOLOv8 model
model = YOLO("best.pt")

# Sidebar inputs
st.sidebar.header("📤 Upload and Filter")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

# Color map per class
colors = {}
for i, name in model.names.items():
    colors[name] = [random.randint(0, 255) for _ in range(3)]

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.markdown("### 📸 Uploaded Image")
    st.image(image, use_container_width=True)

    with st.spinner("Running Detection..."):
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        results = model(img_bgr, conf=conf_threshold)[0]

        detection_data = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            color = colors[label]
            text = f"{label.upper()} {conf:.2f}"
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 4)
            cv2.putText(img_bgr, text, (x1, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3, cv2.LINE_AA)

            detection_data.append({
                "Detected Defect": label,
                "Confidence Score (0-1)": round(conf, 2),
                "Top-Left X": x1, "Top-Left Y": y1,
                "Bottom-Right X": x2, "Bottom-Right Y": y2
            })

        result_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    st.markdown("### ✅ Detection Result")
    st.image(result_img, use_container_width=True)

    if detection_data:
        df = pd.DataFrame(detection_data)
        st.markdown("### 📊 Detection Summary (Explanation Table)")
        st.dataframe(df, use_container_width=True)

        is_success, buffer = cv2.imencode(".png", result_img)
        io_buf = io.BytesIO(buffer)
        st.download_button("💾 Download Result Image", data=io_buf, file_name="detection_result.png", mime="image/png")

        st.markdown("### 📈 Defect Frequency Chart")
        fig, ax = plt.subplots()
        df["Detected Defect"].value_counts().plot(kind="bar", color="skyblue", ax=ax)
        ax.set_title("Detected Defect Types")
        ax.set_xlabel("Defect Type")
        ax.set_ylabel("Count")
        st.pyplot(fig)
else:
    st.markdown("📥 Upload an image from the sidebar to get started.")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Made with ❤️ by Azeem Aslam | Powered by YOLOv8 + Streamlit</p>", unsafe_allow_html=True)
