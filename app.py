import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import io

# Page config
st.set_page_config(page_title="YOLOv8 Detector", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🔍 YOLOv8 Object Detection</h1>", unsafe_allow_html=True)

# Load model
model = YOLO("best.pt")

# Sidebar image upload
st.sidebar.header("📤 Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.markdown("### 📸 Uploaded Image")
    st.image(image, use_container_width=True)

    with st.spinner("Running YOLOv8 Detection..."):
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        results = model(img_bgr)[0]

        detection_data = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            text = f"{label.upper()} {conf:.2f}"
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(img_bgr, text, (x1, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3, cv2.LINE_AA)

            detection_data.append({
                "Label": label,
                "Confidence": round(conf, 2),
                "X1": int(x1), "Y1": int(y1), "X2": int(x2), "Y2": int(y2)
            })

        result_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    st.markdown("### ✅ Detection Result")
    st.image(result_img, use_container_width=True)

    # Show summary table
    if detection_data:
        st.markdown("### 📊 Detection Summary")
        df = pd.DataFrame(detection_data)
        st.dataframe(df, use_container_width=True)

        # Convert to downloadable image
        is_success, buffer = cv2.imencode(".png", result_img)
        io_buf = io.BytesIO(buffer)

        st.download_button(
            label="💾 Download Result Image",
            data=io_buf,
            file_name="detection_result.png",
            mime="image/png"
        )
else:
    st.markdown("📥 Upload an image from the sidebar to get started.")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Made with ❤️ by Azeem Aslam | Powered by YOLOv8 + Streamlit</p>", unsafe_allow_html=True)
