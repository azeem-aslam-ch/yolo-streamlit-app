import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

model = YOLO("best.pt")  # Make sure this file is in the same directory

st.title("YOLOv8 Object Detection Web App")
st.write("Upload an image and detect objects using YOLOv8!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    results = model(img_bgr)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_bgr, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Detection Result", use_column_width=True)
