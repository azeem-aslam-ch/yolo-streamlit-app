# 🧠 YOLOv8 Defect Detection (Single Image App)

This is a simple and intuitive **YOLOv8-based web app** built using **Streamlit** for detecting defects in a single uploaded image. It highlights detected regions with bounding boxes and provides a detailed table of results.

---

## 🚀 Features

- 📤 Upload **one image** at a time
- ✅ Detect defects using a YOLOv8 trained model
- 🖼️ View bounding boxes on the image
- 📋 See a **detection summary table** with:
  - Detected class (defect)
  - Confidence score
  - Bounding box coordinates
- 💾 Download the processed image with annotations

---

## 🧪 How It Works

1. Upload an image (`.jpg`, `.jpeg`, or `.png`)
2. The app runs the YOLOv8 model on it
3. The image is shown with bounding boxes
4. A table shows all detected objects and details
5. Optionally, download the image

---

## 📂 Project Structure

| File              | Description                              |
|-------------------|------------------------------------------|
| `app.py`          | Main Streamlit app file                  |
| `best.pt`         | YOLOv8 trained weights                   |
| `requirements.txt`| Python dependencies for Streamlit        |
| `packages.txt`    | System packages for OpenCV on Streamlit Cloud |
| `README.md`       | This file                                |

---

## 🛠️ Setup Instructions

### 1. Install Python packages

```bash
pip install -r requirements.txt


2. Run the app

streamlit run app.py



🌐 Deploy to Streamlit Cloud
Push this code to a public GitHub repository

Go to streamlit.io/cloud

Select your repo and point to app.py

Add requirements.txt and packages.txt for dependencies



