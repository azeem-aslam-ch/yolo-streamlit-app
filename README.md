# YOLOv8 Defect Detection Streamlit App

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37-red.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An intuitive web application built with Streamlit and powered by YOLOv8 for real-time defect detection in images. Upload an image, and the app will instantly identify and highlight defects, providing detailed analysis and results.

---



## 🚀 Overview

This project provides a simple yet powerful tool for industrial quality control and other defect detection tasks. The application allows users to upload an image and receive immediate visual feedback with bounding boxes drawn around detected defects. It also generates a comprehensive data table with class predictions, confidence scores, and coordinates for each detected object.

### ✨ Key Features

* **Easy Image Upload:** Supports single image uploads (`.jpg`, `.jpeg`, `.png`).
* **High-Performance Detection:** Utilizes a pre-trained YOLOv8 model for fast and accurate defect detection.
* **Interactive Visualization:** Displays the output image with clear bounding boxes and labels.
* **Detailed Analysis:** Provides a summary table of all detected defects with their confidence scores and locations.
* **Downloadable Results:** Allows users to download the annotated image for reporting and analysis.

---

## 🛠️ Tech Stack

This project is built with the following technologies:

* **Python:** Core programming language.
* **Streamlit:** For building the interactive web application.
* **PyTorch:** As the backend for the YOLOv8 model.
* **YOLOv8:** State-of-the-art object detection model from Ultralytics.
* **OpenCV:** For image processing and handling.
* **Docker:** For containerizing the application for easy deployment and scalability.

---

## ⚙️ Getting Started

You can run this application in two ways: locally using a Python environment or with Docker.

### 1. Local Setup Instructions

**Prerequisites:**
-   Python 3.10 or later
-   `pip` package manager

**Installation:**
1.  Clone the repository to your local machine:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

**Running the App:**
1.  Launch the Streamlit app from your terminal:
    ```bash
    streamlit run app.py
    ```
2.  The application will open in your default web browser.

### 2. 🐳 Run with Docker

If you have Docker installed, you can run the application in a container with a single command, without worrying about local dependencies.

1.  **Pull the pre-built image from Docker Hub:**
    ```bash
    docker pull azeemaslamch/yolo-streamlit-app:latest
    ```
2.  **Run the container:**
    ```bash
    docker run -p 8501:8501 azeemaslamch/yolo-streamlit-app:latest
    ```
3.  Open your web browser and navigate to `http://localhost:8501`.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
