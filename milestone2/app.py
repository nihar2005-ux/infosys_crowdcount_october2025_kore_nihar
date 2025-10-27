import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ----------------------------
# 🎨 Page Setup
# ----------------------------
st.set_page_config(page_title="Person Detection & Counting", page_icon="", layout="wide")

# ----------------------------
# 🌈 Custom CSS Styling
# ----------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #f0f4f8;
        color: #111;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stApp h1 {
        color: #1e3a8a;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
    .css-1d391kg {
        background-color: #1f2937 !important;
        color: #f9fafb !important;
        padding: 20px !important;
    }
    .css-1d391kg h2 {
        color: #f3f4f6 !important;
        font-size: 1.5rem;
        text-align: center;
    }
    .stButton>button {
        background: #3b82f6;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 25px;
        margin-top: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: #2563eb;
        transform: translateY(-2px);
        box-shadow: 0px 5px 15px rgba(0,0,0,0.2);
    }
    .stFileUploader > label {
        font-weight: bold;
        color: #1e3a8a;
    }
    .stTabs [role="tab"] {
        background-color: #e0f2fe;
        color: #1e40af;
        font-weight: bold;
        border-radius: 8px;
        padding: 8px 20px;
        margin-right: 5px;
    }
    .stTabs [role="tab"]:hover {
        background-color: #3b82f6;
        color: #fff;
    }
    .stImage {
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.1);
    }
    .css-1kyxreq {
        font-size: 1.2rem !important;
        font-weight: bold !important;
        color: #1e3a8a !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# 🧠 Load YOLO Model
# ----------------------------
model_path = "yolov8n.pt"  # replace with 'best.pt' if using custom model
model = YOLO(model_path)

# ----------------------------
# 🧩 Detection Function
# ----------------------------
def detect_persons(image):
    results = model(image)
    detected_img = results[0].plot()
    person_count = 0
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        if "person" in label.lower():
            person_count += 1
    return detected_img, person_count

# ----------------------------
# 📊 Dashboard (Direct Access)
# ----------------------------
st.sidebar.title("Person Detection & Counting")
st.title("Person Detection & Counting")
st.write("Detect and count people using images, videos, or your webcam.")

mode = st.radio("Select Mode", ["Upload Image", "Upload Video", "Live Webcam"])

# Image Mode
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Detecting persons..."):
            img_array = np.array(image)
            output_img, count = detect_persons(img_array)
            st.image(output_img, caption=f"Detected Persons: {count}", use_column_width=True)
            st.success(f"Persons Detected: {count}")

# Video Mode
elif mode == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.video(video_path)
        stframe = st.empty()
        person_counter_placeholder = st.empty()
        cap = cv2.VideoCapture(video_path)

        st.success("Processing video...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            output_frame = results[0].plot()
            person_count = 0
            for box in results[0].boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if "person" in label.lower():
                    person_count += 1

            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            stframe.image(output_frame, channels="RGB", use_column_width=True)
            person_counter_placeholder.success(f"Persons Detected: {person_count}")

        cap.release()
        st.info("Video processing completed")

# Webcam Mode
elif mode == "Live Webcam":
    stframe = st.empty()
    person_counter_placeholder = st.empty()
    run = st.checkbox("Start Webcam")
    stop = st.button("Stop")

    camera = cv2.VideoCapture(0)
    while run and not stop:
        ret, frame = camera.read()
        if not ret:
            st.warning("Failed to access webcam.")
            break

        results = model(frame)
        output_frame = results[0].plot()
        person_count = 0
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if "person" in label.lower():
                person_count += 1

        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        stframe.image(output_frame, channels="RGB", use_column_width=True)
        person_counter_placeholder.success(f"Persons Detected: {person_count}")

        time.sleep(0.1)

    camera.release()
    st.info("Webcam stopped")

st.markdown("---")
