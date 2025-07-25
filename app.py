import streamlit as st
st.set_page_config(page_title="Retail Shelf Detector", layout="centered")  # Must be first

from PIL import Image
import tempfile
import os
from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # Adjust path to your model
    return model

model = load_model()

# Streamlit UI
st.title("🛒 Retail Shelf Object Detection")
st.markdown("Detects **Cereal Box**, **Soda Can**, and **Water Bottle** using a YOLOv8 model.")

uploaded_file = st.file_uploader("Upload a shelf image (JPG/JPEG only)", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_image_path = tmp_file.name

    st.image(temp_image_path, caption="📤 Uploaded Image", use_column_width=True)

    # Run inference with a spinner
    with st.spinner("🔍 Detecting objects..."):
        results = model.predict(source=temp_image_path, conf=0.4, save=True)

    # Get the result image path
    result_dir = os.path.join("runs", "detect", "predict")
    result_image_path = os.path.join(result_dir, os.path.basename(temp_image_path))

    if os.path.exists(result_image_path):
        st.success("✅ Detection complete!")
        st.image(result_image_path, caption="📦 Detected Objects", use_column_width=True)
    else:
        st.warning("⚠️ No objects detected with confidence ≥ 0.4")
