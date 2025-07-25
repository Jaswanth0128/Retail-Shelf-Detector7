import streamlit as st
st.set_page_config(page_title="Retail Shelf Detector", layout="centered")  # Must be first

from PIL import Image
import tempfile
import os
from ultralytics import YOLO
import cv2
import shutil

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Make sure this path is correct

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

    # Optional: safely clear previous results
    try:
        shutil.rmtree("runs/detect", ignore_errors=True)
    except Exception as e:
        st.warning(f"⚠️ Could not clean old results: {e}")

    # Run inference with a spinner
    with st.spinner("🔍 Detecting objects..."):
        results = model.predict(source=temp_image_path, conf=0.4, save=True)

    # Get result image path dynamically
    save_path = results[0].save_dir
    result_image_path = os.path.join(save_path, os.path.basename(temp_image_path))

    if os.path.exists(result_image_path):
        st.success("✅ Detection complete!")
        st.image(result_image_path, caption="📦 Detected Objects", use_column_width=True)
    else:
        st.warning("⚠️ No objects detected with confidence ≥ 0.4")
