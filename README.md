# 🏍️ Retail Shelf Object Detection using YOLOv8

This project implements a computer vision model that detects and localizes **Cereal Boxes**, **Soda Cans**, and **Water Bottles** in retail shelf images using the YOLOv8 object detection algorithm.

---

## 📁 Project Structure

```
retail-shelf-detector/
├── app.py               # Streamlit web app for detection
├── best.pt              # YOLOv8 trained weights
├── requirements.txt     # Python dependencies
├── render.yaml          # Deployment config for Render
├── sample_results/      # Inference outputs on test images
├── README.md            # Project overview and setup
├── evaluation.md        # Evaluation report
└── ...
```

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Jaswanth0128/Retail-Shelf-Detector7.git
cd Retail-Shelf-Detector
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🧐 Model Overview

* **Model**: YOLOv8m
* **Backbone**: CSPDarknet with SPPF and PANet
* **Training Epochs**: 30
* **Batch Size**: 16
* **Image Size**: 640x640
* **Augmentations**: Mosaic, random flip, HSV adjustments, scale
* **Pre-trained weights**: yolov8m.pt (Ultralytics)
* **Framework**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

---

## 📦 Dataset

1. **Total number of images**: 243

   * Train set: 203 images
   * Validation set: 21 images
   * Test set: 19 images

2. **Classes in the dataset**:

   * 0: None
   * 1: cereal-box
   * 2: drink\_cans
   * 3: waterbottle

3. **Source**:

   * Dataset created using existing datasets in Roboflow
   * 50 images per class (excluding "None") were selected
   * Additionally, 40+ images were collected from nearby retail stores featuring all 4 classes
   * Manual annotation (bounding boxes) was done for collected images using Roboflow

---

## 📊 Training Results

| Metric    | Value  |
| --------- | ------ |
| mAP\@0.5  | 0.9347 |
| Precision | 0.9614 |
| Recall    | 0.8955 |

### 🔎 Class-wise AP:

* **Cereal Box**: 0.995
* **Drink Cans**: 0.994
* **Water Bottle**: 0.815

---

## 📷 Sample Results

Images showing bounding boxes and detected labels:

![output](https://github.com/user-attachments/assets/34e39ca4-df59-436a-a5d6-c9b0804dbb71)
![WhatsApp Image 2025-07-25 at 18 43 50_f432fe6b](https://github.com/user-attachments/assets/3a718465-53a1-44b9-92e9-5b0a477cd1ad)
![WhatsApp Image 2025-07-25 at 18 45 03_bfd6a79c](https://github.com/user-attachments/assets/6a48c87a-2e2b-4b5a-a8c7-33152a132b4f)
<img width="1460" height="1990" alt="image" src="https://github.com/user-attachments/assets/ded62333-6c95-4c4d-b4f1-8247cb1736fb" />
<img width="1460" height="1095" alt="image" src="https://github.com/user-attachments/assets/8880ae69-75f7-443c-9b7a-e5ba20c72105" />



---

## 🧪 Testing Manually

To test an image using your trained model:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict(source="test.jpg", conf=0.4, save=True)
```

Output is saved in `runs/detect/predict/`.

---

## 🌐 Deployment

You can deploy this app on [Render](https://retail-shelf-dectector.onrender.com) using the provided `render.yaml`.

```yaml
services:
  - type: web
    name: retail-shelf-detector
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port 10000 --server.address 0.0.0.0
    envVars:
      - key: PYTHON_VERSION
        value: 3.10
```
The app may crash several times as I am using the Render free plan. Please try again after 30 seconds if the problem arise.
---

## 📄 License

This project is open-source under the MIT License.

---

## 🤛🏼‍♂️ Author

**Jaswanth Surya**
&#x20;       value: 3.10

---


Feel free to reach out or contribute!
