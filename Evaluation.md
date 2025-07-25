# 📊 Model Evaluation Report

## 🔧 Model Details
- Architecture: YOLOv8n
- Pretrained: yolov8n.pt
- Image Size: 640x640
- Training Epochs: 20
- Dataset Size: 243 images (203 train / 21 val / 19 test)

---

## 📈 Overall Performance

| Metric    | Value  |
|-----------|--------|
| mAP@0.5   | 0.9347 |
| Precision | 0.9614 |
| Recall    | 0.8955 |

---

## 🔍 Class-wise Average Precision

| Class         | AP@0.5 |
|---------------|--------|
| cereal-box    | 0.995  |
| drink_cans    | 0.994  |
| waterbottle   | 0.815  |

---

## 🧪 Test Results Summary
- Evaluated on 19 test images.
- All three classes were successfully detected.
- Model struggles slightly with waterbottles under occlusion.

---

## 📌 Observations
- Best performance: cereal-boxes and drink_cans in well-lit conditions.
- Weakest performance: waterbottles when partially hidden.
- No major false positives observed.

---

## 💡 Future Improvements
- Augment more occluded waterbottle images.
- Consider switching to YOLOv8m or v8l if latency allows.
- Try training longer (e.g., 40 epochs) for marginal improvements.

---

## 🔁 Version
- YOLOv8: Ultralytics v8.x
- Python: 3.10
