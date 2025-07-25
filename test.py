from ultralytics import YOLO
import cv2

# Load the model
model = YOLO("best.pt")

# Run inference on a local image
results = model.predict(source="test.jpg", show=True, conf=0.4)

# Save result image
results[0].save(filename="output.jpg")
