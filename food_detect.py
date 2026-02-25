from ultralytics import YOLO
import sys
import json
import os

# Get image path from PHP
image_path = sys.argv[1]

# Absolute model path (CRITICAL)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../ai_models/best_float16.tflite")

# Load model
model = YOLO(MODEL_PATH)

# Run detection
results = model(image_path, conf=0.25)

detections = []

for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])

        detections.append({
            "label": label,
            "confidence": round(conf, 2)
        })

# Return result
if detections:
    print(json.dumps({"detections": detections}))
else:
    print(json.dumps({"error": "No food detected"}))
