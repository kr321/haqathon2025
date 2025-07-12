from ultralytics import YOLO

# Load pretrained YOLOv8n model
model = YOLO("yolov8n.pt")  # Ensure this model is downloaded beforehand

def detect_objects(image_path):
    results = model(image_path)[0]
    labels = set()
    for box in results.boxes:
        cls_id = int(box.cls.item())
        labels.add(model.names[cls_id])
    return list(labels)

