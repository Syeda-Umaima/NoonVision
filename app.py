import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Load pre-trained YOLOv8 small model
model = YOLO("yolov8n.pt")  # small, fast model

def detect_objects(image):
    """
    Input: PIL Image
    Output: Image with bounding boxes, Text description
    """
    results = model(image)  # runs inference

    boxes = results[0].boxes.xyxy.cpu().numpy()  # bounding boxes
    labels = results[0].names
    confidences = results[0].boxes.conf.cpu().numpy()  # confidence scores

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    detected_objects = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cls_id = int(results[0].boxes.cls[i])
        label = labels[cls_id]
        conf = confidences[i]
        detected_objects.append(f"{label} ({conf:.2f})")

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), f"{label} {conf:.2f}", fill="red", font=font)

    description = "Detected objects: " + ", ".join(detected_objects) if detected_objects else "No objects detected."
    return image, description

# Gradio Interface
iface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil", source="webcam"),
    outputs=[gr.Image(type="pil"), gr.Textbox()],
    title="NoonVision – AI Vision for the Visually Impaired",
    description="Bring an object in front of your webcam. NoonVision detects it and tells you what it is!"
)

iface.launch()
