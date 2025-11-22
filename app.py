import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from gtts import gTTS
import os

# Load YOLOv8 small model
model = YOLO("yolov8n.pt")  # small, fast model

def detect_objects_with_voice(image):
    """
    Input: PIL Image
    Output: Annotated Image + Detected Objects List + TTS Audio
    """
    if image is None:
        return None, "No image provided.", None

    # Convert PIL → numpy
    img = np.array(image)
    results = model(img)[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    labels = results.names
    confidences = results.boxes.conf.cpu().numpy()

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    detected_objects = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cls_id = int(results.boxes.cls[i])
        label = labels[cls_id]
        conf = confidences[i]
        detected_objects.append(f"{label} ({conf:.2f})")

        # Draw rectangle + label
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), f"{label} {conf:.2f}", fill="red", font=font)

    description = "Detected objects: " + ", ".join(detected_objects) if detected_objects else "No objects detected."

    # ----------------------
    # Generate TTS audio
    # ----------------------
    if detected_objects:
        tts = gTTS(text=description, lang='en')
        audio_file = "detected_objects.mp3"
        tts.save(audio_file)
    else:
        audio_file = None

    return image, description, audio_file

# ----------------------
# Gradio Interface
# ----------------------
iface = gr.Interface(
    fn=detect_objects_with_voice,
    inputs=gr.Image(type="pil", source="webcam"),
    outputs=[gr.Image(type="pil"), gr.Textbox(), gr.Audio(type="filepath")],
    title="NoonVision – AI Vision for the Visually Impaired",
    description="Point your webcam at an object. NoonVision detects it, shows bounding boxes, and speaks the objects detected!"
)

iface.launch()
