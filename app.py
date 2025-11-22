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
    Output:
        - Annotated Image
        - Text description with confidence for helpers
        - Autoplay Audio with object names only
    """
    if image is None:
        return None, "No image provided.", None

    img = np.array(image)
    results = model(img)[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    labels = results.names
    confidences = results.boxes.conf.cpu().numpy()

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    detected_objects = []
    detected_objects_audio = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cls_id = int(results.boxes.cls[i])
        label = labels[cls_id]
        conf = confidences[i]

        # Keep for text display (with confidence)
        detected_objects.append(f"{label}: {conf:.2f}")

        # Keep only object names for audio
        detected_objects_audio.append(label)

        # Draw bounding box + confidence
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), f"{label} {conf:.2f}", fill="red", font=font)

    # ----------------------
    # Prepare description and TTS audio
    # ----------------------
    description_text = "\n".join(detected_objects) if detected_objects else "No objects detected."
    
    if detected_objects_audio:
        tts_text = "Detected objects are: " + ", ".join(detected_objects_audio)
        tts = gTTS(text=tts_text, lang='en')
        audio_file = "detected_objects.mp3"
        tts.save(audio_file)
    else:
        audio_file = None

    return image, description_text, audio_file

# ----------------------
# Gradio Interface
# ----------------------
iface = gr.Interface(
    fn=detect_objects_with_voice,
    inputs=gr.Image(type="pil", source="webcam"),
    outputs=[
        gr.Image(type="pil"),
        gr.Textbox(lines=10, label="Detected Objects (with confidence)"),
        gr.Audio(type="filepath", autoplay=True)  # autoplay for hands-free experience
    ],
    title="NoonVision – AI Vision for the Visually Impaired",
    description="Point your webcam at an object. NoonVision detects it, shows bounding boxes, lists detected objects with confidence for helpers, and speaks the objects automatically!"
)

iface.launch()
