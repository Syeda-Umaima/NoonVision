import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pyttsx3  # For voice output

# ------------------------------
# Load YOLOv8 small pre-trained model
# ------------------------------
model = YOLO("yolov8n.pt")  # small, fast model

# ------------------------------
# Initialize TTS engine
# ------------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # speech speed

# ------------------------------
# Object Detection Function
# ------------------------------
def detect_objects(image):
    """
    Input: PIL Image
    Output: Image with bounding boxes, Text description
    Also speaks out the detected objects
    """
    if image is None:
        return None, "No image provided."

    img = np.array(image)  # Convert to numpy array
    results = model(img)[0]  # Run detection

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

        # Draw rectangle & label
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), f"{label} {conf:.2f}", fill="red", font=font)

    description = "Detected objects: " + ", ".join(detected_objects) if detected_objects else "No objects detected."

    # Speak detected objects (voice)
    if detected_objects:
        engine.say("Detected objects are " + ", ".join([label.split("(")[0] for label in detected_objects]))
        engine.runAndWait()

    return image, description
  

# ------------------------------
# Gradio Interface
# ------------------------------
iface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil", source="webcam"),
    outputs=[gr.Image(type="pil"), gr.Textbox()],
    title="NoonVision – AI Vision for the Visually Impaired",
    description="Bring an object in front of your webcam. NoonVision detects it and tells you what it is, both visually and via voice!"
)

iface.launch()
