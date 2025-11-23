import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import soundfile as sf
import tempfile
from gtts import gTTS
import io

# Load YOLO model (keep yolov8m.pt or yolov8n.pt)
model = YOLO("yolov8m.pt")


def text_to_audio(text):
    """Convert text to audio bytes using gTTS (works on Hugging Face)."""
    tts = gTTS(text=text, lang='en')
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp.read()


def detect_objects(image):
    """
    Input: PIL image
    Output: annotated image, text results, auto-play audio
    """
    results = model(image)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    labels = results[0].names
    confidences = results[0].boxes.conf.cpu().numpy()

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    detected_objects = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cls_id = int(results[0].boxes.cls[i])
        label = labels[cls_id]
        conf = confidences[i]

        # Save only object names in audio (not accuracy)
        detected_objects.append(label)

        # Show accuracy with object name in big text for invigilators
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), f"{label} ({conf:.2f})", fill="red", font=font)

    if detected_objects:
        text = "Detected objects are: " + ", ".join(detected_objects)
    else:
        text = "No objects detected."

    # Generate audio
    audio_bytes = text_to_audio(text)

    return image, text, (48000, audio_bytes)


def voice_command(command, image):
    """
    Process speech commands.
    If user says 'detect', capture and detect automatically.
    """
    if command is None:
        return None, None, None
    
    command = command.lower()

    trigger_words = ["detect", "what is in front", "what’s in front", "find", "scan"]

    if any(key in command for key in trigger_words):
        return detect_objects(image)

    return None, "Say 'detect' to capture the image.", None


with gr.Blocks(title="NoonVision – AI Assistant for Visually Impaired") as app:

    gr.Markdown("""
    # 🦾 NoonVision
    Speak **"detect"** and the app will automatically capture the camera frame,  
    analyze it, and speak the result out loud.
    """)

    webcam = gr.Image(label="Camera", type="pil", source="webcam")
    voice = gr.Audio(source="microphone", type="filepath", label="Voice Command")
    
    output_img = gr.Image(label="Detected Image")
    output_text = gr.Textbox(label="Detection Result (with accuracy)")
    output_audio = gr.Audio(label="Voice Output", autoplay=True)

    btn = gr.Button("Process Voice Command")

    btn.click(fn=voice_command, inputs=[voice, webcam], outputs=[output_img, output_text, output_audio])

app.launch()
