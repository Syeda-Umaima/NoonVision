import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from gtts import gTTS
import soundfile as sf
import io
import tempfile

# ---------------------------
# Load YOLOv8m model (CPU)
# ---------------------------
model = YOLO("yolov8m.pt")   # your medium model

# -----------------------------------
# Function: Detect objects in image
# -----------------------------------
def detect_objects(image):
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

        detected_objects.append(f"{label} ({conf:.2f})")

        # Draw bounding boxes
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), f"{label} {conf:.2f}", fill="red", font=font)

    # Prepare description text
    if detected_objects:
        text_description = "Detected objects: " + ", ".join(detected_objects)
        speech_text = "Detected objects are " + ", ".join([obj.split("(")[0] for obj in detected_objects])
    else:
        text_description = "No objects detected."
        speech_text = "No objects detected."

    # Convert speech to audio automatically
    tts = gTTS(speech_text)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tts.save(temp_audio.name)

    data, samplerate = sf.read(temp_audio.name)
    audio_bytes = io.BytesIO()
    sf.write(audio_bytes, data, samplerate, format='wav')

    return image, text_description, audio_bytes.getvalue()


# ---------------------------------------------------
# Voice command system
# Detect when user says “detect” → capture image auto
# ---------------------------------------------------
def voice_controller(audio, image):
    import speech_recognition as sr
    recog = sr.Recognizer()

    if audio is None:
        return gr.update(), gr.update(), None

    # Convert to speech Recognizer-friendly file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio)
        audio_file_path = f.name

    with sr.AudioFile(audio_file_path) as source:
        audio_data = recog.record(source)

    try:
        text = recog.recognize_google(audio_data).lower()
    except:
        return image, "Could not understand voice. Try again.", None

    # If user says "detect", perform detection automatically
    if "detect" in text:
        if image is None:
            return image, "Say 'detect' after camera captures image.", None
        return detect_objects(image)

    return image, "Say 'detect' to perform analysis.", None


# ---------------------------
# Gradio User Interface
# ---------------------------
with gr.Blocks(title="NoonVision – Voice Enabled AI") as demo:

    gr.Markdown(
        "<h1 style='text-align:center;'>NoonVision – AI Object Detection for the Visually Impaired</h1>"
        "<p style='font-size:22px; text-align:center;'>Say <b>Detect</b> to capture & analyze automatically.</p>"
    )

    with gr.Row():
        webcam = gr.Image(source="webcam", type="pil", label="Camera View")
        audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Say: Detect")

    output_img = gr.Image(label="Detection Result")
    output_text = gr.Textbox(label="Detected Objects (Accuracy Shown Here)")
    output_audio = gr.Audio(label="Voice Output", autoplay=True)

    audio_in.change(
        fn=voice_controller,
        inputs=[audio_in, webcam],
        outputs=[output_img, output_text, output_audio]
    )

demo.launch()
