import gradio as gr
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import time
from transformers import pipeline
from collections import Counter
import torch

# ======================
# CONFIGURATION
# ======================
CONF_THRESHOLD = 0.25
IMG_SIZE = 640  # smaller size for faster CPU inference
BOX_COLOR = (255, 50, 50)
BOX_WIDTH = 3
FONT_SIZE = 18

# ======================
# MODEL INITIALIZATION
# ======================
print("🔄 Loading YOLOv8m (CPU)...")
try:
    model = YOLO("yolov8m.pt")
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    _ = model(dummy, verbose=False)
    print("✅ YOLOv8m loaded successfully on CPU")
except Exception as e:
    print(f"❌ YOLOv8m failed to load: {e}")
    model = None

# Whisper STT (CPU)
print("🔄 Loading Whisper STT (CPU)...")
device = -1  # CPU
try:
    stt_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device=device)
    print("✅ Whisper STT loaded on CPU")
except Exception as e:
    print(f"❌ Whisper failed: {e}")
    stt_pipe = None

# ======================
# TRIGGER PHRASES
# ======================
TRIGGER_PHRASES = ["detect", "what do you see", "what's in front of me",
                   "what is in front of me", "identify objects", "what's this",
                   "what is this", "tell me what you see", "scan", "look"]

# ======================
# AUDIO GENERATION
# ======================
def generate_audio_description(labels):
    try:
        if not labels:
            tts_text = "I couldn't detect any objects. Please try again."
        else:
            label_counts = Counter(labels)
            items = []
            for obj, count in label_counts.items():
                items.append(f"{count} {obj}{'' if count==1 else 's'}")
            tts_text = "I see " + ", ".join(items) + "."

        timestamp = int(time.time() * 1000)
        audio_file = f"detected_{timestamp}.mp3"
        tts = gTTS(text=tts_text, lang='en', slow=False)
        tts.save(audio_file)
        return audio_file
    except Exception as e:
        print(f"⚠️ Audio generation error: {e}")
        return None

# ======================
# OBJECT DETECTION
# ======================
def detect_objects(image, conf_threshold=CONF_THRESHOLD):
    if image is None or model is None:
        return None, None

    try:
        img_np = np.array(image)
        img_pil = image.copy()
        results = model(img_np, imgsz=IMG_SIZE, conf=conf_threshold, verbose=False)[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        labels = results.names
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()

        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", FONT_SIZE)
        except:
            font = ImageFont.load_default()

        detected_labels = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(class_ids[i])
            label = labels[cls_id]
            conf = confidences[i]
            detected_labels.append(label)
            draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)
            text = f"{label} {conf:.2f}"
            bbox = draw.textbbox((x1, y1 - 20), text, font=font)
            draw.rectangle(bbox, fill=BOX_COLOR)
            draw.text((x1, y1 - 20), text, fill="white", font=font)

        audio_file = generate_audio_description(detected_labels)
        return img_pil, audio_file
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return None, None

# ======================
# STREAMING HANDLERS
# ======================
def transcribe_streaming_audio(audio_tuple):
    if audio_tuple is None or stt_pipe is None:
        return ""
    try:
        sample_rate, audio_data = audio_tuple
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        audio_data = audio_data.astype(np.float32)
        if audio_data.max() > 1.0:
            audio_data = audio_data / 32768.0
        result = stt_pipe({"sampling_rate": sample_rate, "raw": audio_data})
        return result["text"].strip().lower()
    except Exception as e:
        print(f"⚠️ Voice processing error: {e}")
        return ""

def process_camera(latest_frame, transcribed_text, last_image):
    if latest_frame is None:
        return latest_frame, last_image, None, "Waiting for Camera...", ""
    triggered = any(phrase in transcribed_text for phrase in TRIGGER_PHRASES)
    if triggered:
        annotated_img, audio_file = detect_objects(latest_frame, CONF_THRESHOLD)
        return latest_frame, annotated_img, audio_file, f"✅ Detected objects!", ""
    return latest_frame, last_image, None, "🎤 Listening for trigger phrase...", transcribed_text

# ======================
# GRADIO INTERFACE
# ======================
AUTO_JS = """
setTimeout(function(){
    const mic = document.querySelector("button[data-testid='microphone-button']");
    if(mic){ mic.click(); console.log("Auto-start microphone"); }
}, 500);
"""

with gr.Blocks() as demo:
    gr.HTML("<h2 style='text-align:center'>🦾 NoonVision - CPU Hands-Free AI Assistant</h2>")

    with gr.Row():
        image_input = gr.Image(type="pil", sources="webcam", streaming=True, interactive=False, height=400)
        image_output = gr.Image(type="pil", height=400, value=None)

    status_output = gr.Textbox(label="", value="Waiting for permissions...", lines=2)
    voice_input = gr.Audio(sources="microphone", type="numpy", streaming=True, visible=False)
    transcribed_state = gr.Textbox(visible=False)
    audio_output = gr.Audio(type="filepath", autoplay=True, visible=False)

    # Voice transcription
    voice_input.stream(transcribe_streaming_audio, inputs=[voice_input], outputs=[transcribed_state], show_progress=False)
    # Camera streaming
    image_input.stream(process_camera, inputs=[image_input, transcribed_state, image_output],
                       outputs=[image_input, image_output, audio_output, status_output, transcribed_state],
                       every=0.1, show_progress=False)

# ======================
# LAUNCH
# ======================
if __name__ == "__main__":
    demo.launch()
