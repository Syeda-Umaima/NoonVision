import gradio as gr
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import os
import time
from transformers import pipeline
import torch
from collections import Counter

# ============================================
# CONFIGURATION
# ============================================
CONF_THRESHOLD = 0.25  # YOLO confidence threshold
IMG_SIZE = 640         # smaller size for faster CPU inference
BOX_COLOR = (255, 50, 50)
BOX_WIDTH = 3
FONT_SIZE = 20

# ============================================
# MODEL INITIALIZATION (CPU-Friendly)
# ============================================
print("🔄 Loading YOLOv8m (CPU)...")

# Use ultralytics hub model for CPU (avoids pickle errors)
try:
    model = YOLO("yolov8m")  # Auto-download weights
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    _ = model(dummy, verbose=False)
    print("✅ YOLOv8m loaded successfully on CPU")
except Exception as e:
    print(f"❌ YOLO load error: {e}")
    model = None

# Whisper (STT) pipeline, CPU-only
try:
    device = -1  # CPU
    stt_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        device=device
    )
    print("✅ Whisper STT loaded on CPU")
except Exception as e:
    print(f"⚠️ STT load error: {e}")
    stt_pipe = None

# ============================================
# TRIGGER PHRASES
# ============================================
TRIGGER_PHRASES = [
    "detect",
    "what do you see",
    "what's in front of me",
    "what is in front of me",
    "identify objects",
    "what's this",
    "what is this",
    "tell me what you see",
    "scan",
    "look"
]

# ============================================
# AUDIO & DETECTION FUNCTIONS
# ============================================
def generate_audio_description(labels):
    """Generate natural-sounding audio description."""
    if not labels:
        tts_text = "I couldn't detect any objects. Please try again."
    else:
        label_counts = Counter(labels)
        if len(label_counts) == 1:
            obj, count = list(label_counts.items())[0]
            tts_text = f"I see {count} {obj}" if count > 1 else f"I see one {obj}"
        else:
            items = [f"{count} {obj}" if count > 1 else f"one {obj}" for obj, count in label_counts.items()]
            tts_text = "I see " + ", ".join(items[:-1]) + f", and {items[-1]}." if len(items) > 2 else " and ".join(items)
    
    # Save audio
    timestamp = int(time.time() * 1000)
    audio_file = f"detected_{timestamp}.mp3"
    try:
        tts = gTTS(text=tts_text, lang='en', slow=False)
        tts.save(audio_file)
    except Exception as e:
        print(f"⚠️ TTS error: {e}")
        audio_file = None
    return audio_file

def detect_objects(image, conf_threshold=CONF_THRESHOLD):
    """Detect objects on the image and return annotated image + audio."""
    if image is None or model is None:
        return None, None

    # Convert to numpy array if PIL
    img_np = np.array(image) if isinstance(image, Image.Image) else image
    img_pil = image.copy() if isinstance(image, Image.Image) else Image.fromarray(image)

    try:
        # YOLO detection
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

            # Draw box + label
            draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)
            text = f"{label} {conf:.2f}"
            bbox = draw.textbbox((x1, y1 - 25), text, font=font)
            draw.rectangle(bbox, fill=BOX_COLOR)
            draw.text((x1, y1 - 25), text, fill="white", font=font)

        # Generate audio
        audio_file = generate_audio_description(detected_labels)
        return img_pil, audio_file

    except Exception as e:
        print(f"❌ Detection error: {e}")
        return None, None

# ============================================
# STREAMING AUDIO HANDLER
# ============================================
def transcribe_streaming_audio(audio_tuple):
    """Continuously transcribes audio."""
    if audio_tuple is None or stt_pipe is None:
        return ""
    try:
        sr, audio_data = audio_tuple
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        audio_data = audio_data.astype(np.float32)
        if audio_data.max() > 1.0:
            audio_data = audio_data / 32768.0
        result = stt_pipe({"sampling_rate": sr, "raw": audio_data})
        return result["text"].strip().lower()
    except Exception as e:
        print(f"⚠️ Voice error: {e}")
        return ""

# ============================================
# MAIN CAMERA STREAM PROCESS
# ============================================
def process_camera(latest_frame, transcribed_text, last_detection_image):
    """Check trigger phrases and detect objects if triggered."""
    if latest_frame is None:
        return latest_frame, last_detection_image, None, "Waiting for Camera...", ""

    triggered = any(phrase in transcribed_text for phrase in TRIGGER_PHRASES)

    if triggered:
        print(f"🎤 Trigger detected: '{transcribed_text}'")
        annotated_img, audio_file = detect_objects(latest_frame, CONF_THRESHOLD)
        status_msg = f"✅ Command: '{transcribed_text}' | DETECTION DONE"
        return latest_frame, annotated_img, audio_file, status_msg, ""
    else:
        status_msg = f"🎤 Listening... Last Heard: '{transcribed_text}'" if transcribed_text else "🎤 Listening... Say 'Detect'"
        return latest_frame, last_detection_image, None, status_msg, transcribed_text

# ============================================
# GRADIO INTERFACE
# ============================================
AUTO_START_JS = """
function start_mic_stream() {
    const mic_component = document.getElementById("hidden_voice_input");
    if (mic_component) {
        const start_button = mic_component.querySelector("button");
        if (start_button) start_button.click();
    }
}
setTimeout(start_mic_stream, 500);
"""

with gr.Blocks(title="NoonVision - AI Vision Assistant",
               theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green"),
               js=AUTO_START_JS,
               css="""
#hidden_voice_input {display:none !important; visibility:hidden !important; height:0px !important;}
.status-box {font-weight:bold; font-size:1.1em; color:#764ba2;}
""") as demo:

    gr.HTML("<h1>🦾 NoonVision - AI Vision Assistant</h1><p>Hands-Free Voice-Activated Object Detection</p>")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", sources="webcam", streaming=True, interactive=False, height=500)
        with gr.Column(scale=1):
            image_output = gr.Image(type="pil", label="Detection Results", height=500, value=None)

    status_output = gr.Textbox(value="Waiting for permissions... then listening.", lines=2, elem_classes="status-box")

    voice_input = gr.Audio(sources="microphone", type="numpy", streaming=True, visible=False, elem_id="hidden_voice_input")
    transcribed_state = gr.Textbox(value="", visible=False)
    audio_output = gr.Audio(type="filepath", autoplay=True, visible=False)
    conf_constant = gr.State(value=CONF_THRESHOLD)

    voice_input.stream(fn=transcribe_streaming_audio, inputs=[voice_input], outputs=[transcribed_state], show_progress=False)
    image_input.stream(fn=process_camera, inputs=[image_input, transcribed_state, image_output],
                       outputs=[image_input, image_output, audio_output, status_output, transcribed_state],
                       every=0.1, show_progress=False)

# ============================================
# LAUNCH APP
# ============================================
if __name__ == "__main__":
    demo.launch(share=False, show_error=True)
