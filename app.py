import gradio as gr
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import time
from transformers import pipeline
from collections import Counter
import torch
import os
import tempfile

# ======================
# CONFIGURATION
# ======================
CONF_THRESHOLD = 0.25
IMG_SIZE = 640
BOX_COLOR = (255, 50, 50)
BOX_WIDTH = 3
FONT_SIZE = 18

# ======================
# MODEL INITIALIZATION
# ======================
print("🔄 Loading YOLOv8m (CPU)...")
try:
    model = YOLO("yolov8m.pt")
    # Warmup run
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    _ = model(dummy, verbose=False)
    print("✅ YOLOv8m loaded successfully on CPU")
except Exception as e:
    print(f"❌ YOLOv8m failed to load: {e}")
    model = None

# Whisper STT (CPU)
print("🔄 Loading Whisper STT (CPU)...")
try:
    stt_pipe = pipeline(
        "automatic-speech-recognition", 
        model="openai/whisper-tiny.en", 
        device=-1  # CPU
    )
    print("✅ Whisper STT loaded on CPU")
except Exception as e:
    print(f"❌ Whisper failed: {e}")
    stt_pipe = None

# ======================
# TRIGGER PHRASES
# ======================
TRIGGER_PHRASES = [
    "detect", "what do you see", "what's in front of me",
    "what is in front of me", "identify objects", "what's this",
    "what is this", "tell me what you see", "scan", "look"
]

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
                if count == 1:
                    items.append(f"a {obj}")
                else:
                    items.append(f"{count} {obj}s")
            
            if len(items) == 1:
                tts_text = f"I see {items[0]}."
            else:
                tts_text = "I see " + ", ".join(items[:-1]) + f" and {items[-1]}."

        # Create temporary file
        timestamp = int(time.time() * 1000)
        audio_file = f"/tmp/detected_{timestamp}.mp3"
        tts = gTTS(text=tts_text, lang='en', slow=False)
        tts.save(audio_file)
        return audio_file
    except Exception as e:
        print(f"⚠️ Audio generation error: {e}")
        return None

# ======================
# OBJECT DETECTION
# ======================
def detect_objects(image):
    if image is None or model is None:
        return None, "No image provided or model not loaded"

    try:
        img_np = np.array(image)
        img_pil = image.copy()
        results = model(img_np, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]

        # Check if any detections
        if results.boxes is None or len(results.boxes) == 0:
            audio_file = generate_audio_description([])
            return image, "No objects detected", audio_file

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
            
            if conf >= CONF_THRESHOLD:
                detected_labels.append(label)
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)
                # Draw label
                text = f"{label} {conf:.2f}"
                bbox = draw.textbbox((x1, y1 - 20), text, font=font)
                draw.rectangle(bbox, fill=BOX_COLOR)
                draw.text((x1, y1 - 20), text, fill="white", font=font)

        audio_file = generate_audio_description(detected_labels)
        object_count = len(detected_labels)
        status_msg = f"Detected {object_count} object(s): {', '.join(detected_labels)}"
        
        return img_pil, status_msg, audio_file
        
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return None, f"Detection error: {str(e)}", None

# ======================
# SPEECH RECOGNITION
# ======================
def transcribe_audio(audio_data):
    if audio_data is None or stt_pipe is None:
        return ""
    
    try:
        sample_rate, audio_array = audio_data
        
        # Handle stereo audio by converting to mono
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
        # Normalize audio data
        audio_array = audio_array.astype(np.float32)
        if audio_array.max() > 1.0:
            audio_array = audio_array / 32768.0
        
        # Transcribe
        result = stt_pipe({"sampling_rate": sample_rate, "raw": audio_array})
        text = result["text"].strip().lower()
        print(f"🎤 Transcribed: {text}")
        return text
        
    except Exception as e:
        print(f"⚠️ Voice processing error: {e}")
        return ""

# ======================
# MAIN PROCESSING FUNCTION
# ======================
def process_input(audio_data, image):
    """
    Main function that processes both audio and image inputs
    """
    if image is None:
        return None, "Please enable your camera first", None, ""
    
    transcribed_text = ""
    if audio_data is not None:
        transcribed_text = transcribe_audio(audio_data)
    
    # Check if trigger phrase was spoken
    triggered = any(phrase in transcribed_text for phrase in TRIGGER_PHRASES)
    
    if triggered:
        print(f"🚨 Trigger detected: {transcribed_text}")
        annotated_img, status_msg, audio_file = detect_objects(image)
        if annotated_img is not None:
            return annotated_img, status_msg, audio_file, transcribed_text
        else:
            return image, "Detection failed. Please try again.", None, transcribed_text
    else:
        if transcribed_text:
            status_msg = f"Heard: '{transcribed_text}'. Say 'detect' or 'what do you see?' to identify objects."
        else:
            status_msg = "🎤 Speak a command like 'detect' or 'what do you see?'"
        
        return image, status_msg, None, transcribed_text

# ======================
# GRADIO INTERFACE
# ======================
with gr.Blocks(theme=gr.themes.Soft(), title="NoonVision - AI Vision Assistant") as demo:
    gr.HTML("""
    <div style="text-align: center;">
        <h1>🦾 NoonVision - Hands-Free AI Vision Assistant</h1>
        <p>Speak commands like "detect" or "what do you see" to identify objects around you</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column():
            webcam = gr.Image(
                sources="webcam", 
                label="Live Camera", 
                streaming=False,
                height=400
            )
            
        with gr.Column():
            output_image = gr.Image(
                label="Detected Objects", 
                height=400
            )
    
    with gr.Row():
        microphone = gr.Audio(
            sources="microphone", 
            type="numpy",
            label="Speak Commands",
            show_download_button=False
        )
    
    with gr.Row():
        status_display = gr.Textbox(
            label="Status",
            value="🎤 Allow microphone permissions and speak a trigger phrase...",
            lines=2,
            interactive=False
        )
    
    audio_output = gr.Audio(
        label="Audio Description",
        autoplay=True,
        visible=False
    )
    
    transcription_display = gr.Textbox(
        label="What I heard",
        interactive=False,
        visible=True
    )
    
    # Instructions
    with gr.Accordion("📖 How to use:", open=False):
        gr.Markdown("""
        1. **Allow camera and microphone permissions** when prompted
        2. **Speak clearly** one of these trigger phrases:
           - "Detect"
           - "What do you see?"
           - "What's in front of me?"
           - "Identify objects"
           - "Scan"
           - "Look"
        3. **Wait for detection** - objects will be highlighted
        4. **Listen** to the audio description of what was detected
        
        **Note:** This runs entirely on CPU, so detection may take 2-3 seconds.
        """)
    
    # Trigger phrases display
    with gr.Accordion("🎯 Available Commands", open=False):
        gr.Markdown("\n".join([f"- **'{phrase}'**" for phrase in TRIGGER_PHRASES]))
    
    # Process when audio is provided
    microphone.stop_recording(
        fn=process_input,
        inputs=[microphone, webcam],
        outputs=[output_image, status_display, audio_output, transcription_display]
    )
    
    # Also process when image updates (for manual trigger)
    webcam.change(
        fn=lambda img: (img, "Camera active. Speak a command!", None, ""),
        inputs=[webcam],
        outputs=[output_image, status_display, audio_output, transcription_display]
    )

# ======================
# LAUNCH APPLICATION
# ======================
if __name__ == "__main__":
    # Clean up any old audio files
    for file in os.listdir("/tmp"):
        if file.startswith("detected_") and file.endswith(".mp3"):
            try:
                os.remove(os.path.join("/tmp", file))
            except:
                pass
    
    # Launch with simplified configuration
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    )