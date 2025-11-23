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
    # Download model if not exists
    if not os.path.exists("yolov8m.pt"):
        print("📥 Downloading YOLOv8m model...")
        YOLO("yolov8m.pt")
    
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
device = -1  # CPU
try:
    stt_pipe = pipeline(
        "automatic-speech-recognition", 
        model="openai/whisper-tiny.en", 
        device=device
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

        # Check if any detections
        if results.boxes is None or len(results.boxes) == 0:
            audio_file = generate_audio_description([])
            return image, audio_file

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
            
            if conf >= conf_threshold:
                detected_labels.append(label)
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)
                # Draw label
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
# SPEECH RECOGNITION
# ======================
def transcribe_streaming_audio(audio_tuple):
    if audio_tuple is None or stt_pipe is None:
        return ""
    try:
        sample_rate, audio_data = audio_tuple
        
        # Handle stereo audio by converting to mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Normalize audio data
        audio_data = audio_data.astype(np.float32)
        if audio_data.max() > 1.0:
            audio_data = audio_data / 32768.0
        
        # Transcribe
        result = stt_pipe({"sampling_rate": sample_rate, "raw": audio_data})
        text = result["text"].strip().lower()
        print(f"🎤 Transcribed: {text}")
        return text
        
    except Exception as e:
        print(f"⚠️ Voice processing error: {e}")
        return ""

# ======================
# MAIN PROCESSING
# ======================
def process_camera(latest_frame, transcribed_text, last_image):
    if latest_frame is None:
        return latest_frame, last_image, None, "🔄 Waiting for camera...", ""
    
    current_text = transcribed_text or ""
    triggered = any(phrase in current_text for phrase in TRIGGER_PHRASES)
    
    if triggered:
        print(f"🚨 Trigger detected: {current_text}")
        annotated_img, audio_file = detect_objects(latest_frame, CONF_THRESHOLD)
        if annotated_img is not None:
            return latest_frame, annotated_img, audio_file, "✅ Objects detected! Listen to the audio description.", ""
        else:
            return latest_frame, last_image, None, "❌ Detection failed. Please try again.", ""
    
    return latest_frame, last_image, None, "🎤 Listening... Say 'detect' or 'what do you see?'", ""

# ======================
# GRADIO INTERFACE
# ======================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div style="text-align: center;">
        <h1>🦾 NoonVision - Hands-Free AI Vision Assistant</h1>
        <p>Speak commands like "detect" or "what do you see" to identify objects around you</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="pil", 
                sources="webcam", 
                streaming=True, 
                interactive=False, 
                height=400,
                label="Live Camera Feed"
            )
            gr.HTML("<p style='text-align: center; color: #666;'>Camera feed - allow camera permissions</p>")
        
        with gr.Column():
            image_output = gr.Image(
                type="pil", 
                height=400,
                label="Detected Objects"
            )
            gr.HTML("<p style='text-align: center; color: #666;'>Objects will appear here when detected</p>")
    
    status_output = gr.Textbox(
        label="Status",
        value="🎤 Allow microphone permissions and speak a trigger phrase...",
        lines=2,
        max_lines=2
    )
    
    # Hidden components for state management
    voice_input = gr.Audio(
        sources="microphone", 
        type="numpy", 
        streaming=True, 
        visible=False
    )
    transcribed_state = gr.Textbox(visible=False)
    audio_output = gr.Audio(
        type="filepath", 
        autoplay=True, 
        visible=False,
        label="Audio Description"
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
        gr.Markdown("\n".join([f"- '{phrase}'" for phrase in TRIGGER_PHRASES]))
    
    # ======================
    # EVENT HANDLERS
    # ======================
    
    # Voice transcription
    voice_input.stream(
        fn=transcribe_streaming_audio,
        inputs=[voice_input],
        outputs=[transcribed_state],
        show_progress="hidden"
    )
    
    # Camera processing
    image_input.stream(
        fn=process_camera,
        inputs=[image_input, transcribed_state, image_output],
        outputs=[image_input, image_output, audio_output, status_output, transcribed_state],
        every=0.5,  # Process every 500ms to reduce CPU load
        show_progress="hidden"
    )

# ======================
# LAUNCH APPLICATION
# ======================
if __name__ == "__main__":
    # Clean up any old audio files
    for file in os.listdir("."):
        if file.startswith("detected_") and file.endswith(".mp3"):
            try:
                os.remove(file)
            except:
                pass
    
    # Launch with Hugging Face Spaces compatibility
    if os.getenv('SPACE_ID'):
        print("🚀 Launching on Hugging Face Spaces...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True
        )
    else:
        print("🚀 Launching locally...")
        demo.launch(
            server_name="0.0.0.0", 
            server_port=7860,
            share=True,  # Create public link
            debug=False,
            show_error=True
        )