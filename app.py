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
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================
# OPTIMIZED CONFIGURATION FOR SPEED
# ============================================
CONF_THRESHOLD = 0.30
IMG_SIZE = 640
BOX_COLOR = (0, 255, 0)
BOX_WIDTH = 4
FONT_SIZE = 18

# ============================================
# FAST MODEL INITIALIZATION
# ============================================
print("🔄 Loading models (optimized for speed)...")

# YOLOv8 with optimization
try:
    model = YOLO("yolov8m.pt")
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    _ = model(dummy, verbose=False, half=True if torch.cuda.is_available() else False)
    print("✅ YOLOv8m loaded successfully (optimized)")
except Exception as e:
    print(f"❌ YOLO model failed to load: {e}")
    model = None

# Whisper - lightweight
try:
    device = 0 if torch.cuda.is_available() else -1
    stt_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        device=device,
        chunk_length_s=5
    )
    print(f"✅ Whisper loaded on {'GPU' if device == 0 else 'CPU'}")
except Exception as e:
    print(f"⚠️ STT model failed: {e}")
    stt_pipe = None

# ============================================
# TRIGGER PHRASES
# ============================================
TRIGGER_PHRASES = [
    "detect", "what do you see", "what's in front of me",
    "identify", "what's this", "scan", "look"
]

# ============================================
# OPTIMIZED DETECTION FUNCTION
# ============================================
def detect_objects_enhanced(image, conf_threshold=CONF_THRESHOLD):
    """OPTIMIZED: Fast object detection with high accuracy."""
    if image is None or model is None:
        return None, None, "⚠️ No image or model"
    
    try:
        start_time = time.time()
        
        # Convert image efficiently
        if isinstance(image, Image.Image):
            img_np = np.array(image)
            img_pil = image.copy()
        else:
            img_np = image
            img_pil = Image.fromarray(image)
        
        # FAST DETECTION
        results = model(
            img_np, 
            imgsz=IMG_SIZE,
            conf=conf_threshold,
            verbose=False,
            half=True if torch.cuda.is_available() else False,
            device=0 if torch.cuda.is_available() else 'cpu'
        )[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        labels = results.names
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        
        # Drawing
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", FONT_SIZE)
        except:
            font = ImageFont.load_default()
        
        detected_labels = []
        
        # Draw detections
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(class_ids[i])
            label = labels[cls_id]
            conf = confidences[i]
            
            detected_labels.append(label)
            
            draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)
            text = f"{label} {conf:.2f}"
            bbox = draw.textbbox((x1, y1 - 25), text, font=font)
            draw.rectangle(bbox, fill=BOX_COLOR)
            draw.text((x1, y1 - 25), text, fill="black", font=font)
        
        # Generate audio
        audio_file = generate_audio_description(detected_labels)
        
        elapsed = time.time() - start_time
        status = f"✅ Detected {len(detected_labels)} objects in {elapsed:.2f}s"
        
        print(f"⚡ Detection: {len(detected_labels)} objects in {elapsed:.2f}s")
        return img_pil, audio_file, status
        
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return None, None, f"❌ Error: {str(e)}"

# ============================================
# AUDIO GENERATION
# ============================================
def generate_audio_description(labels):
    """Generate natural audio description."""
    try:
        if not labels:
            tts_text = "No objects detected. Try better lighting."
        else:
            label_counts = Counter(labels)
            
            if len(label_counts) == 1:
                obj, count = list(label_counts.items())[0]
                tts_text = f"I see {count} {obj}." if count == 1 else f"I see {count} {obj}s."
            else:
                items = []
                for obj, count in label_counts.items():
                    items.append(f"{count} {obj}" if count > 1 else f"one {obj}")
                
                if len(items) == 2:
                    tts_text = f"I see {items[0]} and {items[1]}."
                else:
                    tts_text = f"I see {', '.join(items[:-1])}, and {items[-1]}."
        
        filename = f"detected_{int(time.time()*1000)}.mp3"
        tts = gTTS(text=tts_text, lang='en', slow=False)
        tts.save(filename)
        return filename
        
    except Exception as e:
        print(f"⚠️ Audio error: {e}")
        return None

# ============================================
# STREAMING HANDLERS
# ============================================
def transcribe_streaming_audio(audio_tuple):
    """Fast audio transcription."""
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
        transcribed_text = result["text"].strip().lower()
        
        if len(transcribed_text) > 1:
            return transcribed_text
        
        return ""
            
    except Exception as e:
        return ""

def process_camera_stream(latest_frame, transcribed_text, last_detection_image):
    """Main processing loop."""
    
    if latest_frame is None:
        return latest_frame, last_detection_image, None, "⏳ Waiting for camera...", ""
    
    is_triggered = any(phrase in transcribed_text for phrase in TRIGGER_PHRASES)
    
    if is_triggered:
        print(f"🎤 Trigger: '{transcribed_text}'")
        annotated_img, audio_file, status_msg = detect_objects_enhanced(latest_frame, CONF_THRESHOLD)
        return latest_frame, annotated_img, audio_file, status_msg, ""
    
    status_msg = f"🎤 Listening... (say 'detect')" if not transcribed_text else f"🎤 Heard: '{transcribed_text}'"
    
    return latest_frame, last_detection_image, None, status_msg, transcribed_text

# ============================================
# GRADIO INTERFACE (FIXED)
# ============================================

AUTO_START_JS = """
function start_mic_stream() {
    setTimeout(() => {
        const mic = document.getElementById("hidden_voice_input");
        if (mic) {
            const btn = mic.querySelector("button");
            if (btn) {
                btn.click();
                console.log("✅ Microphone auto-started");
            }
        }
    }, 1000);
}
start_mic_stream();
"""

with gr.Blocks(
    title="NoonVision - AI Vision Assistant",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green"),
    css="""
        .main-header {
            text-align: center; 
            padding: 25px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            border-radius: 12px; 
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .instruction-box {
            background: linear-gradient(to right, #f0f8ff, #e6f3ff); 
            padding: 20px; 
            border-radius: 10px; 
            border-left: 5px solid #667eea;
            margin-bottom: 20px;
        }
        #hidden_voice_input {
            display: none !important;
        }
        .status-box {
            font-weight: bold;
            font-size: 1.2em;
            color: #764ba2;
            text-align: center;
        }
    """,
    js=AUTO_START_JS
) as demo:
    
    gr.HTML("""
        <div class="main-header">
            <h1>🦾 NoonVision</h1>
            <h2>AI Vision Assistant for the Visually Impaired</h2>
            <p style="font-size: 1.1em; margin-top: 10px;">✨ Hands-Free | ⚡ Real-Time | ♿ Accessible</p>
        </div>
    """)
    
    gr.Markdown("""
    <div class="instruction-box">
    <h3>🎤 Voice-Activated Mode (Auto-Start)</h3>
    
    <h4>📢 Quick Start:</h4>
    <ol>
        <li><strong>Allow permissions</strong> when browser asks (microphone + camera)</li>
        <li><strong>Just say:</strong> "Detect" or "What do you see?"</li>
        <li><strong>Listen</strong> to the audio description (auto-plays)</li>
        <li><strong>Repeat</strong> anytime - completely hands-free!</li>
    </ol>
    
    <p><strong>💡 Pro Tips:</strong> Good lighting • Objects 2-6 feet away • Speak clearly</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                sources=["webcam"],
                label="📷 Live Camera Feed",
                streaming=True,
                interactive=False,
                height=450
            )
        
        with gr.Column(scale=1):
            image_output = gr.Image(
                type="pil",
                label="🎯 Detection Results",
                height=450
            )
    
    gr.Markdown("### 📊 System Status")
    status_output = gr.Textbox(
        label="",
        value="🚀 Ready! Grant permissions to start...",
        lines=2,
        elem_classes="status-box"
    )
    
    # Hidden components (FIX: Using list instead of string for sources)
    voice_input = gr.Audio(
        sources=["microphone"],
        type="numpy",
        streaming=True,
        visible=False,
        elem_id="hidden_voice_input"
    )
    
    transcribed_state = gr.Textbox(value="", visible=False)
    
    audio_output = gr.Audio(
        type="filepath",
        autoplay=True,
        visible=False
    )
    
    # Event handlers
    voice_input.stream(
        fn=transcribe_streaming_audio,
        inputs=[voice_input],
        outputs=[transcribed_state],
        show_progress=False
    )
    
    image_input.stream(
        fn=process_camera_stream,
        inputs=[image_input, transcribed_state, image_output],
        outputs=[image_input, image_output, audio_output, status_output, transcribed_state],
        time_limit=None,
        stream_every=0.15,
        show_progress=False
    )
    
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>⚡ Performance:</strong> 1-2 second response time | <strong>🎯 Accuracy:</strong> 80+ objects detected</p>
        <p><strong>🛠️ Tech Stack:</strong> YOLOv8m + Whisper + gTTS + Gradio</p>
        <p>Made with ❤️ for accessibility | Open Source | MIT License</p>
    </div>
    """)

# ============================================
# LAUNCH (FIXED FOR HUGGING FACE)
# ============================================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # CRITICAL: Required for Hugging Face
        show_error=True,
        show_api=False
    )
