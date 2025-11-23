import gradio as gr
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import time
from transformers import pipeline
import torch
from collections import Counter

# ============================================
# CONFIGURATION
# ============================================
CONF_THRESHOLD = 0.30
IMG_SIZE = 640
BOX_COLOR = (0, 255, 0)
BOX_WIDTH = 4
FONT_SIZE = 18

# ============================================
# MODEL INITIALIZATION
# ============================================
print("🔄 Loading models...")

# YOLOv8
try:
    model = YOLO("yolov8m.pt")
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    _ = model(dummy, verbose=False)
    print("✅ YOLOv8m loaded")
except Exception as e:
    print(f"❌ YOLO failed: {e}")
    model = None

# Whisper
try:
    device = 0 if torch.cuda.is_available() else -1
    stt_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        device=device
    )
    print(f"✅ Whisper loaded on {'GPU' if device == 0 else 'CPU'}")
except Exception as e:
    print(f"⚠️ STT failed: {e}")
    stt_pipe = None

# ============================================
# TRIGGER PHRASES
# ============================================
TRIGGER_PHRASES = ["detect", "what do you see", "what's in front of me", "identify", "scan", "look"]

# ============================================
# DETECTION FUNCTION
# ============================================
def detect_objects(image):
    """Detect objects and return annotated image + audio."""
    if image is None or model is None:
        return None, None, "⚠️ No image or model"
    
    try:
        start = time.time()
        
        # Convert image
        if isinstance(image, Image.Image):
            img_np = np.array(image)
            img_pil = image.copy()
        else:
            img_np = image
            img_pil = Image.fromarray(image)
        
        # Run detection
        results = model(img_np, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        labels = results.names
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        
        # Draw
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
            bbox = draw.textbbox((x1, y1 - 25), text, font=font)
            draw.rectangle(bbox, fill=BOX_COLOR)
            draw.text((x1, y1 - 25), text, fill="black", font=font)
        
        # Generate audio
        audio = generate_audio(detected_labels)
        
        elapsed = time.time() - start
        status = f"✅ Found {len(detected_labels)} objects in {elapsed:.2f}s"
        
        return img_pil, audio, status
        
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"

# ============================================
# AUDIO GENERATION
# ============================================
def generate_audio(labels):
    """Generate TTS audio."""
    try:
        if not labels:
            text = "No objects detected"
        else:
            counts = Counter(labels)
            if len(counts) == 1:
                obj, count = list(counts.items())[0]
                text = f"I see {count} {obj}" if count == 1 else f"I see {count} {obj}s"
            else:
                items = [f"{count} {obj}" if count > 1 else f"one {obj}" for obj, count in counts.items()]
                text = f"I see {', '.join(items[:-1])} and {items[-1]}"
        
        filename = f"detected_{int(time.time()*1000)}.mp3"
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filename)
        return filename
    except Exception as e:
        print(f"⚠️ Audio error: {e}")
        return None

# ============================================
# VOICE COMMAND HANDLER
# ============================================
def process_voice(audio_file, image):
    """Process voice command."""
    if audio_file is None or stt_pipe is None or image is None:
        return None, None, "⚠️ No audio/image/STT"
    
    try:
        result = stt_pipe(audio_file)
        text = result["text"].strip().lower()
        
        if any(phrase in text for phrase in TRIGGER_PHRASES):
            return detect_objects(image)
        else:
            return None, None, f"❓ Heard: '{text}' - Say 'detect'"
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"

# ============================================
# GRADIO INTERFACE
# ============================================
with gr.Blocks(
    title="NoonVision",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green"),
    css="""
        .header {text-align: center; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 12px; margin-bottom: 20px;}
        .info {background: linear-gradient(to right, #f0f8ff, #e6f3ff); padding: 20px; border-radius: 10px; border-left: 5px solid #667eea; margin-bottom: 20px;}
    """
) as demo:
    
    # Header
    gr.HTML('<div class="header"><h1>🦾 NoonVision</h1><h2>AI Vision Assistant</h2><p>✨ Hands-Free | ⚡ Real-Time | ♿ Accessible</p></div>')
    
    # Instructions
    gr.Markdown("""
    <div class="info">
    <h3>🎤 Voice-Activated Detection</h3>
    <ol>
        <li><strong>Allow</strong> camera and microphone permissions</li>
        <li><strong>Record voice:</strong> Say "Detect" or "What do you see?"</li>
        <li><strong>OR click:</strong> "Detect Now" button for instant detection</li>
        <li><strong>Listen</strong> to audio results (auto-plays)</li>
    </ol>
    <p><strong>💡 Tips:</strong> Good lighting • Objects 2-6 feet away • Speak clearly</p>
    </div>
    """)
    
    # Main interface
    with gr.Row():
        with gr.Column(scale=1):
            webcam = gr.Image(sources="webcam", type="pil", label="📷 Camera", height=400)
            voice = gr.Audio(sources="microphone", type="filepath", label="🎤 Voice (Say 'Detect')")
            
            with gr.Row():
                detect_btn = gr.Button("🔍 Detect Now", variant="primary", size="lg")
                voice_btn = gr.Button("🎙️ Use Voice Command", variant="secondary", size="lg")
        
        with gr.Column(scale=1):
            result_img = gr.Image(type="pil", label="🎯 Results", height=400)
            status = gr.Textbox(label="📊 Status", lines=2, value="Ready! Click button or use voice")
            audio_out = gr.Audio(type="filepath", label="🔊 Audio", autoplay=True)
    
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>⚡ Performance:</strong> 1-2 second response | <strong>🎯 Accuracy:</strong> 80+ objects</p>
        <p>YOLOv8m + Whisper + gTTS + Gradio | MIT License | Made with ❤️ for accessibility</p>
    </div>
    """)
    
    # Event handlers
    detect_btn.click(fn=detect_objects, inputs=webcam, outputs=[result_img, audio_out, status])
    voice_btn.click(fn=process_voice, inputs=[voice, webcam], outputs=[result_img, audio_out, status])

# ============================================
# LAUNCH
# ============================================
if __name__ == "__main__":
    demo.launch()