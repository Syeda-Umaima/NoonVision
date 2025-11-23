import gradio as gr
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import os
import time
from transformers import pipeline
import torch
import threading

# ============================================
# CONFIGURATION
# ============================================
CONF_THRESHOLD = 0.25
IMG_SIZE = 640  # Reduced for speed
BOX_COLOR = (0, 255, 0)
BOX_WIDTH = 3
FONT_SIZE = 16
AUTO_DETECT_INTERVAL = 5  # Detect every 5 seconds

# ============================================
# MODEL INITIALIZATION
# ============================================
print("🔄 Loading models...")

try:
    model = YOLO("yolov8m.pt")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = model(dummy, verbose=False)
    print("✅ YOLOv8m loaded")
except Exception as e:
    print(f"❌ YOLO failed: {e}")
    model = None

try:
    device = 0 if torch.cuda.is_available() else -1
    stt_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        device=device
    )
    print(f"✅ Whisper loaded")
except Exception as e:
    print(f"⚠️ STT failed: {e}")
    stt_pipe = None

print("✅ Ready!")

# ============================================
# DETECTION FUNCTION
# ============================================
def detect_objects(image):
    """Detect objects and return annotated image + audio"""
    
    if image is None or model is None:
        return None, None
    
    try:
        if isinstance(image, np.ndarray):
            img_pil = Image.fromarray(image)
        else:
            img_pil = image.copy()
        
        img_np = np.array(img_pil)
        
        # Detect
        results = model(img_np, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        labels = results.names
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        detected_labels = []
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = labels[int(class_ids[i])]
            conf = confidences[i]
            
            detected_labels.append(label)
            
            draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)
            text = f"{label} {conf:.2f}"
            draw.text((x1, y1 - 15), text, fill=BOX_COLOR)
        
        # Generate audio
        if detected_labels:
            from collections import Counter
            counts = Counter(detected_labels)
            
            if len(counts) == 1:
                obj, cnt = list(counts.items())[0]
                speech = f"I see {cnt} {obj}{'s' if cnt > 1 else ''}."
            else:
                items = [f"{cnt} {obj}{'s' if cnt > 1 else ''}" for obj, cnt in counts.items()]
                if len(items) == 2:
                    speech = f"I see {items[0]} and {items[1]}."
                else:
                    speech = f"I see {', '.join(items[:-1])}, and {items[-1]}."
        else:
            speech = "No objects detected."
        
        # Save audio
        audio_file = f"audio_{int(time.time()*1000)}.mp3"
        tts = gTTS(text=speech, lang='en', slow=False)
        tts.save(audio_file)
        
        print(f"✅ Detected: {speech}")
        return img_pil, audio_file
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

# ============================================
# VOICE TRIGGER FUNCTION
# ============================================
def check_voice_trigger(audio):
    """Check if voice says 'detect'"""
    
    if audio is None or stt_pipe is None:
        return False, ""
    
    try:
        result = stt_pipe(audio)
        text = result["text"].strip().lower()
        
        triggers = ["detect", "what do you see", "identify", "scan", "look"]
        triggered = any(t in text for t in triggers)
        
        return triggered, text
    except:
        return False, ""

# ============================================
# INTERFACE WITH AUTO-DETECT
# ============================================
def create_interface():
    with gr.Blocks(title="NoonVision", theme=gr.themes.Soft()) as demo:
        
        gr.HTML("""
            <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px;">
                <h1>🦾 NoonVision - Hands-Free AI Vision</h1>
                <h2>🎤 Say "DETECT" or wait - Auto-detects every 5 seconds!</h2>
            </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                camera = gr.Image(
                    sources="webcam",
                    type="pil",
                    label="📷 Live Camera",
                    streaming=True
                )
                
                gr.Markdown("""
                ### 🎤 Voice Commands:
                Say **"Detect"**, **"What do you see?"**, or **"Identify objects"**
                
                ### ⏰ Auto-Detection:
                App automatically detects every **5 seconds** - just wait!
                """)
                
                voice = gr.Audio(
                    sources="microphone",
                    type="filepath",
                    label="🎙️ Voice Input (Optional)",
                    streaming=False
                )
            
            with gr.Column(scale=1):
                result = gr.Image(
                    type="pil",
                    label="🎯 Detected Objects"
                )
                
                audio_output = gr.Audio(
                    type="filepath",
                    label="🔊 Audio Results",
                    autoplay=True
                )
                
                status = gr.Textbox(
                    label="📊 Status",
                    value="🎤 Ready! Say 'Detect' or wait for auto-detection...",
                    lines=2
                )
        
        gr.Markdown("""
        ---
        ## 🎯 How It Works:
        
        **Option 1 - Voice Control (Best for accessibility):**
        - Say **"Detect"** anytime
        - App captures and analyzes immediately
        - Results spoken aloud
        
        **Option 2 - Automatic:**
        - Every 5 seconds, app auto-detects
        - No interaction needed
        - Completely hands-free
        
        ---
        💡 **Tips:** Good lighting | 2-6 feet from camera | Objects in frame
        
        🔊 **Audio:** Turn up volume to hear results clearly
        """)
        
        # Voice trigger detection
        def handle_voice(audio, image):
            triggered, text = check_voice_trigger(audio)
            
            if triggered:
                img, aud = detect_objects(image)
                return img, aud, f"✅ Command: '{text}' - Detecting..."
            else:
                return None, None, f"❓ Heard: '{text}' - Say 'Detect' to trigger"
        
        voice.change(
            fn=handle_voice,
            inputs=[voice, camera],
            outputs=[result, audio_output, status]
        )
        
        # Auto-detect timer (triggered by camera changes)
        last_detect_time = [0]
        
        def auto_detect_check(image):
            current_time = time.time()
            if current_time - last_detect_time[0] >= AUTO_DETECT_INTERVAL:
                last_detect_time[0] = current_time
                img, aud = detect_objects(image)
                return img, aud, f"🔄 Auto-detected at {time.strftime('%H:%M:%S')}"
            return None, None, "⏳ Waiting..."
        
        camera.change(
            fn=auto_detect_check,
            inputs=[camera],
            outputs=[result, audio_output, status],
            show_progress=False
        )
    
    return demo

# ============================================
# LAUNCH
# ============================================
if __name__ == "__main__":
    app = create_interface()
    app.launch(share=False)