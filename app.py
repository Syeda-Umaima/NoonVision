import gradio as gr
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import os
import time
from transformers import pipeline
import torch

# ============================================
# CONFIGURATION
# ============================================
CONF_THRESHOLD = 0.25
IMG_SIZE = 640
BOX_COLOR = (0, 255, 0)
BOX_WIDTH = 3

print("🔄 Loading models...")

# YOLOv8
try:
    model = YOLO("yolov8m.pt")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
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
    print(f"✅ Whisper loaded")
except Exception as e:
    print(f"⚠️ STT failed: {e}")
    stt_pipe = None

print("✅ Ready!")

# ============================================
# DETECTION
# ============================================
def detect_objects(image):
    if image is None or model is None:
        return None, None
    
    try:
        if isinstance(image, np.ndarray):
            img_pil = Image.fromarray(image)
        else:
            img_pil = image.copy()
        
        img_np = np.array(img_pil)
        results = model(img_np, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        labels = results.names
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.load_default()
        
        detected = []
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = labels[int(class_ids[i])]
            conf = confidences[i]
            detected.append(label)
            
            draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)
            draw.text((x1, y1 - 15), f"{label} {conf:.2f}", fill=BOX_COLOR)
        
        if detected:
            from collections import Counter
            counts = Counter(detected)
            
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
        
        audio_file = f"audio_{int(time.time()*1000)}.mp3"
        tts = gTTS(text=speech, lang='en', slow=False)
        tts.save(audio_file)
        
        print(f"✅ {speech}")
        return img_pil, audio_file
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

# ============================================
# VOICE TRIGGER
# ============================================
def check_voice(audio):
    if audio is None or stt_pipe is None:
        return False, ""
    
    try:
        result = stt_pipe(audio)
        text = result["text"].strip().lower()
        triggers = ["detect", "what do you see", "identify", "scan", "look"]
        return any(t in text for t in triggers), text
    except:
        return False, ""

# ============================================
# INTERFACE
# ============================================
with gr.Blocks(title="NoonVision") as demo:
    
    gr.HTML("""
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px;">
            <h1>🦾 NoonVision - Hands-Free AI Vision</h1>
            <h2>🎤 Say "DETECT" to identify objects!</h2>
        </div>
    """)
    
    with gr.Row():
        with gr.Column():
            camera = gr.Image(
                sources="webcam",
                type="pil",
                label="📷 Camera"
            )
            
            detect_btn = gr.Button(
                "🔍 DETECT NOW",
                variant="primary",
                size="lg"
            )
            
            gr.Markdown("### 🎤 Voice Command")
            voice = gr.Audio(
                sources="microphone",
                type="filepath",
                label="Say 'Detect'"
            )
        
        with gr.Column():
            result = gr.Image(label="🎯 Results")
            audio_out = gr.Audio(
                type="filepath",
                autoplay=True,
                label="🔊 Audio"
            )
            status = gr.Textbox(
                label="Status",
                value="Ready! Click DETECT or say 'Detect'",
                lines=2
            )
    
    gr.Markdown("""
    ---
    ## 📋 Instructions:
    1. **Allow camera/mic** when prompted
    2. **Click GREEN button** OR **say "Detect"**
    3. **Listen** to results
    
    💡 Good lighting | 2-6 feet away | Objects in frame
    """)
    
    # Button click
    detect_btn.click(
        fn=detect_objects,
        inputs=[camera],
        outputs=[result, audio_out]
    )
    
    # Voice trigger
    def handle_voice(audio, img):
        triggered, text = check_voice(audio)
        if triggered:
            res_img, res_audio = detect_objects(img)
            return res_img, res_audio, f"✅ '{text}' - Detecting..."
        return None, None, f"❌ '{text}' - Say 'Detect'"
    
    voice.change(
        fn=handle_voice,
        inputs=[voice, camera],
        outputs=[result, audio_out, status]
    )

demo.launch()