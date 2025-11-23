import gradio as gr
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import time
from transformers import pipeline
import torch
from collections import Counter

# Configuration
CONF_THRESHOLD = 0.30
IMG_SIZE = 640
BOX_COLOR = (0, 255, 0)
BOX_WIDTH = 4
FONT_SIZE = 18

print("Loading models...")

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
    stt_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device=device)
    print(f"✅ Whisper loaded on {'GPU' if device == 0 else 'CPU'}")
except Exception as e:
    print(f"⚠️ STT failed: {e}")
    stt_pipe = None

TRIGGER_PHRASES = ["detect", "what do you see", "what's in front", "identify", "scan", "look"]

def detect_objects(image):
    if image is None or model is None:
        return None, None, "⚠️ No image or model"
    
    try:
        start = time.time()
        
        if isinstance(image, Image.Image):
            img_np = np.array(image)
            img_pil = image.copy()
        else:
            img_np = image
            img_pil = Image.fromarray(image)
        
        results = model(img_np, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        
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
            bbox = draw.textbbox((x1, y1 - 25), text, font=font)
            draw.rectangle(bbox, fill=BOX_COLOR)
            draw.text((x1, y1 - 25), text, fill="black", font=font)
        
        audio = generate_audio(detected_labels)
        
        elapsed = time.time() - start
        status = f"✅ Found {len(detected_labels)} objects in {elapsed:.2f}s"
        
        return img_pil, audio, status
        
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"

def generate_audio(labels):
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

def process_voice(audio_file, image):
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

def clear_all():
    """Clear all outputs and reset to initial state"""
    return None, None, None, "🔄 Cleared! Ready for next detection", None

with gr.Blocks(title="NoonVision", theme=gr.themes.Soft()) as demo:
    
    gr.HTML('<div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 12px; margin-bottom: 20px;"><h1>🦾 NoonVision</h1><h2>AI Vision Assistant</h2><p>✨ Voice-Activated Object Detection</p></div>')
    
    gr.Markdown("""
    ### 🎤 How to Use:
    1. **Allow** camera and microphone permissions
    2. **Record voice** and say "Detect" or "What do you see?"
    3. **OR click** "Detect Now" for instant detection
    4. **Click Clear** to reset and capture new image
    
    💡 **Tips:** Good lighting • Objects 2-6 feet away • Speak clearly
    """)
    
    with gr.Row():
        with gr.Column():
            webcam = gr.Image(sources=["webcam"], type="pil", label="📷 Camera")
            voice = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Voice")
            
            with gr.Row():
                detect_btn = gr.Button("🔍 Detect Now", variant="primary", size="lg")
                voice_btn = gr.Button("🎙️ Use Voice", variant="secondary", size="lg")
                clear_btn = gr.Button("🗑️ Clear", variant="stop", size="lg")
        
        with gr.Column():
            result_img = gr.Image(type="pil", label="🎯 Results")
            status = gr.Textbox(label="📊 Status", value="Ready! Click 'Detect Now' or use voice", lines=2)
            audio_out = gr.Audio(type="filepath", label="🔊 Audio", autoplay=True)
    
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>⚡ Performance:</strong> 1-2 second response | <strong>🎯 Accuracy:</strong> 80+ objects detected</p>
        <p>YOLOv8m + Whisper + gTTS | Made with ❤️ for accessibility</p>
    </div>
    """)
    
    # Event handlers
    detect_btn.click(
        fn=detect_objects, 
        inputs=webcam, 
        outputs=[result_img, audio_out, status]
    )
    
    voice_btn.click(
        fn=process_voice, 
        inputs=[voice, webcam], 
        outputs=[result_img, audio_out, status]
    )
    
    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[webcam, voice, result_img, status, audio_out]
    )

demo.launch()