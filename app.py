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
IMG_SIZE = 960
BOX_COLOR = (255, 50, 50)
BOX_WIDTH = 3
FONT_SIZE = 20

# ============================================
# MODEL INITIALIZATION
# ============================================
print("🔄 Loading models... This may take a moment.")

# YOLOv8 Medium
try:
    model = YOLO("yolov8m.pt")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = model(dummy, verbose=False)
    print("✅ YOLOv8m loaded successfully")
except Exception as e:
    print(f"❌ YOLO model failed to load: {e}")
    model = None

# Whisper for Speech Recognition
try:
    device = 0 if torch.cuda.is_available() else -1
    stt_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        device=device
    )
    print(f"✅ Whisper STT loaded on {'GPU' if device == 0 else 'CPU'}")
except Exception as e:
    print(f"⚠️ STT model failed to load: {e}")
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
# CORE DETECTION FUNCTION
# ============================================
def detect_objects_enhanced(image, conf_threshold=CONF_THRESHOLD):
    """
    Enhanced object detection with better error handling and visualization.
    """
    start_time = time.time()
    
    if image is None:
        return None, "⚠️ No image provided. Please check your webcam."
    
    if model is None:
        return None, "❌ Detection model not available."
    
    try:
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            img_np = np.array(image)
            img_pil = image.copy()
        else:
            img_np = image
            img_pil = Image.fromarray(image)
        
        # Run detection
        results = model(img_np, imgsz=IMG_SIZE, conf=conf_threshold, verbose=False)[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        labels = results.names
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        
        # Prepare drawing
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", FONT_SIZE)
        except:
            font = ImageFont.load_default()
        
        detected_labels = []
        
        # Process each detection
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(class_ids[i])
            label = labels[cls_id]
            conf = confidences[i]
            
            detected_labels.append(label)
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)
            
            # Draw label with background
            text = f"{label} {conf:.2f}"
            bbox = draw.textbbox((x1, y1 - 25), text, font=font)
            draw.rectangle(bbox, fill=BOX_COLOR)
            draw.text((x1, y1 - 25), text, fill="white", font=font)
        
        # Generate audio description
        audio_file = generate_audio_description(detected_labels, confidences)
        
        return img_pil, audio_file
        
    except Exception as e:
        print(f"❌ Detection error: {str(e)}")
        return None, None

# ============================================
# AUDIO GENERATION
# ============================================
def generate_audio_description(labels, confidences):
    """
    Generate natural-sounding audio description of detected objects.
    """
    try:
        if not labels:
            tts_text = "I couldn't detect any objects in the image. Please try again with better lighting or move objects closer to the camera."
        else:
            from collections import Counter
            label_counts = Counter(labels)
            
            if len(label_counts) == 1:
                obj, count = list(label_counts.items())[0]
                if count == 1:
                    tts_text = f"I see one {obj}."
                else:
                    tts_text = f"I see {count} {obj}s."
            else:
                tts_text = "I see the following objects: "
                items = []
                for obj, count in label_counts.items():
                    if count == 1:
                        items.append(f"one {obj}")
                    else:
                        items.append(f"{count} {obj}s")
                
                if len(items) == 2:
                    tts_text += f"{items[0]} and {items[1]}."
                else:
                    tts_text += ", ".join(items[:-1]) + f", and {items[-1]}."
        
        # Generate audio
        timestamp = int(time.time() * 1000)
        audio_file = f"detected_{timestamp}.mp3"
        tts = gTTS(text=tts_text, lang='en', slow=False)
        tts.save(audio_file)
        return audio_file
        
    except Exception as e:
        print(f"⚠️ Audio generation error: {e}")
        return None

# ============================================
# STREAMING VOICE HANDLER
# ============================================
def process_streaming_voice(audio_tuple, image_input):
    """
    Process continuously streaming audio input for always-listening mode.
    Automatically captures image and runs detection when trigger phrase is heard.
    """
    # Return empty outputs while listening (no visual feedback needed)
    if audio_tuple is None:
        return None, None
    
    if stt_pipe is None:
        return None, None
    
    if image_input is None:
        return None, None
    
    try:
        sample_rate, audio_data = audio_tuple
        
        # Convert audio data to format expected by pipeline
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Normalize audio
        audio_data = audio_data.astype(np.float32)
        if audio_data.max() > 1.0:
            audio_data = audio_data / 32768.0
        
        # Transcribe
        result = stt_pipe({"sampling_rate": sample_rate, "raw": audio_data})
        transcribed_text = result["text"].strip().lower()
        
        # Only process if there's actual speech
        if len(transcribed_text) < 2:
            return None, None
        
        # Check for trigger phrase
        is_triggered = any(phrase in transcribed_text for phrase in TRIGGER_PHRASES)
        
        if is_triggered:
            print(f"🎯 Trigger detected: '{transcribed_text}'")
            
            # Run detection immediately
            annotated_img, audio_output = detect_objects_enhanced(image_input, CONF_THRESHOLD)
            
            # Return results - audio will autoplay
            return annotated_img, audio_output
        else:
            # Continue listening silently
            return None, None
            
    except Exception as e:
        # Silently continue listening on errors (background noise, etc.)
        return None, None

# ============================================
# GRADIO INTERFACE - MINIMAL DESIGN
# ============================================
with gr.Blocks(
    title="NoonVision - Voice-Activated AI Vision",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green"),
    css="""
        .gradio-container {max-width: 1200px !important; margin: auto;}
        .main-header {text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                      color: white; border-radius: 15px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
        .main-header h1 {margin: 0; font-size: 2.5em; font-weight: bold;}
        .main-header p {margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.95;}
        .hidden-audio {display: none;}
        .image-container {border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1);}
    """
) as demo:
    
    # Header
    gr.HTML("""
        <div class="main-header">
            <h1>🦾 NoonVision</h1>
            <p>🎤 Voice-Activated AI Vision Assistant</p>
            <p style="font-size: 0.9em; margin-top: 15px;">
                👋 Just say <strong>"Detect"</strong> and I'll tell you what I see
            </p>
        </div>
    """)
    
    # Main content - only camera feed and results
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📷 Live Camera")
            image_input = gr.Image(
                type="pil",
                source="webcam",
                streaming=True,
                label="",
                height=450,
                elem_classes="image-container"
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Detection Results")
            image_output = gr.Image(
                type="pil",
                label="",
                height=450,
                elem_classes="image-container"
            )
    
    # Hidden audio component for voice input (streaming, always on)
    voice_input = gr.Audio(
        sources=["microphone"],
        type="numpy",
        streaming=True,
        label="",
        elem_classes="hidden-audio",
        autoplay=False,
        visible=False
    )
    
    # Hidden audio output component (autoplay enabled)
    audio_output = gr.Audio(
        type="filepath",
        autoplay=True,
        label="",
        elem_classes="hidden-audio",
        visible=False
    )
    
    # Instructions
    gr.Markdown("""
    ---
    ### 📖 How It Works:
    1. **🎤 Grant microphone permission** when prompted (one-time only)
    2. **📷 Position yourself** in front of the camera
    3. **🗣️ Say "Detect"** (or "What do you see?", "Scan", etc.)
    4. **🔊 Listen** to the automatic audio description
    5. **🔄 Repeat** anytime - no clicking needed!
    
    💡 **Tip:** Works best with good lighting and objects 2-6 feet from camera
    
    ---
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Made with ❤️ for accessibility | Powered by YOLOv8 + Whisper + Gradio</p>
        <p>🌟 Helping the visually impaired navigate their world independently</p>
    </div>
    """)
    
    # Event Handler - STREAMING MODE (silent, always listening)
    voice_input.stream(
        fn=process_streaming_voice,
        inputs=[voice_input, image_input],
        outputs=[image_output, audio_output],
        show_progress=False
    )

# ============================================
# LAUNCH
# ============================================
if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=True,
        show_api=False
    )