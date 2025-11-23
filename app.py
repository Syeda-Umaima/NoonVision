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
    """Enhanced object detection with visualization."""
    start_time = time.time()
    
    if image is None:
        return None, None, None
    
    if model is None:
        return None, None, None
    
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
        
        # Generate audio
        audio_file = generate_audio_description(detected_labels)
        
        return img_pil, audio_file, None
        
    except Exception as e:
        print(f"❌ Detection error: {str(e)}")
        return None, None, None

# ============================================
# AUDIO GENERATION
# ============================================
def generate_audio_description(labels):
    """Generate natural-sounding audio description."""
    try:
        if not labels:
            tts_text = "I couldn't detect any objects. Please try again with better lighting."
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
                tts_text = "I see "
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
        
        # Generate audio with unique filename
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
    """Process continuously streaming audio for always-listening mode."""
    
    # Return empty if no audio or no models loaded
    if audio_tuple is None or stt_pipe is None or image_input is None:
        return image_input, None, None
    
    try:
        sample_rate, audio_data = audio_tuple
        
        # Convert audio data to proper format
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Normalize audio
        audio_data = audio_data.astype(np.float32)
        if audio_data.max() > 1.0:
            audio_data = audio_data / 32768.0
        
        # Transcribe
        result = stt_pipe({"sampling_rate": sample_rate, "raw": audio_data})
        transcribed_text = result["text"].strip().lower()
        
        # Only process if there's actual speech (more than 2 characters)
        if len(transcribed_text) < 2:
            return image_input, None, None
        
        # Check for trigger phrase
        is_triggered = any(phrase in transcribed_text for phrase in TRIGGER_PHRASES)
        
        if is_triggered:
            # Run detection
            annotated_img, audio, _ = detect_objects_enhanced(image_input, CONF_THRESHOLD)
            
            if annotated_img is not None and audio is not None:
                return image_input, annotated_img, audio
        
        # Continue listening silently
        return image_input, None, None
            
    except Exception as e:
        # Silent error handling - just continue listening
        return image_input, None, None

# ============================================
# GRADIO INTERFACE - MINIMAL DESIGN
# ============================================
with gr.Blocks(
    title="NoonVision - AI Vision Assistant",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green"),
    css="""
        .main-header {
            text-align: center; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            border-radius: 10px; 
            margin-bottom: 20px;
        }
        .instruction-box {
            background: #f0f8ff; 
            padding: 20px; 
            border-radius: 8px; 
            border-left: 4px solid #667eea;
            margin-bottom: 20px;
        }
    """
) as demo:
    
    # Header
    gr.HTML("""
        <div class="main-header">
            <h1>🦾 NoonVision - AI Vision Assistant</h1>
            <p>Hands-Free Voice-Activated Object Detection</p>
        </div>
    """)
    
    # Instructions
    gr.Markdown("""
    <div class="instruction-box">
    <h3>🎤 Voice-Activated Mode</h3>
    <p>When you open this app, it will request <strong>microphone and camera permissions</strong>.</p>
    <p>Once granted, the app will <strong>automatically start listening</strong> for your voice commands.</p>
    
    <h4>📢 How to Use:</h4>
    <ol>
        <li><strong>Allow permissions</strong> when prompted</li>
        <li><strong>Simply say:</strong> "Detect" or "What do you see?"</li>
        <li><strong>Listen</strong> to the audio description of detected objects</li>
        <li><strong>Repeat</strong> anytime - no buttons to click!</li>
    </ol>
    
    <p><strong>💡 Tips:</strong></p>
    <ul>
        <li>Position objects 2-6 feet from camera</li>
        <li>Ensure good lighting for best detection</li>
        <li>Speak clearly: "Detect", "What's in front of me?", "Identify objects"</li>
    </ul>
    </div>
    """)
    
    # Main Interface - Just Camera and Results
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                sources=["webcam"],  # Changed from 'source' to 'sources'
                streaming=True,
                label="📷 Live Camera Feed",
                height=500
            )
        
        with gr.Column(scale=1):
            image_output = gr.Image(
                type="pil",
                label="🎯 Detection Results",
                height=500
            )
    
    # Hidden components for voice and audio (no UI elements shown)
    voice_input = gr.Audio(
        sources=["microphone"],
        type="numpy",
        streaming=True,
        visible=False  # Hidden - auto-starts on load
    )
    
    audio_output = gr.Audio(
        type="filepath",
        autoplay=True,
        visible=False  # Hidden - plays automatically
    )
    
    # Event Handler - Streaming voice processing
    voice_input.stream(
        fn=process_streaming_voice,
        inputs=[voice_input, image_input],
        outputs=[image_input, image_output, audio_output],
        show_progress=False
    )
    
    # Footer
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>🎙️ Always Listening</strong> | <strong>🔊 Auto-Play Results</strong> | <strong>♿ Fully Accessible</strong></p>
        <p>Powered by YOLOv8 + Whisper + Gradio | Made with ❤️ for accessibility</p>
    </div>
    """)

# ============================================
# LAUNCH WITH AUTO-PERMISSION REQUEST
# ============================================
if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=True,
        show_api=False
    )