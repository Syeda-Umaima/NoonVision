
---

## 🚀 **3. Enhanced app.py (Production-Ready)**

```python
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
CONF_THRESHOLD = 0.25  # Increased for better accuracy
IMG_SIZE = 960  # Optimal for speed/accuracy balance
AUDIO_OUTPUT = "detected_objects.mp3"
BOX_COLOR = (255, 50, 50)  # Bright red for visibility
BOX_WIDTH = 3
FONT_SIZE = 20

# ============================================
# MODEL INITIALIZATION
# ============================================
print("🔄 Loading models... This may take a moment.")

# YOLOv8 Medium - Best balance of speed and accuracy
try:
    model = YOLO("yolov8m.pt")
    # Warm up model
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
    "tell me what you see"
]

# ============================================
# CORE DETECTION FUNCTION
# ============================================
def detect_objects_enhanced(image, conf_threshold=CONF_THRESHOLD):
    """
    Enhanced object detection with better error handling and visualization.
    
    Args:
        image: PIL Image or numpy array
        conf_threshold: Confidence threshold (0.0 to 1.0)
    
    Returns:
        tuple: (annotated_image, description_text, audio_file, status_message)
    """
    start_time = time.time()
    
    # Input validation
    if image is None:
        error_msg = "⚠️ No image provided. Please check your webcam."
        return None, error_msg, None, error_msg
    
    if model is None:
        error_msg = "❌ Detection model not available."
        return None, error_msg, None, error_msg
    
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
        
        detected_objects = []
        detected_labels = []
        
        # Process each detection
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(class_ids[i])
            label = labels[cls_id]
            conf = confidences[i]
            
            # Store detection info
            detected_objects.append(f"{label}: {conf*100:.1f}%")
            detected_labels.append(label)
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)
            
            # Draw label background for better visibility
            text = f"{label} {conf:.2f}"
            bbox = draw.textbbox((x1, y1 - 25), text, font=font)
            draw.rectangle(bbox, fill=BOX_COLOR)
            draw.text((x1, y1 - 25), text, fill="white", font=font)
        
        # Prepare results
        detection_time = time.time() - start_time
        
        if detected_labels:
            # Create description
            description_text = f"🎯 Detected {len(detected_labels)} object(s) in {detection_time:.2f}s:\n\n"
            description_text += "\n".join(detected_objects)
            
            # Generate audio
            audio_file = generate_audio_description(detected_labels, confidences)
            
            status_msg = f"✅ Detection complete! Found {len(detected_labels)} object(s)."
        else:
            description_text = f"ℹ️ No objects detected above {conf_threshold*100:.0f}% confidence.\n\nTry:\n• Better lighting\n• Moving objects closer\n• Lowering confidence threshold"
            audio_file = generate_audio_description([], [])
            status_msg = "ℹ️ No objects detected. Try adjusting the scene."
        
        return img_pil, description_text, audio_file, status_msg
        
    except Exception as e:
        error_msg = f"❌ Detection error: {str(e)}"
        print(error_msg)
        return None, error_msg, None, error_msg

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
            # Count unique objects
            from collections import Counter
            label_counts = Counter(labels)
            
            # Build natural sentence
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
        tts = gTTS(text=tts_text, lang='en', slow=False)
        tts.save(AUDIO_OUTPUT)
        return AUDIO_OUTPUT
        
    except Exception as e:
        print(f"⚠️ Audio generation error: {e}")
        return None

# ============================================
# VOICE COMMAND HANDLER
# ============================================
def process_voice_command(audio_file, image_input, conf_threshold):
    """
    Process voice input and trigger detection if command recognized.
    """
    if audio_file is None:
        return image_input, None, "🎤 No audio recorded. Please try again.", None, "Waiting for command..."
    
    if stt_pipe is None:
        return image_input, None, "❌ Speech recognition unavailable.", None, "STT Error"
    
    try:
        # Transcribe audio
        result = stt_pipe(audio_file)
        transcribed_text = result["text"].strip().lower()
        
        # Check for trigger phrase
        is_triggered = any(phrase in transcribed_text for phrase in TRIGGER_PHRASES)
        
        if is_triggered:
            status_msg = f"✅ Command recognized: '{transcribed_text}'\n🔄 Processing..."
            
            # Run detection
            annotated_img, desc, audio, final_status = detect_objects_enhanced(
                image_input, 
                conf_threshold
            )
            
            return image_input, annotated_img, f"🎤 \"{transcribed_text}\"\n\n{desc}", audio, final_status
        else:
            status_msg = f"❓ Command not recognized: '{transcribed_text}'\n\nTry saying: 'Detect' or 'What do you see?'"
            return image_input, None, status_msg, None, "Command not recognized"
            
    except Exception as e:
        error_msg = f"❌ Voice processing error: {str(e)}"
        return image_input, None, error_msg, None, "Error"

# ============================================
# BUTTON DETECTION HANDLER
# ============================================
def manual_detect(image_input, conf_threshold):
    """
    Manual detection via button click.
    """
    if image_input is None:
        return None, "⚠️ No image available. Please check your webcam.", None, "No image"
    
    annotated_img, desc, audio, status = detect_objects_enhanced(image_input, conf_threshold)
    return annotated_img, desc, audio, status

# ============================================
# GRADIO INTERFACE
# ============================================
with gr.Blocks(
    title="NoonVision - AI Vision Assistant",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green"),
    css="""
        .main-header {text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;}
        .instruction-box {background: #f0f8ff; padding: 15px; border-radius: 8px; border-left: 4px solid #667eea;}
        .status-box {padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;}
    """
) as demo:
    
    # Header
    gr.HTML("""
        <div class="main-header">
            <h1>🦾 NoonVision - AI Vision Assistant</h1>
            <p>Hands-Free Object Detection for the Visually Impaired</p>
        </div>
    """)
    
    # Instructions
    with gr.Row():
        gr.Markdown("""
        <div class="instruction-box">
        <h3>📋 How to Use:</h3>
        <ol>
            <li><strong>Voice Control (Recommended):</strong> Click the 🎤 microphone and say <strong>"Detect"</strong> or <strong>"What do you see?"</strong></li>
            <li><strong>Manual Mode:</strong> Click the <strong>"Detect Objects"</strong> button</li>
            <li><strong>Results:</strong> Listen to the audio description (auto-plays)</li>
        </ol>
        <p><strong>💡 Tip:</strong> Ensure good lighting and position objects 2-6 feet from camera for best results.</p>
        </div>
        """)
    
    # Main Interface
    with gr.Row():
        # Left Column - Camera and Output
        with gr.Column(scale=2):
            image_input = gr.Image(
                type="pil",
                source="webcam",
                streaming=True,
                label="📷 Live Webcam Feed",
                height=400
            )
            
            image_output = gr.Image(
                type="pil",
                label="🎯 Detection Results",
                height=400
            )
        
        # Right Column - Controls and Results
        with gr.Column(scale=1):
            gr.Markdown("### 🎤 Voice Command")
            voice_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Click to Record (Say 'Detect')"
            )
            
            gr.Markdown("### ⚙️ Settings")
            conf_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.25,
                step=0.05,
                label="Confidence Threshold",
                info="Higher = fewer but more certain detections"
            )
            
            detect_button = gr.Button(
                "🔍 Detect Objects",
                variant="primary",
                size="lg"
            )
            
            gr.Markdown("### 📊 Status")
            status_output = gr.Textbox(
                label="System Status",
                value="Ready to detect objects",
                lines=2
            )
            
            gr.Markdown("### 📝 Detection Details")
            text_output = gr.Textbox(
                label="Detected Objects",
                lines=6
            )
            
            gr.Markdown("### 🔊 Audio Output")
            audio_output = gr.Audio(
                type="filepath",
                autoplay=True,
                label="Spoken Results"
            )
    
    # Event Handlers
    voice_input.change(
        fn=process_voice_command,
        inputs=[voice_input, image_input, conf_slider],
        outputs=[image_input, image_output, text_output, audio_output, status_output],
        show_progress="full"
    )
    
    detect_button.click(
        fn=manual_detect,
        inputs=[image_input, conf_slider],
        outputs=[image_output, text_output, audio_output, status_output],
        show_progress="full"
    )
    
    # Footer
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Made with ❤️ for accessibility | Powered by YOLOv8 + Whisper + Gradio</p>
        <p>🌟 If this helps you, please share it with others who might benefit!</p>
    </div>
    """)

# ============================================
# LAUNCH
# ============================================
if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=True,
        show_api=False
    )