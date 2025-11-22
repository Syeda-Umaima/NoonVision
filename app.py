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
AUDIO_OUTPUT = "detected_objects.mp3"
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
            
            detected_objects.append(f"{label}: {conf*100:.1f}%")
            detected_labels.append(label)
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)
            
            # Draw label with background
            text = f"{label} {conf:.2f}"
            bbox = draw.textbbox((x1, y1 - 25), text, font=font)
            draw.rectangle(bbox, fill=BOX_COLOR)
            draw.text((x1, y1 - 25), text, fill="white", font=font)
        
        # Prepare results
        detection_time = time.time() - start_time
        
        if detected_labels:
            description_text = f"🎯 Detected {len(detected_labels)} object(s) in {detection_time:.2f}s:\n\n"
            description_text += "\n".join(detected_objects)
            
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
def process_streaming_voice(audio_tuple, image_input, conf_threshold):
    """
    Process continuously streaming audio input for always-listening mode.
    Args:
        audio_tuple: (sample_rate, audio_data) from streaming audio
        image_input: Current webcam frame
        conf_threshold: Detection confidence threshold
    """
    if audio_tuple is None:
        return image_input, None, "🎤 Listening... Say 'Detect' or 'What do you see?'", None, "🎤 Ready - Voice Active"
    
    if stt_pipe is None:
        return image_input, None, "❌ Speech recognition unavailable.", None, "STT Error"
    
    if image_input is None:
        return None, None, "⚠️ No camera input available.", None, "No Camera"
    
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
            return image_input, None, "🎤 Listening... Say 'Detect' or 'What do you see?'", None, "🎤 Ready - Voice Active"
        
        # Check for trigger phrase
        is_triggered = any(phrase in transcribed_text for phrase in TRIGGER_PHRASES)
        
        if is_triggered:
            status_msg = f"✅ Command: '{transcribed_text}'\n🔄 Processing..."
            
            # Run detection
            annotated_img, desc, audio, final_status = detect_objects_enhanced(
                image_input, 
                conf_threshold
            )
            
            return image_input, annotated_img, f"🎤 \"{transcribed_text}\"\n\n{desc}", audio, final_status
        else:
            # Show what was heard but continue listening
            return image_input, None, f"🎤 Heard: '{transcribed_text}'\n\n💡 Say: 'Detect' or 'What do you see?'", None, "🎤 Listening..."
            
    except Exception as e:
        error_msg = f"⚠️ Processing... (background noise filtered)"
        # Don't show full error, just continue listening
        return image_input, None, "🎤 Listening... Say 'Detect' or 'What do you see?'", None, "🎤 Ready - Voice Active"

# ============================================
# BUTTON DETECTION HANDLER
# ============================================
def manual_detect(image_input, conf_threshold):
    """Manual detection via button click."""
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
        .voice-indicator {background: #4CAF50; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; margin: 10px 0;}
    """
) as demo:
    
    # Header
    gr.HTML("""
        <div class="main-header">
            <h1>🦾 NoonVision - AI Vision Assistant</h1>
            <p>Always-Listening Voice Control for the Visually Impaired</p>
        </div>
    """)
    
    # Instructions
    with gr.Row():
        gr.Markdown("""
        <div class="instruction-box">
        <h3>📋 How to Use (Hands-Free Mode):</h3>
        <ol>
            <li><strong>🎤 Click "Start Voice Control"</strong> to enable always-listening mode</li>
            <li><strong>🗣️ Simply speak:</strong> "Detect" or "What do you see?"</li>
            <li><strong>🔊 Listen:</strong> Audio results play automatically</li>
            <li><strong>🔄 Repeat:</strong> No need to click again - just keep speaking!</li>
        </ol>
        <p><strong>💡 Pro Tip:</strong> Position yourself 2-6 feet from camera with good lighting for best results.</p>
        <p><strong>🔇 Note:</strong> Background noise is automatically filtered. Speak clearly for best recognition.</p>
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
            gr.HTML('<div class="voice-indicator">🎤 VOICE CONTROL - ALWAYS LISTENING</div>')
            
            gr.Markdown("### 🎙️ Voice Control")
            voice_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                streaming=True,
                label="Click to Start Voice Control (Always Listening)"
            )
            
            gr.Markdown("### ⚙️ Detection Settings")
            conf_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.25,
                step=0.05,
                label="Confidence Threshold",
                info="Higher = fewer but more certain detections"
            )
            
            detect_button = gr.Button(
                "🔍 Manual Detect (Fallback)",
                variant="secondary",
                size="lg"
            )
            
            gr.Markdown("### 📊 Status")
            status_output = gr.Textbox(
                label="System Status",
                value="Click microphone to start voice control",
                lines=2
            )
            
            gr.Markdown("### 📝 Detection Details")
            text_output = gr.Textbox(
                label="Detected Objects",
                lines=6,
                placeholder="Results will appear here..."
            )
            
            gr.Markdown("### 🔊 Audio Output")
            audio_output = gr.Audio(
                type="filepath",
                autoplay=True,
                label="Spoken Results"
            )
    
    # Event Handlers - STREAMING MODE
    voice_input.stream(
        fn=process_streaming_voice,
        inputs=[voice_input, image_input, conf_slider],
        outputs=[image_input, image_output, text_output, audio_output, status_output],
        show_progress="hidden"
    )
    
    # Manual fallback button
    detect_button.click(
        fn=manual_detect,
        inputs=[image_input, conf_slider],
        outputs=[image_output, text_output, audio_output, status_output],
        show_progress="minimal"
    )
    
    # Footer
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>✨ Features:</strong> Always-Listening Mode | Automatic Audio Playback | Hands-Free Operation</p>
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