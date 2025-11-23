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
import threading

# ============================================
# CONFIGURATION - OPTIMIZED FOR SPEED
# ============================================
CONF_THRESHOLD = 0.35  # Increased from 0.25 for better accuracy
IMG_SIZE = 640  # Reduced from 960 for faster inference
BOX_COLOR = (255, 50, 50)
BOX_WIDTH = 3
FONT_SIZE = 20
DETECTION_COOLDOWN = 1.5  # Prevent rapid repeated detections

# ============================================
# GPU DETECTION AND OPTIMIZATION
# ============================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  Running on: {device.upper()}")

if device == 'cuda':
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    torch.backends.cudnn.benchmark = True  # Auto-optimize for your GPU
else:
    print("⚠️  Running on CPU - Slower performance expected")

# ============================================
# MODEL INITIALIZATION - OPTIMIZED
# ============================================
print("🔄 Loading models... This may take a moment.")

# YOLOv8 Nano (Faster) or Medium (More Accurate) - Choose based on your needs
try:
    # For SPEED: Use 'yolov8n.pt' (nano - fastest)
    # For ACCURACY: Use 'yolov8m.pt' (medium - more accurate)
    # For BALANCE: Use 'yolov8s.pt' (small - good balance)
    
    model = YOLO("yolov8s.pt")  # Small model for balanced performance
    model.to(device)  # Move model to GPU if available
    
    # Warmup run to optimize model
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = model(dummy, verbose=False, device=device)
    print(f"✅ YOLOv8s loaded on {device.upper()}")
except Exception as e:
    print(f"❌ YOLO model failed to load: {e}")
    model = None

# Whisper for Speech Recognition - Using Tiny for Speed
try:
    whisper_device = 0 if torch.cuda.is_available() else -1
    stt_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        device=whisper_device,
        chunk_length_s=5,  # Process in 5-second chunks for faster response
    )
    print(f"✅ Whisper STT loaded on {'GPU' if whisper_device == 0 else 'CPU'}")
except Exception as e:
    print(f"⚠️ STT model failed to load: {e}")
    stt_pipe = None

# ============================================
# TRIGGER PHRASES - EXPANDED
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
    "look",
    "show me",
    "recognize",
    "find objects",
    "what objects"
]

# ============================================
# AUDIO CACHE FOR FASTER PLAYBACK
# ============================================
audio_cache = {}
last_detection_time = 0

def generate_audio_description(labels):
    """Generate natural-sounding audio description with caching."""
    global audio_cache
    
    try:
        if not labels:
            tts_text = "I couldn't detect any objects. Please try again with better lighting."
        else:
            label_counts = Counter(labels)
            
            if len(label_counts) == 1:
                obj, count = list(label_counts.items())[0]
                if count == 1:
                    tts_text = f"I see one {obj}."
                else:
                    # Handle plural forms better
                    plural_obj = obj if obj.endswith('s') else f"{obj}s"
                    tts_text = f"I see {count} {plural_obj}."
            else:
                tts_text = "I see "
                items = []
                for obj, count in label_counts.items():
                    if count == 1:
                        items.append(f"one {obj}")
                    else:
                        plural_obj = obj if obj.endswith('s') else f"{obj}s"
                        items.append(f"{count} {plural_obj}")
                
                if len(items) == 2:
                    tts_text += f"{items[0]} and {items[1]}."
                else:
                    tts_text += ", ".join(items[:-1]) + f", and {items[-1]}."
        
        # Check cache first
        if tts_text in audio_cache:
            return audio_cache[tts_text]
        
        # Generate audio with unique filename
        timestamp = int(time.time() * 1000)
        audio_file = f"detected_{timestamp}.mp3"
        tts = gTTS(text=tts_text, lang='en', slow=False)
        tts.save(audio_file)
        
        # Cache the audio file
        audio_cache[tts_text] = audio_file
        
        # Limit cache size
        if len(audio_cache) > 20:
            oldest = list(audio_cache.keys())[0]
            old_file = audio_cache.pop(oldest)
            if os.path.exists(old_file):
                try:
                    os.remove(old_file)
                except:
                    pass
        
        return audio_file
        
    except Exception as e:
        print(f"⚠️ Audio generation error: {e}")
        return None

def detect_objects_enhanced(image, conf_threshold=CONF_THRESHOLD):
    """Enhanced object detection with visualization - OPTIMIZED."""
    global last_detection_time
    
    if image is None or model is None:
        return None, None
    
    try:
        start_time = time.time()
        
        # Prepare image for YOLO
        if isinstance(image, Image.Image):
            img_np = np.array(image)
            img_pil = image.copy()
        else:
            img_np = image
            img_pil = Image.fromarray(image)
        
        # Run detection with optimized settings
        results = model(
            img_np,
            imgsz=IMG_SIZE,
            conf=conf_threshold,
            verbose=False,
            device=device,
            half=True if device == 'cuda' else False,  # Use FP16 on GPU for speed
            max_det=50  # Limit max detections for speed
        )[0]
        
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
            
            # Draw bounding box and label
            draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)
            text = f"{label} {conf:.2f}"
            bbox = draw.textbbox((x1, y1 - 25), text, font=font)
            draw.rectangle(bbox, fill=BOX_COLOR)
            draw.text((x1, y1 - 25), text, fill="white", font=font)
        
        # Generate audio in background thread for speed
        audio_file = None
        
        def generate_audio_async():
            nonlocal audio_file
            audio_file = generate_audio_description(detected_labels)
        
        audio_thread = threading.Thread(target=generate_audio_async)
        audio_thread.start()
        audio_thread.join(timeout=1.0)  # Wait max 1 second for audio
        
        detection_time = time.time() - start_time
        print(f"⚡ Detection completed in {detection_time:.2f}s")
        
        last_detection_time = time.time()
        
        return img_pil, audio_file
        
    except Exception as e:
        print(f"❌ Detection error: {str(e)}")
        return None, None

# ============================================
# STREAMING HANDLER FUNCTIONS
# ============================================

def transcribe_streaming_audio(audio_tuple):
    """Continuously transcribes audio - OPTIMIZED."""
    if audio_tuple is None or stt_pipe is None:
        return ""
    
    try:
        sample_rate, audio_data = audio_tuple
        
        # Skip very short audio clips (noise)
        if len(audio_data) < sample_rate * 0.5:  # Less than 0.5 seconds
            return ""
        
        # Convert audio data to proper format (mono, float32)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        audio_data = audio_data.astype(np.float32)
        if audio_data.max() > 1.0:
            audio_data = audio_data / 32768.0

        # Transcribe with optimized settings
        result = stt_pipe(
            {"sampling_rate": sample_rate, "raw": audio_data},
            return_timestamps=False  # Faster without timestamps
        )
        transcribed_text = result["text"].strip().lower()

        # Only return text if it seems like actual speech
        if len(transcribed_text) > 2:
            return transcribed_text
        
        return ""
            
    except Exception as e:
        print(f"⚠️ Voice processing error: {e}")
        return ""

def process_camera_stream(latest_frame, transcribed_text, last_detection_image):
    """Main loop - OPTIMIZED with cooldown."""
    global last_detection_time
    
    # 1. Check for basic camera input availability
    if latest_frame is None:
        return latest_frame, last_detection_image, None, "📷 Waiting for Camera Input...", ""

    # 2. Check for cooldown to prevent rapid detections
    current_time = time.time()
    time_since_last = current_time - last_detection_time
    
    # 3. Check for trigger command
    is_triggered = any(phrase in transcribed_text for phrase in TRIGGER_PHRASES)

    if is_triggered:
        # Check cooldown
        if time_since_last < DETECTION_COOLDOWN:
            remaining = DETECTION_COOLDOWN - time_since_last
            status_msg = f"⏱️ Please wait {remaining:.1f}s before next detection..."
            return latest_frame, last_detection_image, None, status_msg, transcribed_text
        
        print(f"🎤 Trigger detected: '{transcribed_text}'")
        
        # Run detection on the latest frame
        annotated_img, audio_file = detect_objects_enhanced(latest_frame, CONF_THRESHOLD)
        
        if annotated_img is not None:
            status_msg = f"✅ Detected! Command: '{transcribed_text}'"
        else:
            status_msg = f"❌ Detection failed. Please try again."
        
        # Return the annotated image and audio, and clear the transcription state
        return latest_frame, annotated_img, audio_file, status_msg, ""
    
    # 4. No trigger: just maintain the live feed
    if transcribed_text:
        status_msg = f"🎤 Listening... Last: '{transcribed_text}' (Say 'Detect' to scan)"
    else:
        status_msg = "🎤 Listening... Say 'Detect' or 'What do you see?'"
    
    return latest_frame, last_detection_image, None, status_msg, transcribed_text

# ============================================
# GRADIO INTERFACE - MINIMAL DESIGN
# ============================================

AUTO_START_JS = """
function start_mic_stream() {
    const mic_component = document.getElementById("hidden_voice_input");
    if (mic_component) {
        const start_button = mic_component.querySelector("button");
        if (start_button && !start_button.classList.contains('hidden')) {
            start_button.click();
            console.log("✅ Auto-start: Microphone stream initiated.");
            return;
        }
    }
    // Retry after a short delay if not ready
    setTimeout(start_mic_stream, 500);
}
setTimeout(start_mic_stream, 500); 
"""

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
        #hidden_voice_input {
            display: none !important;
            visibility: hidden !important;
            height: 0px !important;
        }
        .status-box {
            font-weight: bold;
            font-size: 1.1em;
            color: #764ba2;
            padding: 10px;
            background: #f9f9f9;
            border-radius: 5px;
        }
        .performance-info {
            background: #e8f5e9;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            text-align: center;
        }
    """,
    js=AUTO_START_JS
) as demo:
    
    # Header
    device_emoji = "🚀" if device == 'cuda' else "🖥️"
    gr.HTML(f"""
        <div class="main-header">
            <h1>🦾 NoonVision - AI Vision Assistant</h1>
            <p>Hands-Free Voice-Activated Object Detection {device_emoji}</p>
            <p style="font-size: 0.9em; margin-top: 5px;">Running on: {device.upper()}</p>
        </div>
    """)
    
    # Instructions
    gr.Markdown("""
    <div class="instruction-box">
    <h3>🎤 Voice-Activated Mode - OPTIMIZED FOR SPEED</h3>
    <p>This app **automatically starts listening** once you grant **microphone and camera permissions**.</p>
    
    <h4>📢 How to Use:</h4>
    <ol>
        <li><strong>Allow permissions</strong> when prompted</li>
        <li><strong>Simply say:</strong> "Detect", "What do you see?", or "Scan"</li>
        <li><strong>Wait 1-3 seconds</strong> for detection to complete</li>
        <li><strong>Listen</strong> to the audio description</li>
        <li><strong>Repeat</strong> anytime - with 1.5s cooldown between detections</li>
    </ol>
    
    <p><strong>⚡ Performance Tips:</strong></p>
    <ul>
        <li>Good lighting = faster & more accurate detection</li>
        <li>Keep objects 2-6 feet from camera</li>
        <li>Speak clearly and wait for results</li>
        <li>Maximum 1 detection every 1.5 seconds</li>
    </ul>
    </div>
    """)
    
    # Main Interface - Camera and Results
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                sources="webcam",
                label="📷 Live Camera Feed",
                streaming=True,
                interactive=False,
                height=500
            )
        
        with gr.Column(scale=1):
            image_output = gr.Image(
                type="pil",
                label="🎯 Detection Results",
                height=500,
                value=None
            )

    gr.Markdown("### 📊 System Status")
    status_output = gr.Textbox(
        label="",
        value="Waiting for permissions... then listening.",
        lines=2,
        elem_classes="status-box"
    )
    
    # Performance indicator
    gr.HTML(f"""
        <div class="performance-info">
            <strong>⚡ Detection Mode:</strong> Balanced (YOLOv8s) | 
            <strong>Target Speed:</strong> 1-3 seconds | 
            <strong>Confidence:</strong> 35% | 
            <strong>Device:</strong> {device.upper()}
        </div>
    """)
    
    # Hidden components
    voice_input = gr.Audio(
        sources="microphone",
        type="numpy",
        streaming=True,
        visible=False,
        elem_id="hidden_voice_input" 
    )
    
    transcribed_state = gr.Textbox(
        value="", 
        visible=False
    )
    
    audio_output = gr.Audio(
        type="filepath",
        autoplay=True,
        visible=False
    )

    # --- Event Handlers ---
    
    # Voice Stream Handler
    voice_input.stream(
        fn=transcribe_streaming_audio,
        inputs=[voice_input],
        outputs=[transcribed_state],
        show_progress=False
    )

    # Camera Stream Handler (Optimized timing)
    image_input.stream(
        fn=process_camera_stream,
        inputs=[image_input, transcribed_state, image_output],
        outputs=[image_input, image_output, audio_output, status_output, transcribed_state],
        every=0.2,  # Check every 200ms (balanced speed)
        show_progress=False
    )
    
    # Footer
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>🎙️ Always Listening</strong> | <strong>🔊 Auto-Play Results</strong> | <strong>♿ Fully Accessible</strong></p>
        <p><strong>⚡ Optimized for Speed</strong> | <strong>🎯 Balanced Accuracy</strong> | <strong>🚀 GPU Accelerated</strong></p>
        <p>Powered by YOLOv8s + Whisper + Gradio | Made with ❤️ for accessibility</p>
    </div>
    """)

# ============================================
# LAUNCH
# ============================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 NoonVision Starting...")
    print(f"📊 Device: {device.upper()}")
    print(f"🎯 Model: YOLOv8s (Balanced)")
    print(f"⚡ Target Detection Time: 1-3 seconds")
    print("="*50 + "\n")
    
    demo.launch(
        share=False,
        show_error=True,
        show_api=False
    )