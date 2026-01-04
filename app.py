# =============================================================================
# 🦾 NOONVISION - Hands-Free AI Vision Assistant
# =============================================================================
# A voice-controlled object detection system designed for blind and
# visually impaired users. Built with YOLOv8, Gradio, and Web Speech API.
# =============================================================================

# Fix for HfFolder import error in newer huggingface_hub versions
try:
    from huggingface_hub import HfFolder
except ImportError:
    import huggingface_hub
    class HfFolder:
        @classmethod
        def get_token(cls): return None
        @classmethod  
        def save_token(cls, token): pass
        @classmethod
        def delete_token(cls): pass
    huggingface_hub.HfFolder = HfFolder

import gradio as gr
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import time
from collections import Counter
import os
import base64
import tempfile
import io

# =============================================================================
# CONFIGURATION
# =============================================================================

print("=" * 60)
print("🦾 NOONVISION - Hands-Free AI Vision Assistant")
print("=" * 60)

CONF_THRESHOLD = 0.30
IMG_SIZE = 640
VERSION = "1.0.0"

# =============================================================================
# MODEL LOADING
# =============================================================================

model = None
try:
    print("📦 Loading YOLOv8 model...")
    model = YOLO("yolov8m.pt")
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    _ = model(dummy, verbose=False)
    print("✅ YOLOv8m model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading error: {e}")

# =============================================================================
# AUDIO FUNCTIONS
# =============================================================================

def make_audio_file(text):
    """Create TTS audio file and return path"""
    try:
        fd, path = tempfile.mkstemp(suffix='.mp3')
        os.close(fd)
        gTTS(text=text, lang='en', slow=False).save(path)
        return path
    except Exception as e:
        print(f"Audio error: {e}")
        return None

def get_audio_base64(text):
    """Create TTS audio and return as base64 string"""
    try:
        fd, path = tempfile.mkstemp(suffix='.mp3')
        os.close(fd)
        gTTS(text=text, lang='en', slow=False).save(path)
        with open(path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        os.remove(path)
        return b64
    except Exception as e:
        print(f"Audio error: {e}")
        return ""

# Pre-generate audio
print("🔊 Generating startup audio...")
STARTUP_AUDIO_B64 = get_audio_base64(
    "Welcome to NoonVision. Your hands-free AI vision assistant is ready. "
    "Simply say 'detect' to identify objects around you."
)
PROCESSING_AUDIO_B64 = get_audio_base64("Analyzing. Please wait.")
print("✅ Audio ready!")

# =============================================================================
# DETECTION FUNCTION
# =============================================================================

def detect_objects(image_data):
    """
    Main detection function - receives base64 image from JavaScript
    Returns: (result_image, audio_file, status_text)
    """
    print(f"\n{'='*50}")
    print("🔍 DETECTION STARTED")
    print(f"{'='*50}")
    
    if not image_data or len(image_data) < 100:
        print("❌ No valid image data received")
        audio = make_audio_file("I couldn't capture an image. Please make sure your camera is working and try again.")
        return None, audio, "❌ No image captured"
    
    if model is None:
        audio = make_audio_file("The detection model is not ready. Please wait a moment and try again.")
        return None, audio, "❌ Model not loaded"
    
    try:
        t0 = time.time()
        
        # Decode base64 image
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        image_bytes = base64.b64decode(image_data)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        np_image = np.array(pil_image)
        
        print(f"📷 Image size: {np_image.shape[1]}x{np_image.shape[0]}")
        
        # Run YOLO detection
        results = model(np_image, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        class_names = results.names
        
        print(f"🎯 Detected {len(boxes)} objects")
        
        # Draw bounding boxes with enhanced styling
        draw = ImageDraw.Draw(pil_image)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
            small_font = font
        
        detected_objects = []
        colors = [
            (46, 204, 113), (52, 152, 219), (155, 89, 182), 
            (241, 196, 15), (230, 126, 34), (231, 76, 60),
            (26, 188, 156), (52, 73, 94)
        ]
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = class_names[int(class_ids[i])]
            conf = confidences[i]
            detected_objects.append(label)
            
            color = colors[i % len(colors)]
            
            # Draw filled rectangle with transparency effect
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            
            # Draw label background
            text = f"{label} {conf:.0%}"
            bbox = draw.textbbox((x1, y1 - 30), text, font=font)
            padding = 5
            draw.rectangle(
                [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding], 
                fill=color
            )
            draw.text((x1, y1 - 30), text, fill="white", font=font)
        
        # Generate natural speech
        if not detected_objects:
            speech = (
                "I'm not able to clearly identify any objects in front of the camera right now. "
                "This could be due to lighting conditions or camera angle. "
                "Please try adjusting your position and say detect again."
            )
            status = "🔍 No objects detected"
        else:
            counts = Counter(detected_objects)
            parts = []
            for obj, count in counts.items():
                if count > 1:
                    parts.append(f"{count} {obj}s")
                else:
                    parts.append(f"{'an' if obj[0].lower() in 'aeiou' else 'a'} {obj}")
            
            if len(parts) == 1:
                items_text = parts[0]
            elif len(parts) == 2:
                items_text = f"{parts[0]} and {parts[1]}"
            else:
                items_text = f"{', '.join(parts[:-1])}, and {parts[-1]}"
            
            speech = f"I can see {items_text} in front of you."
            status = f"✅ Found: {', '.join(set(detected_objects))}"
        
        speech += " ... NoonVision ready. Say detect to scan again."
        audio_path = make_audio_file(speech)
        
        elapsed = time.time() - t0
        print(f"⏱️ Processing time: {elapsed:.2f}s")
        print(f"{'='*50}\n")
        
        return pil_image, audio_path, status
        
    except Exception as e:
        print(f"❌ Detection error: {e}")
        import traceback
        traceback.print_exc()
        audio = make_audio_file("I encountered an error while processing. Please try again.")
        return None, audio, f"❌ Error: {str(e)}"

# =============================================================================
# ENHANCED UI STYLES
# =============================================================================

CUSTOM_CSS = """
/* Global Styles */
* {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}

/* Main Container */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* Hide default footer */
footer { display: none !important; }

/* Header Styles */
.header-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    border-radius: 20px;
    padding: 40px;
    margin-bottom: 25px;
    box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
    position: relative;
    overflow: hidden;
}

.header-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    animation: shimmer 15s infinite linear;
}

@keyframes shimmer {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.header-title {
    font-size: 3.5em;
    font-weight: 800;
    color: white;
    margin: 0;
    text-shadow: 2px 4px 20px rgba(0,0,0,0.3);
    position: relative;
    z-index: 1;
}

.header-subtitle {
    font-size: 1.4em;
    color: rgba(255,255,255,0.95);
    margin: 10px 0 0 0;
    font-weight: 400;
    position: relative;
    z-index: 1;
}

.header-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    padding: 8px 16px;
    border-radius: 30px;
    font-size: 0.85em;
    color: white;
    margin-top: 15px;
    backdrop-filter: blur(10px);
    position: relative;
    z-index: 1;
}

/* Instructions Card */
.instructions-card {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 2px solid #22c55e;
    border-radius: 16px;
    padding: 25px;
    margin-bottom: 25px;
    box-shadow: 0 4px 20px rgba(34, 197, 94, 0.15);
}

.instructions-title {
    font-size: 1.3em;
    font-weight: 700;
    color: #166534;
    margin: 0 0 15px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.instructions-steps {
    display: flex;
    justify-content: space-around;
    flex-wrap: wrap;
    gap: 15px;
}

.step-item {
    display: flex;
    align-items: center;
    gap: 10px;
    background: white;
    padding: 12px 20px;
    border-radius: 12px;
    font-weight: 500;
    color: #166534;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    transition: transform 0.2s, box-shadow 0.2s;
}

.step-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.step-number {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: white;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.9em;
}

/* Status Box */
.status-box {
    padding: 20px 25px;
    border-radius: 16px;
    margin: 15px 0;
    font-size: 1.2em;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.status-listening {
    background: linear-gradient(135deg, rgba(34,197,94,0.15) 0%, rgba(34,197,94,0.05) 100%);
    border-left: 5px solid #22c55e;
    animation: pulse-green 2s infinite;
}

.status-processing {
    background: linear-gradient(135deg, rgba(59,130,246,0.15) 0%, rgba(59,130,246,0.05) 100%);
    border-left: 5px solid #3b82f6;
    animation: pulse-blue 1s infinite;
}

.status-error {
    background: linear-gradient(135deg, rgba(239,68,68,0.15) 0%, rgba(239,68,68,0.05) 100%);
    border-left: 5px solid #ef4444;
}

@keyframes pulse-green {
    0%, 100% { box-shadow: 0 4px 15px rgba(34,197,94,0.2); }
    50% { box-shadow: 0 4px 25px rgba(34,197,94,0.4); }
}

@keyframes pulse-blue {
    0%, 100% { box-shadow: 0 4px 15px rgba(59,130,246,0.2); }
    50% { box-shadow: 0 4px 25px rgba(59,130,246,0.4); }
}

/* Voice Feedback */
.voice-feedback {
    background: linear-gradient(135deg, #fefce8 0%, #fef9c3 100%);
    padding: 15px 20px;
    border-radius: 12px;
    text-align: center;
    border-left: 5px solid #eab308;
    font-size: 1.1em;
    color: #854d0e;
    margin: 15px 0;
    box-shadow: 0 2px 10px rgba(234,179,8,0.15);
}

/* Camera Container */
.camera-section {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
}

.camera-title {
    color: #94a3b8;
    font-size: 0.95em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.camera-title .live-dot {
    width: 10px;
    height: 10px;
    background: #ef4444;
    border-radius: 50%;
    animation: blink 1s infinite;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

#camera-container video {
    width: 100%;
    border-radius: 12px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.3);
}

/* Results Section */
.results-section {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    border: 1px solid #e2e8f0;
}

.results-title {
    color: #475569;
    font-size: 0.95em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 15px;
}

/* Features Grid */
.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin: 30px 0;
}

.feature-card {
    background: white;
    border-radius: 16px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    border: 1px solid #e5e7eb;
    transition: transform 0.2s, box-shadow 0.2s;
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}

.feature-icon {
    font-size: 2.5em;
    margin-bottom: 15px;
}

.feature-title {
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 8px;
}

.feature-desc {
    color: #6b7280;
    font-size: 0.9em;
}

/* Footer */
.footer-container {
    background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
    border-radius: 20px;
    padding: 30px;
    margin-top: 30px;
    color: white;
    text-align: center;
}

.footer-links {
    display: flex;
    justify-content: center;
    gap: 30px;
    margin-bottom: 20px;
}

.footer-link {
    color: #9ca3af;
    text-decoration: none;
    font-size: 0.95em;
    transition: color 0.2s;
}

.footer-link:hover {
    color: white;
}

.footer-credit {
    color: #6b7280;
    font-size: 0.85em;
}

/* Voice Commands Section */
.commands-section {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border-radius: 16px;
    padding: 20px;
    margin: 20px 0;
    border: 1px solid #93c5fd;
}

.commands-title {
    font-weight: 700;
    color: #1e40af;
    margin-bottom: 15px;
    font-size: 1.1em;
}

.commands-list {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.command-chip {
    background: white;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 0.95em;
    color: #1e40af;
    font-weight: 500;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}

/* Responsive */
@media (max-width: 768px) {
    .header-title { font-size: 2.2em; }
    .header-subtitle { font-size: 1.1em; }
    .instructions-steps { flex-direction: column; }
    .features-grid { grid-template-columns: 1fr; }
}

/* Hide unnecessary elements */
.hidden-row { display: none !important; }
"""

# =============================================================================
# BUILD GRADIO INTERFACE
# =============================================================================

with gr.Blocks(
    title="NoonVision - Hands-Free AI Vision Assistant",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft()
) as demo:
    
    # =========================================================================
    # HEADER
    # =========================================================================
    gr.HTML("""
    <div class="header-container">
        <h1 class="header-title">🦾 NoonVision</h1>
        <p class="header-subtitle">Hands-Free AI Vision Assistant for the Visually Impaired</p>
        <span class="header-badge">✨ 100% Voice Controlled • No Buttons Required</span>
    </div>
    """)
    
    # =========================================================================
    # INSTRUCTIONS
    # =========================================================================
    gr.HTML("""
    <div class="instructions-card">
        <h3 class="instructions-title">🎯 How It Works</h3>
        <div class="instructions-steps">
            <div class="step-item">
                <span class="step-number">1</span>
                <span>Click anywhere to start</span>
            </div>
            <div class="step-item">
                <span class="step-number">2</span>
                <span>Allow camera & microphone</span>
            </div>
            <div class="step-item">
                <span class="step-number">3</span>
                <span>Say "Detect"</span>
            </div>
            <div class="step-item">
                <span class="step-number">4</span>
                <span>Listen to results</span>
            </div>
        </div>
    </div>
    """)
    
    # =========================================================================
    # STATUS & VOICE FEEDBACK
    # =========================================================================
    gr.HTML("""
    <div id="status-box" class="status-box status-listening">
        🎤 <strong>Click anywhere on the page to start</strong> — Camera and voice recognition will initialize
    </div>
    <div id="voice-feedback" class="voice-feedback">
        🗣️ Waiting for voice command...
    </div>
    """)
    
    # =========================================================================
    # MAIN CONTENT - CAMERA & RESULTS
    # =========================================================================
    with gr.Row():
        # Camera Column
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="camera-section">
                <div class="camera-title">
                    <span class="live-dot"></span>
                    <span>LIVE CAMERA FEED</span>
                </div>
                <div id="camera-container">
                    <video id="webcam" autoplay playsinline muted></video>
                    <canvas id="canvas" style="display:none"></canvas>
                </div>
            </div>
            """)
        
        # Results Column
        with gr.Column(scale=1):
            gr.HTML('<div class="results-section"><div class="results-title">🎯 DETECTION RESULTS</div>')
            result_image = gr.Image(
                label="", 
                type="pil", 
                elem_id="result-image",
                show_label=False
            )
            result_status = gr.Textbox(
                label="Status", 
                value="Ready for detection", 
                interactive=False,
                elem_id="result-status"
            )
            result_audio = gr.Audio(
                label="Audio Response", 
                type="filepath", 
                autoplay=True, 
                elem_id="result-audio"
            )
            gr.HTML('</div>')
    
    # =========================================================================
    # VOICE COMMANDS
    # =========================================================================
    gr.HTML("""
    <div class="commands-section">
        <div class="commands-title">🗣️ Voice Commands You Can Use</div>
        <div class="commands-list">
            <span class="command-chip">"Detect"</span>
            <span class="command-chip">"What do you see?"</span>
            <span class="command-chip">"Scan"</span>
            <span class="command-chip">"Look"</span>
            <span class="command-chip">"Identify"</span>
            <span class="command-chip">"Check"</span>
        </div>
    </div>
    """)
    
    # =========================================================================
    # FEATURES GRID
    # =========================================================================
    gr.HTML("""
    <div class="features-grid">
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">80+ Objects</div>
            <div class="feature-desc">Detects people, vehicles, animals, furniture, and everyday items</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Real-Time</div>
            <div class="feature-desc">Fast detection with results in 1-2 seconds</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔊</div>
            <div class="feature-title">Audio Feedback</div>
            <div class="feature-desc">Natural speech describes what's in front of you</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">♿</div>
            <div class="feature-title">Accessible</div>
            <div class="feature-desc">Designed specifically for blind and visually impaired users</div>
        </div>
    </div>
    """)
    
    # =========================================================================
    # FOOTER
    # =========================================================================
    gr.HTML(f"""
    <div class="footer-container">
        <div class="footer-links">
            <span class="footer-link">Built with YOLOv8 + Gradio</span>
            <span class="footer-link">•</span>
            <span class="footer-link">Chrome/Edge Recommended</span>
            <span class="footer-link">•</span>
            <span class="footer-link">Version {VERSION}</span>
        </div>
        <p class="footer-credit">
            Made with ❤️ for Accessibility | NoonVision © 2025
        </p>
    </div>
    """)
    
    # =========================================================================
    # HIDDEN COMPONENTS
    # =========================================================================
    with gr.Row(elem_classes="hidden-row"):
        hidden_input = gr.Textbox(elem_id="hidden-input", label="hidden")
        hidden_btn = gr.Button("Detect", elem_id="hidden-btn")
    
    # Connect detection
    hidden_btn.click(
        fn=detect_objects,
        inputs=[hidden_input],
        outputs=[result_image, result_audio, result_status]
    )
    
    # Audio elements
    gr.HTML(f'''
    <audio id="audio-startup" src="data:audio/mp3;base64,{STARTUP_AUDIO_B64}"></audio>
    <audio id="audio-processing" src="data:audio/mp3;base64,{PROCESSING_AUDIO_B64}"></audio>
    ''')
    
    # =========================================================================
    # JAVASCRIPT
    # =========================================================================
    demo.load(
        fn=None,
        inputs=None,
        outputs=None,
        js="""
        function() {
            console.log('🦾 NoonVision: Initializing...');
            
            // State variables
            let videoStream = null;
            let recognition = null;
            let isProcessing = false;
            let isListening = false;
            let isInitialized = false;
            
            const TRIGGERS = ['detect', 'what do you see', 'identify', 'scan', 'look', 'check'];
            
            // DOM elements
            const video = document.getElementById('webcam');
            const canvas = document.getElementById('canvas');
            const statusBox = document.getElementById('status-box');
            const voiceFeedback = document.getElementById('voice-feedback');
            const startupAudio = document.getElementById('audio-startup');
            const processingAudio = document.getElementById('audio-processing');
            
            // Update status
            function setStatus(text, type) {
                if (statusBox) {
                    statusBox.innerHTML = text;
                    statusBox.className = 'status-box status-' + type;
                }
            }
            
            // Update voice feedback
            function setVoice(text) {
                if (voiceFeedback) {
                    voiceFeedback.innerHTML = text;
                }
            }
            
            // Check triggers
            function hasTrigger(text) {
                return TRIGGERS.some(t => text.toLowerCase().includes(t));
            }
            
            // Start camera
            async function startCamera() {
                console.log('📷 Starting camera...');
                try {
                    videoStream = await navigator.mediaDevices.getUserMedia({
                        video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
                        audio: false
                    });
                    
                    video.srcObject = videoStream;
                    await video.play();
                    
                    canvas.width = video.videoWidth || 640;
                    canvas.height = video.videoHeight || 480;
                    
                    console.log('✅ Camera ready:', canvas.width + 'x' + canvas.height);
                    return true;
                } catch (err) {
                    console.error('❌ Camera error:', err);
                    setStatus('⚠️ <strong>Camera Error:</strong> ' + err.message, 'error');
                    return false;
                }
            }
            
            // Capture frame
            function captureFrame() {
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                return canvas.toDataURL('image/jpeg', 0.85);
            }
            
            // Initialize speech
            function initSpeech() {
                console.log('🎤 Initializing speech...');
                
                if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                    setStatus('⚠️ <strong>Voice not supported.</strong> Please use Chrome or Edge.', 'error');
                    return false;
                }
                
                const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
                recognition = new SR();
                recognition.continuous = true;
                recognition.interimResults = false;
                recognition.lang = 'en-US';
                
                recognition.onstart = () => {
                    isListening = true;
                    console.log('🎤 Listening...');
                    if (!isProcessing) {
                        setStatus('🎤 <strong>Listening...</strong> Say "Detect" to scan your surroundings', 'listening');
                    }
                };
                
                recognition.onresult = (event) => {
                    if (isProcessing) return;
                    
                    const text = event.results[event.results.length - 1][0].transcript;
                    console.log('🗣️ Heard:', text);
                    setVoice('🗣️ Heard: "' + text + '"');
                    
                    if (hasTrigger(text)) {
                        console.log('🎯 TRIGGER!');
                        doDetection();
                    }
                };
                
                recognition.onerror = (e) => {
                    console.log('Speech error:', e.error);
                    isListening = false;
                    if (e.error !== 'no-speech' && e.error !== 'aborted') {
                        setTimeout(startListening, 1000);
                    }
                };
                
                recognition.onend = () => {
                    isListening = false;
                    if (!isProcessing) setTimeout(startListening, 300);
                };
                
                return true;
            }
            
            function startListening() {
                if (!recognition || isListening || isProcessing) return;
                try { recognition.start(); } catch (e) {}
            }
            
            function stopListening() {
                if (recognition && isListening) {
                    try { recognition.stop(); } catch (e) {}
                }
            }
            
            // Main detection
            async function doDetection() {
                if (isProcessing) return;
                
                isProcessing = true;
                stopListening();
                
                console.log('🔍 Starting detection...');
                
                setStatus('📸 <strong>Capturing image...</strong>', 'processing');
                setVoice('📸 Capturing...');
                
                if (processingAudio) {
                    processingAudio.currentTime = 0;
                    processingAudio.play().catch(() => {});
                }
                
                await new Promise(r => setTimeout(r, 300));
                
                const imageData = captureFrame();
                console.log('📷 Frame captured');
                
                setStatus('🔍 <strong>Analyzing image with AI...</strong>', 'processing');
                setVoice('🔍 Analyzing...');
                
                const hiddenInput = document.querySelector('#hidden-input textarea');
                const hiddenBtn = document.querySelector('#hidden-btn');
                
                if (hiddenInput && hiddenBtn) {
                    hiddenInput.value = imageData;
                    hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
                    
                    await new Promise(r => setTimeout(r, 100));
                    hiddenBtn.click();
                    
                    setTimeout(() => {
                        if (isProcessing) resumeListening();
                    }, 15000);
                } else {
                    console.error('Hidden elements not found');
                    setStatus('⚠️ Error: UI elements not found', 'error');
                    isProcessing = false;
                    setTimeout(startListening, 2000);
                }
            }
            
            function resumeListening() {
                console.log('🔄 Resuming...');
                isProcessing = false;
                setStatus('🎤 <strong>Listening...</strong> Say "Detect" to scan again', 'listening');
                setVoice('🗣️ Ready — say "Detect"');
                startListening();
            }
            
            // Monitor audio end
            function setupAudioMonitor() {
                const observer = new MutationObserver(() => {
                    document.querySelectorAll('audio').forEach(audio => {
                        if (audio.id === 'audio-startup' || audio.id === 'audio-processing') return;
                        if (audio.dataset.monitored) return;
                        
                        audio.dataset.monitored = 'true';
                        audio.addEventListener('ended', () => {
                            console.log('🔊 Audio ended');
                            if (isProcessing) resumeListening();
                        });
                    });
                });
                
                observer.observe(document.body, { childList: true, subtree: true });
            }
            
            // Initialize
            async function init() {
                if (isInitialized) return;
                isInitialized = true;
                
                console.log('🚀 Initializing NoonVision...');
                
                setStatus('⏳ <strong>Starting camera...</strong>', 'processing');
                const camOk = await startCamera();
                if (!camOk) return;
                
                setStatus('⏳ <strong>Setting up voice recognition...</strong>', 'processing');
                const speechOk = initSpeech();
                if (!speechOk) return;
                
                setupAudioMonitor();
                
                if (startupAudio) {
                    try {
                        await startupAudio.play();
                        startupAudio.onended = () => {
                            setStatus('🎤 <strong>Listening...</strong> Say "Detect" to scan your surroundings', 'listening');
                            setVoice('🗣️ Ready — say "Detect"');
                            startListening();
                        };
                    } catch (e) {
                        setStatus('🎤 <strong>Listening...</strong> Say "Detect" to scan your surroundings', 'listening');
                        setVoice('🗣️ Ready — say "Detect"');
                        startListening();
                    }
                }
                
                console.log('✅ NoonVision ready!');
            }
            
            // Expose globally
            window.noonvision = {
                init, doDetection, startListening, resumeListening,
                status: () => ({ isProcessing, isListening, isInitialized })
            };
            
            // Start on click
            document.addEventListener('click', function firstClick() {
                document.removeEventListener('click', firstClick);
                init();
            }, { once: true });
            
            // Auto-init fallback
            setTimeout(() => {
                if (!isInitialized) init();
            }, 3000);
            
            console.log('💡 NoonVision loaded. Click to start.');
        }
        """
    )

print("🚀 NoonVision ready to launch!")
demo.launch()