import gradio as gr
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import time
from collections import Counter
import os
import base64
import io

# Configuration
CONF_THRESHOLD = 0.30
IMG_SIZE = 640
BOX_COLOR = (0, 255, 0)
BOX_WIDTH = 4
FONT_SIZE = 18

print("🚀 Loading NoonVision...")

# YOLOv8
try:
    model = YOLO("yolov8m.pt")
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    _ = model(dummy, verbose=False)
    print("✅ YOLOv8m loaded successfully")
except Exception as e:
    print(f"❌ YOLO failed: {e}")
    model = None

def get_audio_base64(filepath):
    """Convert audio file to base64"""
    try:
        if filepath and os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except Exception as e:
        print(f"⚠️ Base64 conversion error: {e}")
    return ""

def generate_startup_audio():
    """Generate startup announcement"""
    try:
        text = "NoonVision ready. Say detect to identify objects around you."
        filename = "startup_audio.mp3"
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filename)
        return filename
    except Exception as e:
        print(f"⚠️ Startup audio error: {e}")
        return None

def generate_processing_audio():
    """Generate processing indicator"""
    try:
        text = "Processing."
        filename = "processing_audio.mp3"
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filename)
        return filename
    except Exception as e:
        print(f"⚠️ Processing audio error: {e}")
        return None

def generate_result_audio(labels):
    """Generate speech for detected objects"""
    try:
        if not labels:
            text = "I don't see any recognizable objects at the moment. Try pointing the camera at something else, or move a bit closer."
        else:
            counts = Counter(labels)
            if len(counts) == 1:
                obj, count = list(counts.items())[0]
                if count == 1:
                    text = f"I can see a {obj} in front of you."
                else:
                    text = f"I can see {count} {obj}s in front of you."
            else:
                items = []
                for obj, count in counts.items():
                    if count == 1:
                        items.append(f"a {obj}")
                    else:
                        items.append(f"{count} {obj}s")
                
                if len(items) == 2:
                    text = f"I can see {items[0]} and {items[1]}."
                else:
                    text = f"I can see {', '.join(items[:-1])}, and {items[-1]}."
        
        text += " ... Listening. Say detect when ready."
        
        filename = f"result_{int(time.time()*1000)}.mp3"
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filename)
        return filename
    except Exception as e:
        print(f"⚠️ Audio error: {e}")
        return None

def generate_error_audio(text):
    """Generate error message audio"""
    try:
        text += " ... Listening. Say detect when ready."
        filename = f"error_{int(time.time()*1000)}.mp3"
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filename)
        return filename
    except:
        return None

def detect_from_base64(image_base64):
    """Detect objects from base64 encoded image"""
    print(f"[DEBUG] Received image data, length: {len(image_base64) if image_base64 else 0}")
    
    if not image_base64 or len(image_base64) < 100:
        error_text = "I cannot see anything right now. Please make sure the camera is working."
        error_audio = generate_error_audio(error_text)
        return None, error_audio, "⚠️ No image received from camera"
    
    if model is None:
        error_text = "The detection system is not ready yet. Please wait a moment."
        error_audio = generate_error_audio(error_text)
        return None, error_audio, "⚠️ Model not loaded"
    
    try:
        start = time.time()
        
        # Decode base64 image
        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        img_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_np = np.array(img_pil)
        
        print(f"[DEBUG] Image decoded, shape: {img_np.shape}")
        
        # Run YOLO detection
        results = model(img_np, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        labels = results.names
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        
        # Draw bounding boxes
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
        except:
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
        audio = generate_result_audio(detected_labels)
        
        elapsed = time.time() - start
        
        if detected_labels:
            status = f"✅ Found {len(detected_labels)} object(s) in {elapsed:.2f}s: {', '.join(set(detected_labels))}"
        else:
            status = f"🔍 No objects detected ({elapsed:.2f}s)"
        
        print(f"[DEBUG] Detection complete: {status}")
        return img_pil, audio, status
        
    except Exception as e:
        print(f"[DEBUG] Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        error_text = "Sorry, something went wrong. Please try again."
        error_audio = generate_error_audio(error_text)
        return None, error_audio, f"❌ Error: {str(e)}"

# Pre-generate audio files
print("🔊 Generating startup audio files...")
startup_audio_file = generate_startup_audio()
processing_audio_file = generate_processing_audio()
startup_audio_base64 = get_audio_base64(startup_audio_file)
processing_audio_base64 = get_audio_base64(processing_audio_file)
print("✅ Audio files ready")

# Custom CSS
CUSTOM_CSS = """
#video-container {
    position: relative;
    width: 100%;
    max-width: 640px;
    margin: 0 auto;
    border-radius: 12px;
    overflow: hidden;
    border: 3px solid #667eea;
}

#webcam-video {
    width: 100%;
    height: auto;
    display: block;
    transform: scaleX(-1);
}

#hidden-canvas {
    display: none;
}

.listening-active {
    background: linear-gradient(90deg, #22c55e20, #22c55e10);
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #22c55e;
    animation: pulse 2s infinite;
    margin: 10px 0;
}

.listening-paused {
    background: #f59e0b20;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #f59e0b;
    margin: 10px 0;
}

.processing-active {
    background: #3b82f620;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #3b82f6;
    margin: 10px 0;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

#status-display {
    font-size: 1.2em;
    padding: 15px;
    background: #f0f9ff;
    border-radius: 10px;
    text-align: center;
    margin: 10px 0;
}

#heard-text {
    font-size: 1em;
    padding: 10px;
    background: #fefce8;
    border-radius: 8px;
    text-align: center;
    margin: 5px 0;
    min-height: 40px;
}

.browser-warning {
    background: #fef2f2;
    border: 1px solid #ef4444;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    display: none;
}

.browser-warning.show {
    display: block;
}

.camera-status {
    position: absolute;
    top: 10px;
    left: 10px;
    background: rgba(34, 197, 94, 0.9);
    color: white;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: bold;
}

.camera-status.error {
    background: rgba(239, 68, 68, 0.9);
}
"""

# Build the interface
with gr.Blocks(
    title="NoonVision - Hands-Free AI Vision Assistant",
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS
) as demo:
    
    # Header
    gr.HTML('''
    <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 12px; margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 2.5em;">🦾 NoonVision</h1>
        <h2 style="margin: 10px 0; font-weight: normal;">Hands-Free AI Vision Assistant</h2>
        <p style="margin: 5px 0; opacity: 0.9;">✨ Completely Voice-Activated • No Buttons Required</p>
    </div>
    ''')
    
    # Browser warning
    gr.HTML('''
    <div id="browser-warning" class="browser-warning">
        <strong>⚠️ Browser Compatibility:</strong> Voice recognition works best in <strong>Google Chrome</strong> or <strong>Microsoft Edge</strong>.
    </div>
    ''')
    
    # Instructions
    gr.HTML('''
    <div style="background: #ecfdf5; padding: 20px; border-radius: 12px; margin-bottom: 20px; border: 2px solid #22c55e;">
        <h3 style="margin-top: 0; color: #166534;">🎤 How to Use (Completely Hands-Free):</h3>
        <ol style="font-size: 1.1em; line-height: 1.8;">
            <li><strong>Allow permissions</strong> when prompted for camera and microphone</li>
            <li><strong>Say "Detect"</strong> or "What do you see?" at any time</li>
            <li><strong>Listen</strong> to the audio description of detected objects</li>
            <li><strong>Repeat</strong> - Camera stays open, just say detect again!</li>
        </ol>
        <p style="margin-bottom: 0; color: #166534;"><strong>💡 Tips:</strong> Speak clearly • Good lighting helps • Hold objects 2-6 feet from camera</p>
    </div>
    ''')
    
    # Status indicators
    gr.HTML('<div id="listening-indicator" class="listening-active">🎤 <span style="color: #22c55e; font-weight: bold;">Initializing...</span></div>')
    gr.HTML('<div id="status-display">🚀 Starting NoonVision...</div>')
    gr.HTML('<div id="heard-text">🗣️ Waiting for voice command...</div>')
    
    # Main content
    with gr.Row():
        with gr.Column(scale=1):
            # Custom HTML video element for live camera
            gr.HTML('''
            <div id="video-container">
                <div id="camera-status" class="camera-status">📷 Camera Loading...</div>
                <video id="webcam-video" autoplay playsinline muted></video>
                <canvas id="hidden-canvas"></canvas>
            </div>
            ''')
        
        with gr.Column(scale=1):
            result_img = gr.Image(
                type="pil",
                label="🎯 Detection Results",
                elem_id="result-image"
            )
            
            status_text = gr.Textbox(
                label="Detection Status",
                value="Ready - Say 'Detect' to identify objects",
                lines=2,
                interactive=False,
                elem_id="status-textbox"
            )
            
            audio_out = gr.Audio(
                type="filepath",
                label="🔊 Audio Result",
                autoplay=True,
                elem_id="audio-output"
            )
    
    # Hidden components for API communication
    image_input = gr.Textbox(visible=False, elem_id="image-input")
    detect_btn = gr.Button("Detect", visible=False, elem_id="detect-btn")
    
    # Audio elements
    gr.HTML(f'''
    <audio id="startup-audio" preload="auto">
        <source src="data:audio/mp3;base64,{startup_audio_base64}" type="audio/mp3">
    </audio>
    <audio id="processing-audio" preload="auto">
        <source src="data:audio/mp3;base64,{processing_audio_base64}" type="audio/mp3">
    </audio>
    ''')
    
    # Main JavaScript
    gr.HTML('''
    <script>
    (function() {
        // State
        let recognition = null;
        let isProcessing = false;
        let isListening = false;
        let hasInteracted = false;
        let videoStream = null;
        let video = null;
        let canvas = null;
        let ctx = null;
        
        const TRIGGER_PHRASES = ["detect", "what do you see", "what's in front", "identify", "scan", "look", "what is in front", "what's this", "what is this"];
        
        function containsTrigger(text) {
            const lowerText = text.toLowerCase();
            return TRIGGER_PHRASES.some(phrase => lowerText.includes(phrase));
        }
        
        function updateUI(state, message) {
            const indicator = document.getElementById('listening-indicator');
            const status = document.getElementById('status-display');
            
            if (indicator) {
                if (state === 'listening') {
                    indicator.className = 'listening-active';
                    indicator.innerHTML = '🎤 <span style="color: #22c55e; font-weight: bold;">Listening...</span> Say "Detect" or "What do you see?"';
                } else if (state === 'processing') {
                    indicator.className = 'processing-active';
                    indicator.innerHTML = '🔍 <span style="color: #3b82f6; font-weight: bold;">Processing...</span> Please wait';
                } else if (state === 'paused') {
                    indicator.className = 'listening-paused';
                    indicator.innerHTML = '⏸️ <span style="color: #f59e0b;">Paused</span> - ' + message;
                } else if (state === 'error') {
                    indicator.className = 'listening-paused';
                    indicator.innerHTML = '⚠️ <span style="color: #ef4444;">' + message + '</span>';
                }
            }
            
            if (status && message) {
                status.innerHTML = message;
            }
        }
        
        function updateCameraStatus(text, isError) {
            const statusEl = document.getElementById('camera-status');
            if (statusEl) {
                statusEl.textContent = text;
                statusEl.className = isError ? 'camera-status error' : 'camera-status';
            }
        }
        
        // Initialize webcam
        async function initCamera() {
            video = document.getElementById('webcam-video');
            canvas = document.getElementById('hidden-canvas');
            
            if (!video || !canvas) {
                console.error('Video or canvas element not found');
                setTimeout(initCamera, 500);
                return;
            }
            
            ctx = canvas.getContext('2d');
            
            try {
                videoStream = await navigator.mediaDevices.getUserMedia({
                    video: { 
                        facingMode: 'user',
                        width: { ideal: 640 },
                        height: { ideal: 480 }
                    },
                    audio: false
                });
                
                video.srcObject = videoStream;
                video.onloadedmetadata = () => {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    updateCameraStatus('📷 Camera Active', false);
                    console.log('📷 Camera initialized:', video.videoWidth, 'x', video.videoHeight);
                };
                
            } catch (err) {
                console.error('Camera error:', err);
                updateCameraStatus('❌ Camera Error', true);
                updateUI('error', 'Camera access denied. Please allow camera permission and refresh.');
            }
        }
        
        // Capture current frame
        function captureFrame() {
            if (!video || !canvas || !ctx) {
                console.error('Video/canvas not ready');
                return null;
            }
            
            if (video.readyState !== video.HAVE_ENOUGH_DATA) {
                console.error('Video not ready');
                return null;
            }
            
            // Draw current frame to canvas (flip horizontally to match mirrored video)
            ctx.save();
            ctx.scale(-1, 1);
            ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
            ctx.restore();
            
            // Get base64 image
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            console.log('📸 Frame captured, size:', imageData.length);
            return imageData;
        }
        
        // Speech recognition
        function initSpeechRecognition() {
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                console.error('Speech recognition not supported');
                document.getElementById('browser-warning').classList.add('show');
                updateUI('error', 'Speech recognition not supported. Please use Chrome or Edge.');
                return false;
            }
            
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = false;
            recognition.lang = 'en-US';
            
            recognition.onstart = function() {
                isListening = true;
                if (!isProcessing) {
                    updateUI('listening', '🎤 Listening for voice commands...');
                }
            };
            
            recognition.onresult = function(event) {
                if (isProcessing) return;
                
                const last = event.results.length - 1;
                const text = event.results[last][0].transcript;
                
                console.log('Heard:', text);
                document.getElementById('heard-text').innerHTML = '🗣️ Heard: "' + text + '"';
                
                if (containsTrigger(text)) {
                    console.log('🎯 Trigger detected!');
                    triggerDetection();
                }
            };
            
            recognition.onerror = function(event) {
                console.error('Speech error:', event.error);
                if (event.error === 'not-allowed') {
                    updateUI('error', 'Microphone denied. Please allow and refresh.');
                } else if (event.error !== 'no-speech') {
                    setTimeout(startListening, 1000);
                }
            };
            
            recognition.onend = function() {
                isListening = false;
                if (!isProcessing) {
                    setTimeout(startListening, 300);
                }
            };
            
            return true;
        }
        
        function startListening() {
            if (!recognition && !initSpeechRecognition()) return;
            
            if (!isListening && !isProcessing) {
                try {
                    recognition.start();
                } catch (e) {}
            }
        }
        
        function stopListening() {
            if (recognition && isListening) {
                try { recognition.stop(); } catch (e) {}
            }
        }
        
        // Trigger detection
        function triggerDetection() {
            if (isProcessing) return;
            
            isProcessing = true;
            updateUI('processing', '🔍 Capturing and analyzing image...');
            stopListening();
            
            // Play processing sound
            const processingAudio = document.getElementById('processing-audio');
            if (processingAudio) {
                processingAudio.currentTime = 0;
                processingAudio.play().catch(() => {});
            }
            
            // Capture frame
            const imageData = captureFrame();
            
            if (!imageData) {
                console.error('Failed to capture frame');
                updateUI('error', 'Failed to capture image. Please check camera.');
                isProcessing = false;
                startListening();
                return;
            }
            
            // Send to Gradio
            const imageInput = document.querySelector('#image-input textarea');
            const detectBtn = document.querySelector('#detect-btn');
            
            if (imageInput && detectBtn) {
                // Set the image data
                imageInput.value = imageData;
                imageInput.dispatchEvent(new Event('input', { bubbles: true }));
                
                // Click detect button after a short delay
                setTimeout(() => {
                    detectBtn.click();
                }, 100);
            } else {
                console.error('Could not find Gradio components');
                isProcessing = false;
                startListening();
            }
        }
        
        // Called when detection is complete
        window.detectionComplete = function() {
            console.log('✅ Detection complete');
            isProcessing = false;
            
            // Clear result image after audio finishes (optional - keep for now)
            setTimeout(() => {
                updateUI('listening', '✅ Ready for next detection!');
                startListening();
            }, 1000);
        };
        
        // Play startup and begin
        function playStartupAndBegin() {
            const startupAudio = document.getElementById('startup-audio');
            if (startupAudio) {
                startupAudio.play()
                    .then(() => {
                        startupAudio.onended = () => startListening();
                    })
                    .catch(() => startListening());
            } else {
                startListening();
            }
        }
        
        function handleFirstInteraction() {
            if (!hasInteracted) {
                hasInteracted = true;
                playStartupAndBegin();
            }
        }
        
        // Initialize
        function init() {
            console.log('🚀 NoonVision initializing...');
            
            updateUI('paused', 'Click anywhere to activate NoonVision');
            
            // Init camera
            initCamera();
            
            // Set up interaction handlers
            document.addEventListener('click', handleFirstInteraction);
            document.addEventListener('keydown', handleFirstInteraction);
            document.addEventListener('touchstart', handleFirstInteraction);
            
            // Try auto-start after delay
            setTimeout(() => {
                if (!hasInteracted) {
                    if (initSpeechRecognition()) {
                        startListening();
                        hasInteracted = true;
                        const startupAudio = document.getElementById('startup-audio');
                        if (startupAudio) startupAudio.play().catch(() => {});
                    }
                }
            }, 2000);
        }
        
        // Start when ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', init);
        } else {
            setTimeout(init, 500);
        }
    })();
    </script>
    ''')
    
    # Event handler
    detect_btn.click(
        fn=detect_from_base64,
        inputs=image_input,
        outputs=[result_img, audio_out, status_text]
    ).then(
        fn=None,
        inputs=None,
        outputs=None,
        js="() => { if(window.detectionComplete) window.detectionComplete(); }"
    )
    
    # Footer
    gr.HTML('''
    <div style="text-align: center; color: #666; padding: 20px; margin-top: 20px; border-top: 1px solid #e5e7eb;">
        <p><strong>🎯 Detects 80+ objects:</strong> People, furniture, electronics, food, animals, vehicles</p>
        <p><strong>⚡ Response time:</strong> 1-2 seconds</p>
        <p style="margin-top: 15px; font-size: 0.9em;">
            Built with YOLOv8 + Web Speech API | Chrome or Edge recommended<br>
            Made with ❤️ for accessibility
        </p>
    </div>
    ''')

if __name__ == "__main__":
    demo.launch()