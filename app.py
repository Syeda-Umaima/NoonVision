# Fix for HfFolder import error
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

print("=" * 50)
print("NoonVision Starting...")
print("=" * 50)

CONF_THRESHOLD = 0.30
IMG_SIZE = 640

# Load YOLO model
model = None
try:
    model = YOLO("yolov8m.pt")
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    _ = model(dummy, verbose=False)
    print("YOLOv8m model loaded successfully")
except Exception as e:
    print(f"Model error: {e}")

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

def detect_objects(image_data):
    """
    Main detection function
    image_data: base64 encoded image string from JavaScript
    Returns: (result_image, audio_file, status_text)
    """
    print(f"\n[DETECT] Received data length: {len(image_data) if image_data else 0}")
    
    if not image_data or len(image_data) < 100:
        print("[DETECT] No valid image data")
        audio = make_audio_file("No image captured. Please make sure camera is working and try again.")
        return None, audio, "No image captured"
    
    if model is None:
        audio = make_audio_file("Detection model not ready. Please wait and try again.")
        return None, audio, "Model not loaded"
    
    try:
        t0 = time.time()
        
        # Remove data URL prefix if present
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_data)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        np_image = np.array(pil_image)
        
        print(f"[DETECT] Image shape: {np_image.shape}")
        
        # Run YOLO detection
        results = model(np_image, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        class_names = results.names
        
        print(f"[DETECT] Found {len(boxes)} objects")
        
        # Draw bounding boxes
        draw = ImageDraw.Draw(pil_image)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        detected_objects = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = class_names[int(class_ids[i])]
            conf = confidences[i]
            detected_objects.append(label)
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=4)
            
            # Draw label background and text
            text = f"{label} {conf:.0%}"
            bbox = draw.textbbox((x1, y1 - 25), text, font=font)
            draw.rectangle(bbox, fill=(0, 255, 0))
            draw.text((x1, y1 - 25), text, fill="black", font=font)
        
        # Generate speech description
        if not detected_objects:
            speech = "I cannot see anything clearly in front of the camera. Try adjusting the camera angle or improving the lighting."
        else:
            counts = Counter(detected_objects)
            parts = []
            for obj, count in counts.items():
                if count > 1:
                    parts.append(f"{count} {obj}s")
                else:
                    parts.append(f"a {obj}")
            
            if len(parts) == 1:
                speech = f"I can see {parts[0]} in front of you."
            elif len(parts) == 2:
                speech = f"I can see {parts[0]} and {parts[1]} in front of you."
            else:
                speech = f"I can see {', '.join(parts[:-1])}, and {parts[-1]} in front of you."
        
        speech += " ... NoonVision ready. Say detect to scan again."
        
        # Create audio file
        audio_path = make_audio_file(speech)
        
        # Calculate processing time
        elapsed = time.time() - t0
        status = f"Found {len(detected_objects)} object(s) in {elapsed:.1f}s"
        if detected_objects:
            status += f": {', '.join(set(detected_objects))}"
        
        print(f"[DETECT] Complete: {status}")
        return pil_image, audio_path, status
        
    except Exception as e:
        print(f"[DETECT] Error: {e}")
        import traceback
        traceback.print_exc()
        audio = make_audio_file("An error occurred during detection. Please try again.")
        return None, audio, f"Error: {str(e)}"

# Create startup and processing audio as base64
print("Creating audio files...")

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

STARTUP_AUDIO_B64 = get_audio_base64("NoonVision is ready. Say detect to identify objects in front of you.")
PROCESSING_AUDIO_B64 = get_audio_base64("Processing. Please wait.")
print("Audio files ready")

# Build the Gradio interface
with gr.Blocks(
    title="NoonVision - Hands-Free Vision Assistant",
    head="""
    <style>
        .status-box {
            padding: 18px;
            border-radius: 12px;
            margin: 12px 0;
            font-size: 18px;
            font-weight: 500;
        }
        .listening {
            background: linear-gradient(90deg, rgba(34,197,94,0.2), rgba(34,197,94,0.1));
            border-left: 5px solid #22c55e;
            animation: pulse 2s infinite;
        }
        .processing {
            background: linear-gradient(90deg, rgba(59,130,246,0.2), rgba(59,130,246,0.1));
            border-left: 5px solid #3b82f6;
        }
        .error {
            background: linear-gradient(90deg, rgba(239,68,68,0.2), rgba(239,68,68,0.1));
            border-left: 5px solid #ef4444;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        #voice-feedback {
            background: #fef9c3;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
            border-left: 5px solid #eab308;
            margin: 8px 0;
            font-size: 16px;
        }
        #camera-container {
            background: #1a1a1a;
            border-radius: 12px;
            padding: 10px;
            text-align: center;
        }
        #camera-container video {
            width: 100%;
            max-width: 640px;
            border-radius: 8px;
        }
        .hidden-row {
            display: none !important;
        }
    </style>
    """
) as demo:
    
    # Header
    gr.HTML("""
    <div style="text-align:center;padding:25px;background:linear-gradient(135deg,#667eea,#764ba2);color:white;border-radius:15px;margin-bottom:20px">
        <h1 style="margin:0;font-size:2.2em">🦾 NoonVision</h1>
        <p style="margin:8px 0 0 0;opacity:0.9">Hands-Free AI Vision Assistant</p>
    </div>
    """)
    
    # Instructions
    gr.HTML("""
    <div style="background:#ecfdf5;padding:15px 20px;border-radius:10px;border:2px solid #22c55e;margin-bottom:20px">
        <p style="margin:0;font-size:1.1em;color:#166534">
            <b>🎤 100% Hands-Free:</b> Just say <b>"Detect"</b> — camera captures, analyzes, and speaks results automatically!
        </p>
    </div>
    """)
    
    # Status display
    status_html = gr.HTML(
        '<div id="status-box" class="status-box listening">🎤 Click anywhere on the page to start, then allow camera and microphone access.</div>'
    )
    
    # Voice feedback
    voice_html = gr.HTML(
        '<div id="voice-feedback">🗣️ Waiting for voice command...</div>'
    )
    
    # Camera and results
    with gr.Row():
        with gr.Column():
            gr.HTML('<h3 style="margin:0 0 10px 0;color:#374151">📷 Live Camera</h3>')
            gr.HTML('''
            <div id="camera-container">
                <video id="webcam" autoplay playsinline muted></video>
                <canvas id="canvas" style="display:none"></canvas>
            </div>
            ''')
        
        with gr.Column():
            gr.HTML('<h3 style="margin:0 0 10px 0;color:#374151">🎯 Detection Result</h3>')
            result_image = gr.Image(label="", type="pil", elem_id="result-image")
            result_status = gr.Textbox(label="Status", value="Ready", interactive=False)
            result_audio = gr.Audio(label="Audio", type="filepath", autoplay=True, elem_id="result-audio")
    
    # Hidden components for JavaScript communication
    with gr.Row(elem_classes="hidden-row"):
        hidden_input = gr.Textbox(elem_id="hidden-input", label="hidden")
        hidden_btn = gr.Button("Detect", elem_id="hidden-btn")
    
    # Connect hidden button to detection
    hidden_btn.click(
        fn=detect_objects,
        inputs=[hidden_input],
        outputs=[result_image, result_audio, result_status]
    )
    
    # Startup and processing audio (hidden)
    gr.HTML(f'''
    <audio id="audio-startup" src="data:audio/mp3;base64,{STARTUP_AUDIO_B64}"></audio>
    <audio id="audio-processing" src="data:audio/mp3;base64,{PROCESSING_AUDIO_B64}"></audio>
    ''')
    
    # Footer
    gr.HTML("""
    <div style="text-align:center;color:#666;padding:20px;margin-top:20px;border-top:1px solid #e5e7eb">
        <p style="margin:5px 0"><b>🎯 Detects 80+ objects</b> • <b>⚡ Fast response</b> • <b>🔊 Audio feedback</b></p>
        <p style="margin:5px 0;font-size:0.9em">Works best in Chrome or Edge</p>
    </div>
    """)
    
    # Main JavaScript - loaded via Gradio's load event
    demo.load(
        fn=None,
        inputs=None,
        outputs=None,
        js="""
        function() {
            console.log('NoonVision: Initializing...');
            
            // State
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
            
            // Update status display
            function setStatus(text, type) {
                if (statusBox) {
                    statusBox.innerHTML = text;
                    statusBox.className = 'status-box ' + type;
                }
            }
            
            // Update voice feedback
            function setVoice(text) {
                if (voiceFeedback) {
                    voiceFeedback.innerHTML = text;
                }
            }
            
            // Check for trigger words
            function hasTrigger(text) {
                const lower = text.toLowerCase();
                return TRIGGERS.some(t => lower.includes(t));
            }
            
            // Start camera
            async function startCamera() {
                console.log('NoonVision: Starting camera...');
                try {
                    videoStream = await navigator.mediaDevices.getUserMedia({
                        video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
                        audio: false
                    });
                    
                    video.srcObject = videoStream;
                    await video.play();
                    
                    canvas.width = video.videoWidth || 640;
                    canvas.height = video.videoHeight || 480;
                    
                    console.log('NoonVision: Camera started', canvas.width + 'x' + canvas.height);
                    return true;
                } catch (err) {
                    console.error('NoonVision: Camera error', err);
                    setStatus('⚠️ Camera Error: ' + err.message + '. Please allow camera access and refresh.', 'error');
                    return false;
                }
            }
            
            // Capture frame
            function captureFrame() {
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                return canvas.toDataURL('image/jpeg', 0.85);
            }
            
            // Initialize speech recognition
            function initSpeech() {
                console.log('NoonVision: Initializing speech recognition...');
                
                if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                    setStatus('⚠️ Speech not supported. Use Chrome or Edge.', 'error');
                    return false;
                }
                
                const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
                recognition = new SR();
                recognition.continuous = true;
                recognition.interimResults = false;
                recognition.lang = 'en-US';
                
                recognition.onstart = () => {
                    isListening = true;
                    console.log('NoonVision: Listening...');
                    if (!isProcessing) {
                        setStatus('🎤 <b>Listening...</b> Say "Detect" to scan', 'listening');
                    }
                };
                
                recognition.onresult = (event) => {
                    if (isProcessing) return;
                    
                    const text = event.results[event.results.length - 1][0].transcript;
                    console.log('NoonVision: Heard:', text);
                    setVoice('🗣️ Heard: "' + text + '"');
                    
                    if (hasTrigger(text)) {
                        console.log('NoonVision: TRIGGER DETECTED!');
                        doDetection();
                    }
                };
                
                recognition.onerror = (e) => {
                    console.log('NoonVision: Speech error', e.error);
                    isListening = false;
                    if (e.error !== 'no-speech' && e.error !== 'aborted') {
                        setTimeout(startListening, 1000);
                    }
                };
                
                recognition.onend = () => {
                    isListening = false;
                    console.log('NoonVision: Speech ended');
                    if (!isProcessing) {
                        setTimeout(startListening, 300);
                    }
                };
                
                return true;
            }
            
            // Start listening
            function startListening() {
                if (!recognition || isListening || isProcessing) return;
                try {
                    recognition.start();
                } catch (e) {}
            }
            
            // Stop listening
            function stopListening() {
                if (recognition && isListening) {
                    try { recognition.stop(); } catch (e) {}
                }
            }
            
            // Main detection function
            async function doDetection() {
                if (isProcessing) return;
                
                isProcessing = true;
                stopListening();
                
                console.log('NoonVision: === STARTING DETECTION ===');
                
                setStatus('📸 <b>Capturing...</b>', 'processing');
                setVoice('📸 Capturing image...');
                
                // Play processing sound
                if (processingAudio) {
                    processingAudio.currentTime = 0;
                    processingAudio.play().catch(() => {});
                }
                
                // Wait a moment
                await new Promise(r => setTimeout(r, 300));
                
                // Capture frame
                console.log('NoonVision: Capturing frame...');
                const imageData = captureFrame();
                console.log('NoonVision: Frame captured, size:', imageData.length);
                
                // Update status
                setStatus('🔍 <b>Analyzing...</b>', 'processing');
                setVoice('🔍 Analyzing image...');
                
                // Find hidden input and button
                const hiddenInput = document.querySelector('#hidden-input textarea');
                const hiddenBtn = document.querySelector('#hidden-btn');
                
                console.log('NoonVision: Hidden input:', hiddenInput ? 'found' : 'NOT FOUND');
                console.log('NoonVision: Hidden button:', hiddenBtn ? 'found' : 'NOT FOUND');
                
                if (hiddenInput && hiddenBtn) {
                    // Set image data
                    hiddenInput.value = imageData;
                    hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
                    
                    await new Promise(r => setTimeout(r, 100));
                    
                    // Click detect button
                    hiddenBtn.click();
                    console.log('NoonVision: Detection triggered');
                    
                    // Set timeout to resume
                    setTimeout(() => {
                        if (isProcessing) {
                            console.log('NoonVision: Timeout - resuming');
                            resumeListening();
                        }
                    }, 15000);
                } else {
                    console.error('NoonVision: Hidden elements not found!');
                    setStatus('⚠️ Error: UI elements not found', 'error');
                    isProcessing = false;
                    setTimeout(startListening, 2000);
                }
            }
            
            // Resume after detection
            function resumeListening() {
                console.log('NoonVision: Resuming...');
                isProcessing = false;
                setStatus('🎤 <b>Listening...</b> Say "Detect" to scan again', 'listening');
                setVoice('🗣️ Ready - say "Detect"');
                startListening();
            }
            
            // Monitor for audio end
            function setupAudioMonitor() {
                const observer = new MutationObserver(() => {
                    document.querySelectorAll('audio').forEach(audio => {
                        if (audio.id === 'audio-startup' || audio.id === 'audio-processing') return;
                        if (audio.dataset.monitored) return;
                        
                        audio.dataset.monitored = 'true';
                        audio.addEventListener('ended', () => {
                            console.log('NoonVision: Audio ended');
                            if (isProcessing) {
                                resumeListening();
                            }
                        });
                    });
                });
                
                observer.observe(document.body, { childList: true, subtree: true });
            }
            
            // Initialize everything
            async function init() {
                if (isInitialized) return;
                isInitialized = true;
                
                console.log('NoonVision: === INITIALIZING ===');
                
                setStatus('⏳ <b>Starting camera...</b>', 'processing');
                const camOk = await startCamera();
                if (!camOk) return;
                
                setStatus('⏳ <b>Setting up voice...</b>', 'processing');
                const speechOk = initSpeech();
                if (!speechOk) return;
                
                setupAudioMonitor();
                
                // Play startup audio
                if (startupAudio) {
                    try {
                        await startupAudio.play();
                        startupAudio.onended = () => {
                            setStatus('🎤 <b>Listening...</b> Say "Detect" to scan', 'listening');
                            setVoice('🗣️ Ready - say "Detect"');
                            startListening();
                        };
                    } catch (e) {
                        console.log('NoonVision: Startup audio blocked');
                        setStatus('🎤 <b>Listening...</b> Say "Detect" to scan', 'listening');
                        setVoice('🗣️ Ready - say "Detect"');
                        startListening();
                    }
                } else {
                    setStatus('🎤 <b>Listening...</b> Say "Detect" to scan', 'listening');
                    setVoice('🗣️ Ready - say "Detect"');
                    startListening();
                }
                
                console.log('NoonVision: === READY ===');
            }
            
            // Expose globally for debugging
            window.noonvision = {
                init: init,
                doDetection: doDetection,
                startListening: startListening,
                resumeListening: resumeListening,
                status: () => ({ isProcessing, isListening, isInitialized })
            };
            
            // Start on click (required for permissions)
            document.addEventListener('click', function firstClick() {
                document.removeEventListener('click', firstClick);
                init();
            }, { once: true });
            
            // Also try auto-init after delay
            setTimeout(() => {
                if (!isInitialized) {
                    console.log('NoonVision: Auto-init attempt');
                    init();
                }
            }, 3000);
            
            console.log('NoonVision: Script loaded. Click page to start or wait 3 seconds.');
        }
        """
    )

demo.launch()