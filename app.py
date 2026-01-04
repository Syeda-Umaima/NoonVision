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
print("🚀 NoonVision Starting...")
print("=" * 50)

CONF_THRESHOLD = 0.30
IMG_SIZE = 640

# Load YOLO model
model = None
try:
    model = YOLO("yolov8m.pt")
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    _ = model(dummy, verbose=False)
    print("✅ YOLOv8m model loaded")
except Exception as e:
    print(f"❌ Model error: {e}")

def make_audio(text):
    """Create TTS audio and return as base64"""
    try:
        fd, path = tempfile.mkstemp(suffix='.mp3')
        os.close(fd)
        gTTS(text=text, lang='en', slow=False).save(path)
        with open(path, 'rb') as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        os.remove(path)
        return audio_b64
    except Exception as e:
        print(f"Audio error: {e}")
        return ""

def detect_from_base64(image_b64):
    """
    Main detection function - receives base64 image from JavaScript
    Returns: (result_image_b64, audio_b64, status_text)
    """
    print(f"\n[DETECT] Received base64 image, length: {len(image_b64) if image_b64 else 0}")
    
    if not image_b64 or len(image_b64) < 100:
        print("[DETECT] No valid image data")
        audio = make_audio("No image captured. Please try again. Say detect.")
        return "", audio, "⚠️ No image captured"
    
    if model is None:
        audio = make_audio("Model not ready. Please wait.")
        return "", audio, "⚠️ Model not loaded"
    
    try:
        t0 = time.time()
        
        # Decode base64 image
        # Remove data URL prefix if present
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        
        image_data = base64.b64decode(image_b64)
        pil = Image.open(io.BytesIO(image_data)).convert('RGB')
        arr = np.array(pil)
        
        print(f"[DETECT] Image shape: {arr.shape}")
        
        # Run YOLO detection
        res = model(arr, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy()
        names = res.names
        
        print(f"[DETECT] Found {len(boxes)} objects")
        
        # Draw bounding boxes
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        objects = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = names[int(cls_ids[i])]
            conf = confs[i]
            objects.append(label)
            
            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=4)
            txt = f"{label} {conf:.0%}"
            tb = draw.textbbox((x1, y1-25), txt, font=font)
            draw.rectangle(tb, fill=(0, 255, 0))
            draw.text((x1, y1-25), txt, fill="black", font=font)
        
        # Generate speech
        if not objects:
            speech = "I cannot see anything clearly in front of the camera. Please try adjusting the camera angle or lighting conditions."
        else:
            counts = Counter(objects)
            parts = []
            for obj, n in counts.items():
                if n > 1:
                    parts.append(f"{n} {obj}s")
                else:
                    parts.append(f"a {obj}")
            
            if len(parts) == 1:
                speech = f"I can see {parts[0]} in front of you."
            elif len(parts) == 2:
                speech = f"I can see {parts[0]} and {parts[1]} in front of you."
            else:
                speech = f"I can see {', '.join(parts[:-1])}, and {parts[-1]} in front of you."
        
        # Add ready prompt
        speech += " ... NoonVision ready. Say detect to scan again."
        
        # Convert result image to base64
        buffer = io.BytesIO()
        pil.save(buffer, format='JPEG', quality=90)
        result_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Generate audio
        audio_b64 = make_audio(speech)
        
        dt = time.time() - t0
        status = f"✅ Found {len(objects)} object(s) in {dt:.1f}s"
        if objects:
            status += f": {', '.join(set(objects))}"
        
        print(f"[DETECT] Complete: {status}")
        return result_b64, audio_b64, status
        
    except Exception as e:
        print(f"[DETECT] Error: {e}")
        import traceback
        traceback.print_exc()
        audio = make_audio("An error occurred. Please try again. Say detect.")
        return "", audio, f"❌ Error: {str(e)}"

# Pre-generate startup audio
print("🔊 Creating startup audio...")
STARTUP_AUDIO = make_audio("NoonVision is ready. Say detect to identify objects in front of you. This is completely hands free.")
PROCESSING_AUDIO = make_audio("Processing. Please wait.")
print("✅ Audio ready")

# ============================================================
# CUSTOM HTML/CSS/JS FOR HANDS-FREE OPERATION
# ============================================================

CUSTOM_HTML = f'''
<div id="noonvision-app">
    <!-- Header -->
    <div style="text-align:center;padding:25px;background:linear-gradient(135deg,#667eea,#764ba2);color:white;border-radius:15px;margin-bottom:20px">
        <h1 style="margin:0;font-size:2.2em">🦾 NoonVision</h1>
        <p style="margin:8px 0 0 0;opacity:0.9">Hands-Free AI Vision Assistant</p>
    </div>
    
    <!-- Instructions -->
    <div style="background:#ecfdf5;padding:15px 20px;border-radius:10px;border:2px solid #22c55e;margin-bottom:20px">
        <p style="margin:0;font-size:1.1em;color:#166534">
            <b>🎤 100% Hands-Free:</b> Just say <b>"Detect"</b> — camera captures, analyzes, and speaks results automatically!
        </p>
    </div>
    
    <!-- Status Display -->
    <div id="status-display" style="padding:18px;border-radius:12px;margin-bottom:20px;font-size:18px;font-weight:500;background:linear-gradient(90deg,rgba(34,197,94,0.2),rgba(34,197,94,0.1));border-left:5px solid #22c55e">
        🎤 <b>Initializing...</b> Please allow camera and microphone access.
    </div>
    
    <!-- Voice Feedback -->
    <div id="voice-feedback" style="background:#fef9c3;padding:12px;border-radius:8px;text-align:center;border-left:5px solid #eab308;margin-bottom:20px;font-size:16px">
        🗣️ Waiting for voice command...
    </div>
    
    <!-- Live Camera -->
    <div style="background:#f3f4f6;border-radius:12px;padding:15px;text-align:center;margin-bottom:20px">
        <h3 style="margin:0 0 10px 0;color:#374151">📷 Live Camera (Always Running)</h3>
        <video id="live-video" autoplay playsinline muted style="width:100%;max-width:640px;border-radius:8px;background:#000"></video>
        <canvas id="capture-canvas" style="display:none"></canvas>
    </div>
    
    <!-- Hidden audio elements for startup/processing sounds -->
    <audio id="audio-startup" src="data:audio/mp3;base64,{STARTUP_AUDIO}"></audio>
    <audio id="audio-processing" src="data:audio/mp3;base64,{PROCESSING_AUDIO}"></audio>
</div>

<script>
(function() {{
    // ============================================================
    // STATE VARIABLES
    // ============================================================
    let videoStream = null;
    let recognition = null;
    let isProcessing = false;
    let isListening = false;
    let isInitialized = false;
    
    const TRIGGER_WORDS = ['detect', 'what do you see', 'identify', 'scan', 'look', 'check', 'what is'];
    
    // ============================================================
    // DOM ELEMENTS
    // ============================================================
    const video = document.getElementById('live-video');
    const canvas = document.getElementById('capture-canvas');
    const statusDisplay = document.getElementById('status-display');
    const voiceFeedback = document.getElementById('voice-feedback');
    const audioStartup = document.getElementById('audio-startup');
    const audioProcessing = document.getElementById('audio-processing');
    
    // ============================================================
    // UTILITY FUNCTIONS
    // ============================================================
    function setStatus(html, type) {{
        const colors = {{
            'listening': 'background:linear-gradient(90deg,rgba(34,197,94,0.2),rgba(34,197,94,0.1));border-left:5px solid #22c55e',
            'processing': 'background:linear-gradient(90deg,rgba(59,130,246,0.2),rgba(59,130,246,0.1));border-left:5px solid #3b82f6',
            'error': 'background:linear-gradient(90deg,rgba(239,68,68,0.2),rgba(239,68,68,0.1));border-left:5px solid #ef4444'
        }};
        statusDisplay.innerHTML = html;
        statusDisplay.style.cssText = `padding:18px;border-radius:12px;margin-bottom:20px;font-size:18px;font-weight:500;${{colors[type] || colors.listening}}`;
    }}
    
    function setVoiceFeedback(text) {{
        voiceFeedback.innerHTML = text;
    }}
    
    function hasTriggerWord(text) {{
        const lower = text.toLowerCase();
        return TRIGGER_WORDS.some(word => lower.includes(word));
    }}
    
    function sleep(ms) {{
        return new Promise(resolve => setTimeout(resolve, ms));
    }}
    
    // ============================================================
    // CAMERA FUNCTIONS
    // ============================================================
    async function startCamera() {{
        try {{
            console.log('📷 Starting camera...');
            
            // Try environment camera first (back camera on mobile), fall back to any camera
            try {{
                videoStream = await navigator.mediaDevices.getUserMedia({{
                    video: {{ facingMode: 'environment', width: {{ ideal: 640 }}, height: {{ ideal: 480 }} }},
                    audio: false
                }});
            }} catch {{
                videoStream = await navigator.mediaDevices.getUserMedia({{
                    video: {{ width: {{ ideal: 640 }}, height: {{ ideal: 480 }} }},
                    audio: false
                }});
            }}
            
            video.srcObject = videoStream;
            
            // Wait for video to be ready
            await new Promise((resolve) => {{
                video.onloadedmetadata = () => {{
                    video.play();
                    resolve();
                }};
            }});
            
            // Set canvas size to match video
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            
            console.log('✅ Camera started:', canvas.width, 'x', canvas.height);
            return true;
        }} catch (err) {{
            console.error('❌ Camera error:', err);
            setStatus('⚠️ <b>Camera Error:</b> ' + err.message + '. Please allow camera access and refresh.', 'error');
            return false;
        }}
    }}
    
    function captureFrame() {{
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL('image/jpeg', 0.85);
    }}
    
    // ============================================================
    // SPEECH RECOGNITION
    // ============================================================
    function initSpeechRecognition() {{
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {{
            console.error('Speech recognition not supported');
            setStatus('⚠️ <b>Voice Not Supported:</b> Please use Chrome or Edge browser', 'error');
            return false;
        }}
        
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = false;
        recognition.lang = 'en-US';
        recognition.maxAlternatives = 1;
        
        recognition.onstart = () => {{
            isListening = true;
            console.log('🎤 Speech recognition started');
            if (!isProcessing) {{
                setStatus('🎤 <b>Listening...</b> Say "Detect" to scan objects', 'listening');
            }}
        }};
        
        recognition.onresult = (event) => {{
            if (isProcessing) {{
                console.log('Ignoring speech - currently processing');
                return;
            }}
            
            const transcript = event.results[event.results.length - 1][0].transcript;
            console.log('🗣️ Heard:', transcript);
            setVoiceFeedback('🗣️ Heard: "' + transcript + '"');
            
            if (hasTriggerWord(transcript)) {{
                console.log('🎯 TRIGGER WORD DETECTED!');
                performDetection();
            }}
        }};
        
        recognition.onerror = (event) => {{
            console.log('Speech recognition error:', event.error);
            isListening = false;
            
            // Don't restart on abort or no-speech
            if (event.error === 'no-speech' || event.error === 'aborted') {{
                setTimeout(startListening, 500);
            }} else if (event.error === 'not-allowed') {{
                setStatus('⚠️ <b>Microphone Blocked:</b> Please allow microphone access', 'error');
            }} else {{
                setTimeout(startListening, 1000);
            }}
        }};
        
        recognition.onend = () => {{
            isListening = false;
            console.log('🎤 Speech recognition ended');
            if (!isProcessing) {{
                setTimeout(startListening, 300);
            }}
        }};
        
        return true;
    }}
    
    function startListening() {{
        if (!recognition) {{
            console.log('Recognition not initialized');
            return;
        }}
        if (isListening) {{
            console.log('Already listening');
            return;
        }}
        if (isProcessing) {{
            console.log('Currently processing, not starting listener');
            return;
        }}
        
        try {{
            recognition.start();
            console.log('🎤 Starting speech recognition...');
        }} catch (e) {{
            console.log('Recognition start error:', e.message);
            // May already be started
        }}
    }}
    
    function stopListening() {{
        if (recognition && isListening) {{
            try {{
                recognition.stop();
                console.log('🎤 Stopping speech recognition...');
            }} catch (e) {{
                console.log('Recognition stop error:', e.message);
            }}
        }}
        isListening = false;
    }}
    
    // ============================================================
    // MAIN DETECTION FUNCTION
    // ============================================================
    async function performDetection() {{
        if (isProcessing) {{
            console.log('Already processing, ignoring');
            return;
        }}
        
        isProcessing = true;
        stopListening();
        
        console.log('========================================');
        console.log('🚀 STARTING DETECTION');
        console.log('========================================');
        
        try {{
            // Step 1: Show processing status
            setStatus('📸 <b>Capturing image...</b>', 'processing');
            setVoiceFeedback('📸 Capturing...');
            
            // Play processing sound
            audioProcessing.currentTime = 0;
            audioProcessing.play().catch(() => {{}});
            
            await sleep(300);
            
            // Step 2: Capture frame from video
            console.log('📸 Capturing frame...');
            const imageData = captureFrame();
            console.log('✅ Frame captured, size:', imageData.length);
            
            // Step 3: Send to backend for detection
            setStatus('🔍 <b>Analyzing image...</b>', 'processing');
            setVoiceFeedback('🔍 Analyzing...');
            
            console.log('📤 Sending to backend...');
            
            // Find the hidden Gradio textbox and button
            const hiddenInput = document.querySelector('#hidden-image-input textarea');
            const hiddenButton = document.querySelector('#hidden-detect-btn');
            
            if (hiddenInput && hiddenButton) {{
                // Set the base64 image data
                hiddenInput.value = imageData;
                hiddenInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                
                await sleep(100);
                
                // Click the hidden button to trigger detection
                hiddenButton.click();
                console.log('✅ Detection triggered');
                
                // Set a fallback timeout to resume if audio listener doesn't fire
                setTimeout(() => {{
                    if (isProcessing) {{
                        console.log('⏰ Fallback timeout - resuming...');
                        resumeAfterDetection();
                    }}
                }}, 15000); // 15 second timeout
                
            }} else {{
                console.error('❌ Hidden elements not found!');
                console.log('Looking for: #hidden-image-input textarea and #hidden-detect-btn');
                console.log('All textareas:', document.querySelectorAll('textarea'));
                console.log('All buttons:', document.querySelectorAll('button'));
                throw new Error('UI elements not found');
            }}
            
        }} catch (error) {{
            console.error('Detection error:', error);
            setStatus('⚠️ <b>Error:</b> ' + error.message, 'error');
            isProcessing = false;
            setTimeout(startListening, 2000);
        }}
    }}
    
    // ============================================================
    // RESULT HANDLER - Monitor Gradio's audio component
    // ============================================================
    function setupAudioEndListener() {{
        // Find Gradio's audio element and listen for end
        const checkAudio = setInterval(() => {{
            const audioElements = document.querySelectorAll('#result-audio audio, audio[data-testid="audio"]');
            audioElements.forEach(audio => {{
                if (!audio.hasAttribute('data-noonvision-listener')) {{
                    audio.setAttribute('data-noonvision-listener', 'true');
                    audio.addEventListener('ended', () => {{
                        console.log('🔊 Result audio ended');
                        resumeAfterDetection();
                    }});
                    console.log('✅ Audio end listener attached');
                }}
            }});
        }}, 1000);
    }}
    
    window.noonvision = window.noonvision || {{}};
    window.noonvision.resumeAfterAudio = resumeAfterDetection;
    
    function resumeAfterDetection() {{
        if (!isProcessing) return; // Already resumed
        
        console.log('🔄 Resuming after detection...');
        
        isProcessing = false;
        setStatus('🎤 <b>Listening...</b> Say "Detect" to scan again', 'listening');
        setVoiceFeedback('🗣️ Ready - say "Detect"');
        startListening();
        
        console.log('========================================');
        console.log('✅ READY FOR NEXT DETECTION');
        console.log('========================================');
    }}
    
    // ============================================================
    // INITIALIZATION
    // ============================================================
    async function initialize() {{
        if (isInitialized) return;
        isInitialized = true;
        
        console.log('🚀 NoonVision initializing...');
        setStatus('⏳ <b>Initializing...</b> Starting camera...', 'processing');
        
        // Start camera
        const cameraOk = await startCamera();
        if (!cameraOk) return;
        
        // Initialize speech recognition
        setStatus('⏳ <b>Initializing...</b> Setting up voice recognition...', 'processing');
        const speechOk = initSpeechRecognition();
        if (!speechOk) return;
        
        // Setup audio end listener
        setupAudioEndListener();
        
        // Play startup audio
        setStatus('🔊 <b>Playing welcome message...</b>', 'processing');
        
        try {{
            await audioStartup.play();
            audioStartup.onended = () => {{
                setStatus('🎤 <b>Listening...</b> Say "Detect" to scan objects', 'listening');
                setVoiceFeedback('🗣️ Ready - say "Detect"');
                startListening();
            }};
        }} catch (e) {{
            console.log('Startup audio blocked, starting anyway');
            setStatus('🎤 <b>Listening...</b> Say "Detect" to scan objects', 'listening');
            setVoiceFeedback('🗣️ Ready - say "Detect"');
            startListening();
        }}
        
        console.log('✅ NoonVision ready!');
    }}
    
    // Start initialization after page loads
    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', () => setTimeout(initialize, 1500));
    }} else {{
        setTimeout(initialize, 1500);
    }}
    
    // Also initialize on first click (for browsers that block autoplay)
    document.addEventListener('click', function firstClick() {{
        document.removeEventListener('click', firstClick);
        initialize();
    }});
    
    // Expose for debugging
    window.noonvision = {{
        performDetection,
        startListening,
        stopListening,
        captureFrame,
        resumeAfterDetection,
        status: () => ({{ isProcessing, isListening, isInitialized }})
    }};
    
    console.log('💡 Debug: Type noonvision.status() in console to check status');
    
}})();
</script>
'''

# ============================================================
# GRADIO INTERFACE
# ============================================================

def process_detection(image_b64):
    """Wrapper function called by Gradio - returns image and audio file"""
    if not image_b64 or len(image_b64) < 100:
        return None, None, "⚠️ No image"
    
    result_img_b64, audio_b64, status = detect_from_base64(image_b64)
    
    # Convert base64 back to image for Gradio
    result_image = None
    if result_img_b64:
        try:
            img_data = base64.b64decode(result_img_b64)
            result_image = Image.open(io.BytesIO(img_data))
        except:
            pass
    
    # Save audio to temp file for Gradio
    audio_path = None
    if audio_b64:
        try:
            fd, audio_path = tempfile.mkstemp(suffix='.mp3')
            os.close(fd)
            with open(audio_path, 'wb') as f:
                f.write(base64.b64decode(audio_b64))
        except:
            audio_path = None
    
    return result_image, audio_path, status

with gr.Blocks(title="NoonVision - Hands-Free Vision", css="""
    #hidden-row { display: none !important; }
    #result-image { min-height: 200px; }
    #result-audio { margin-top: 10px; }
""") as demo:
    
    # Custom HTML with camera and voice UI
    gr.HTML(CUSTOM_HTML)
    
    # Result display area (visible)
    with gr.Row():
        with gr.Column():
            result_image = gr.Image(label="🎯 Detection Result", elem_id="result-image", type="pil")
        with gr.Column():
            result_status = gr.Textbox(label="Status", value="Ready - Say 'Detect'", interactive=False)
            result_audio = gr.Audio(label="🔊 Audio Result", elem_id="result-audio", autoplay=True, type="filepath")
    
    # Hidden input for base64 image from JavaScript
    with gr.Row(elem_id="hidden-row"):
        hidden_input = gr.Textbox(elem_id="hidden-image-input", label="Image Data")
        hidden_btn = gr.Button("Detect", elem_id="hidden-detect-btn")
    
    # Connect hidden button to detection function
    hidden_btn.click(
        fn=process_detection,
        inputs=[hidden_input],
        outputs=[result_image, result_audio, result_status]
    )
    
    # JavaScript to handle audio end and resume listening
    gr.HTML('''
    <script>
    (function() {
        // Monitor audio element for when result audio finishes
        function setupAudioMonitor() {
            const audioElements = document.querySelectorAll('audio');
            audioElements.forEach(audio => {
                // Check if this is the result audio (not startup/processing)
                if (audio.id === 'audio-startup' || audio.id === 'audio-processing' || audio.id === 'audio-result') return;
                
                audio.addEventListener('ended', () => {
                    console.log('🔊 Gradio audio ended');
                    if (window.noonvision && window.noonvision.resumeAfterAudio) {
                        window.noonvision.resumeAfterAudio();
                    }
                });
            });
        }
        
        // Set up observer to detect when audio element is added/changed
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList' || mutation.type === 'attributes') {
                    setupAudioMonitor();
                }
            });
        });
        
        observer.observe(document.body, { 
            childList: true, 
            subtree: true,
            attributes: true,
            attributeFilter: ['src']
        });
        
        // Initial setup
        setTimeout(setupAudioMonitor, 2000);
    })();
    </script>
    ''')
    
    # Footer
    gr.HTML('''
    <div style="text-align:center;color:#666;padding:20px;margin-top:20px;border-top:1px solid #e5e7eb">
        <p style="margin:5px 0"><b>🎯 Detects 80+ objects</b> • <b>⚡ Fast response</b> • <b>🔊 Audio feedback</b></p>
        <p style="margin:5px 0;font-size:0.9em">Works best in Chrome or Edge • Made with ❤️ for accessibility</p>
        <p style="margin:10px 0 0 0;font-size:0.85em;color:#999">Debug: Open console (F12) and type <code>noonvision.status()</code></p>
    </div>
    ''')

demo.launch()