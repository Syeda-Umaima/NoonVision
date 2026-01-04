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
print("🚀 NoonVision Starting - Fully Hands-Free Mode")
print("=" * 50)

CONF_THRESHOLD = 0.30
IMG_SIZE = 640

# Load YOLO model
model = None
try:
    model = YOLO("yolov8m.pt")
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    _ = model(dummy, verbose=False)
    print("✅ YOLOv8m model loaded successfully")
except Exception as e:
    print(f"❌ Model error: {e}")

def make_audio(text):
    """Create TTS audio file"""
    try:
        fd, path = tempfile.mkstemp(suffix='.mp3')
        os.close(fd)
        gTTS(text=text, lang='en', slow=False).save(path)
        return path
    except Exception as e:
        print(f"Audio error: {e}")
        return None

def audio_to_base64(text):
    """Convert TTS to base64 for embedding"""
    path = make_audio(text)
    if path and os.path.exists(path):
        with open(path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        os.remove(path)
        return b64
    return ""

def decode_base64_image(base64_string):
    """Decode base64 string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image.convert('RGB')
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def process_detection(base64_image):
    """Main detection function - receives base64 image from JavaScript"""
    print(f"\n[DETECT] Received base64 image, length: {len(base64_image) if base64_image else 0}")
    
    if not base64_image or len(base64_image) < 100:
        print("[DETECT] No valid image data")
        return None, None, "⚠️ No image captured"
    
    if model is None:
        audio = make_audio("Model not ready. Please wait.")
        return None, audio, "⚠️ Model not loaded"
    
    try:
        t0 = time.time()
        
        # Decode base64 to PIL Image
        pil_image = decode_base64_image(base64_image)
        if pil_image is None:
            audio = make_audio("Could not process image. Please try again.")
            return None, audio, "⚠️ Image decode failed"
        
        img_array = np.array(pil_image)
        print(f"[DETECT] Image shape: {img_array.shape}")
        
        # Run YOLO detection
        results = model(img_array, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy()
        names = results.names
        
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
            label = names[int(cls_ids[i])]
            conf = confs[i]
            detected_objects.append(label)
            
            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=4)
            text = f"{label} {conf:.0%}"
            text_bbox = draw.textbbox((x1, y1 - 25), text, font=font)
            draw.rectangle(text_bbox, fill=(0, 255, 0))
            draw.text((x1, y1 - 25), text, fill="black", font=font)
        
        # Generate speech description
        if not detected_objects:
            speech = "I cannot see anything clearly in front of the camera. Try adjusting the camera angle or lighting."
        else:
            counts = Counter(detected_objects)
            parts = []
            for obj, count in counts.items():
                if count == 1:
                    parts.append(f"a {obj}")
                else:
                    parts.append(f"{count} {obj}s")
            
            if len(parts) == 1:
                speech = f"I can see {parts[0]} in front of you."
            elif len(parts) == 2:
                speech = f"I can see {parts[0]} and {parts[1]} in front of you."
            else:
                speech = f"I can see {', '.join(parts[:-1])}, and {parts[-1]} in front of you."
        
        audio_path = make_audio(speech)
        
        elapsed = time.time() - t0
        status = f"✅ Detected {len(detected_objects)} object(s) in {elapsed:.1f}s"
        if detected_objects:
            status += f": {', '.join(set(detected_objects))}"
        
        print(f"[DETECT] Complete: {status}")
        return pil_image, audio_path, status
        
    except Exception as e:
        print(f"[DETECT] Error: {e}")
        import traceback
        traceback.print_exc()
        audio = make_audio("An error occurred. Please try again. Say detect.")
        return None, audio, f"❌ Error: {str(e)}"

# Pre-generate audio clips
print("🔊 Generating audio clips...")
AUDIO_STARTUP = audio_to_base64("NoonVision is ready. Say detect to identify what's in front of you.")
AUDIO_PROCESSING = audio_to_base64("Processing. Please wait.")
AUDIO_READY = audio_to_base64("Ready. Say detect to scan again.")
print("✅ Audio clips ready")

# Complete HTML/CSS/JS for hands-free operation
CUSTOM_HTML = f'''
<div id="noonvision-container">
    <!-- Audio elements -->
    <audio id="audio-startup" src="data:audio/mp3;base64,{AUDIO_STARTUP}"></audio>
    <audio id="audio-processing" src="data:audio/mp3;base64,{AUDIO_PROCESSING}"></audio>
    <audio id="audio-ready" src="data:audio/mp3;base64,{AUDIO_READY}"></audio>
    
    <!-- Status display -->
    <div id="main-status" class="status-box status-initializing">
        🔄 Initializing camera and microphone...
    </div>
    
    <!-- Voice feedback -->
    <div id="voice-feedback">
        🗣️ Waiting for voice command...
    </div>
    
    <!-- Live video stream -->
    <div id="video-container">
        <video id="live-video" autoplay playsinline muted></video>
        <div id="video-overlay" class="hidden">
            <div class="overlay-text">📸 Captured - Analyzing...</div>
        </div>
    </div>
    
    <!-- Hidden canvas for frame capture -->
    <canvas id="capture-canvas" style="display:none;"></canvas>
</div>

<style>
#noonvision-container {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}}

.status-box {{
    padding: 20px;
    border-radius: 12px;
    margin: 15px 0;
    font-size: 20px;
    font-weight: 600;
    text-align: center;
}}

.status-initializing {{
    background: linear-gradient(135deg, #fef3c7, #fde68a);
    border: 2px solid #f59e0b;
    color: #92400e;
}}

.status-listening {{
    background: linear-gradient(135deg, #d1fae5, #a7f3d0);
    border: 2px solid #10b981;
    color: #065f46;
    animation: pulse-green 2s infinite;
}}

.status-processing {{
    background: linear-gradient(135deg, #dbeafe, #bfdbfe);
    border: 2px solid #3b82f6;
    color: #1e40af;
    animation: pulse-blue 1s infinite;
}}

.status-error {{
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    border: 2px solid #ef4444;
    color: #991b1b;
}}

@keyframes pulse-green {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.8; transform: scale(0.98); }}
}}

@keyframes pulse-blue {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.7; }}
}}

#voice-feedback {{
    background: #fefce8;
    border-left: 5px solid #eab308;
    padding: 15px 20px;
    border-radius: 8px;
    margin: 10px 0;
    font-size: 18px;
    text-align: center;
}}

#video-container {{
    position: relative;
    width: 100%;
    max-width: 640px;
    margin: 20px auto;
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
}}

#live-video {{
    width: 100%;
    display: block;
    background: #1a1a2e;
}}

#video-overlay {{
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    transition: opacity 0.3s;
}}

#video-overlay.hidden {{
    opacity: 0;
    pointer-events: none;
}}

.overlay-text {{
    color: white;
    font-size: 24px;
    font-weight: bold;
    text-align: center;
    padding: 20px;
}}

.results-container {{
    margin-top: 20px;
    padding: 15px;
    background: #f8fafc;
    border-radius: 10px;
}}
</style>

<script>
(function() {{
    // ============ STATE ============
    let videoStream = null;
    let recognition = null;
    let isListening = false;
    let isProcessing = false;
    let isInitialized = false;
    
    const TRIGGER_WORDS = ['detect', 'what do you see', 'identify', 'scan', 'look', 'check', 'what is'];
    
    // ============ DOM ELEMENTS ============
    const video = document.getElementById('live-video');
    const canvas = document.getElementById('capture-canvas');
    const overlay = document.getElementById('video-overlay');
    const statusBox = document.getElementById('main-status');
    const voiceFeedback = document.getElementById('voice-feedback');
    
    // ============ UTILITY FUNCTIONS ============
    function setStatus(message, type) {{
        if (statusBox) {{
            statusBox.innerHTML = message;
            statusBox.className = 'status-box status-' + type;
        }}
    }}
    
    function setVoiceFeedback(text) {{
        if (voiceFeedback) {{
            voiceFeedback.innerHTML = '🗣️ ' + text;
        }}
    }}
    
    function playAudio(id) {{
        return new Promise((resolve) => {{
            const audio = document.getElementById(id);
            if (audio) {{
                audio.currentTime = 0;
                audio.onended = resolve;
                audio.play().catch(() => resolve());
            }} else {{
                resolve();
            }}
        }});
    }}
    
    function hasTriggerWord(text) {{
        const lower = text.toLowerCase();
        return TRIGGER_WORDS.some(trigger => lower.includes(trigger));
    }}
    
    function sleep(ms) {{
        return new Promise(resolve => setTimeout(resolve, ms));
    }}
    
    // ============ CAMERA FUNCTIONS ============
    async function startCamera() {{
        try {{
            console.log('📷 Starting camera...');
            setStatus('📷 Starting camera...', 'initializing');
            
            videoStream = await navigator.mediaDevices.getUserMedia({{
                video: {{ 
                    facingMode: 'environment',
                    width: {{ ideal: 640 }},
                    height: {{ ideal: 480 }}
                }},
                audio: false
            }});
            
            video.srcObject = videoStream;
            await video.play();
            
            console.log('✅ Camera started successfully');
            return true;
        }} catch (err) {{
            console.error('❌ Camera error:', err);
            setStatus('❌ Camera access denied. Please allow camera and refresh.', 'error');
            return false;
        }}
    }}
    
    function captureFrame() {{
        if (!video || !video.videoWidth) {{
            console.error('Video not ready');
            return null;
        }}
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        
        return canvas.toDataURL('image/jpeg', 0.8);
    }}
    
    function showOverlay(show) {{
        if (overlay) {{
            overlay.classList.toggle('hidden', !show);
        }}
    }}
    
    // ============ SPEECH RECOGNITION ============
    function initSpeechRecognition() {{
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {{
            console.error('Speech recognition not supported');
            setStatus('⚠️ Voice not supported. Please use Chrome or Edge.', 'error');
            return false;
        }}
        
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = false;
        recognition.lang = 'en-US';
        
        recognition.onstart = () => {{
            isListening = true;
            console.log('🎤 Listening started');
            if (!isProcessing) {{
                setStatus('🎤 <b>Listening...</b> Say "Detect" to scan', 'listening');
            }}
        }};
        
        recognition.onresult = (event) => {{
            if (isProcessing) return;
            
            const result = event.results[event.results.length - 1];
            const text = result[0].transcript;
            
            console.log('🗣️ Heard:', text);
            setVoiceFeedback('Heard: "' + text + '"');
            
            if (hasTriggerWord(text)) {{
                console.log('🎯 TRIGGER DETECTED!');
                performDetection();
            }}
        }};
        
        recognition.onerror = (event) => {{
            console.log('Speech error:', event.error);
            isListening = false;
            
            if (event.error !== 'no-speech' && event.error !== 'aborted') {{
                setTimeout(startListening, 1000);
            }}
        }};
        
        recognition.onend = () => {{
            isListening = false;
            console.log('🎤 Listening ended');
            
            if (!isProcessing && isInitialized) {{
                setTimeout(startListening, 500);
            }}
        }};
        
        return true;
    }}
    
    function startListening() {{
        if (!recognition) {{
            if (!initSpeechRecognition()) return;
        }}
        
        if (!isListening && !isProcessing) {{
            try {{
                recognition.start();
            }} catch (e) {{
                console.log('Recognition start error:', e);
            }}
        }}
    }}
    
    function stopListening() {{
        if (recognition && isListening) {{
            try {{
                recognition.stop();
            }} catch (e) {{}}
        }}
    }}
    
    // ============ MAIN DETECTION FUNCTION ============
    async function performDetection() {{
        if (isProcessing) {{
            console.log('Already processing, ignoring...');
            return;
        }}
        
        isProcessing = true;
        stopListening();
        
        console.log('========================================');
        console.log('🚀 STARTING DETECTION');
        console.log('========================================');
        
        try {{
            // Step 1: Show processing state
            setStatus('📸 <b>Capturing frame...</b>', 'processing');
            showOverlay(true);
            await playAudio('audio-processing');
            
            // Step 2: Capture frame
            const frameData = captureFrame();
            if (!frameData) {{
                throw new Error('Failed to capture frame');
            }}
            console.log('📸 Frame captured, size:', frameData.length);
            
            // Step 3: Send to Gradio for processing
            setStatus('🔍 <b>Analyzing image...</b>', 'processing');
            
            // Find the hidden textbox and update it to trigger detection
            // Try multiple selectors for the hidden textbox
            let textbox = document.querySelector('#hidden-trigger textarea');
            if (!textbox) textbox = document.querySelector('#trigger-textbox textarea');
            if (!textbox) textbox = document.querySelector('textarea[data-testid="textbox"]');
            if (!textbox) {{
                // Last resort: find any textarea that's hidden
                const allTextareas = document.querySelectorAll('textarea');
                for (const ta of allTextareas) {{
                    const parent = ta.closest('[style*="display: none"], [style*="display:none"], .hidden');
                    if (parent || ta.offsetParent === null) {{
                        textbox = ta;
                        break;
                    }}
                }}
            }}
            
            if (textbox) {{
                console.log('📤 Found textbox, sending frame...');
                
                // Set the value using Object.getOwnPropertyDescriptor to trigger React
                const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value'
                ).set;
                nativeInputValueSetter.call(textbox, frameData);
                
                // Trigger multiple events to ensure Gradio picks up the change
                textbox.dispatchEvent(new Event('input', {{ bubbles: true, cancelable: true }}));
                await sleep(50);
                textbox.dispatchEvent(new Event('change', {{ bubbles: true, cancelable: true }}));
                await sleep(50);
                textbox.dispatchEvent(new KeyboardEvent('keyup', {{ bubbles: true }}));
                
                console.log('📤 Frame sent to Gradio');
            }} else {{
                console.error('❌ Hidden textbox not found! Trying button fallback...');
                // Fallback: try to find and click detect button
                const detectBtn = document.querySelector('#detect-btn button, button#detect-btn');
                if (detectBtn) {{
                    detectBtn.click();
                    console.log('Clicked fallback button');
                }}
            }}
            
            // Step 4: Wait for detection to complete (audio will play)
            console.log('⏳ Waiting for results...');
            await sleep(6000);
            
        }} catch (err) {{
            console.error('Detection error:', err);
            setStatus('⚠️ Error: ' + err.message, 'error');
        }}
        
        // Step 5: Reset for next detection
        showOverlay(false);
        isProcessing = false;
        
        // Clear the hidden textbox
        let clearTextbox = document.querySelector('#hidden-trigger textarea');
        if (!clearTextbox) clearTextbox = document.querySelector('#trigger-textbox textarea');
        if (clearTextbox) {{
            const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                window.HTMLTextAreaElement.prototype, 'value'
            ).set;
            nativeInputValueSetter.call(clearTextbox, '');
            clearTextbox.dispatchEvent(new Event('input', {{ bubbles: true }}));
        }}
        
        // Play ready audio and resume listening
        await playAudio('audio-ready');
        setStatus('🎤 <b>Listening...</b> Say "Detect" to scan again', 'listening');
        setVoiceFeedback('Waiting for voice command...');
        
        console.log('========================================');
        console.log('✅ READY FOR NEXT DETECTION');
        console.log('========================================');
        
        startListening();
    }}
    
    // ============ INITIALIZATION ============
    async function initialize() {{
        if (isInitialized) return;
        
        console.log('🚀 NoonVision initializing...');
        setStatus('🔄 Initializing...', 'initializing');
        
        // Start camera
        const cameraOk = await startCamera();
        if (!cameraOk) return;
        
        // Initialize speech recognition
        const speechOk = initSpeechRecognition();
        if (!speechOk) return;
        
        isInitialized = true;
        
        // Play startup audio
        setStatus('🔊 Playing startup message...', 'initializing');
        await playAudio('audio-startup');
        
        // Start listening
        setStatus('🎤 <b>Listening...</b> Say "Detect" to scan', 'listening');
        startListening();
        
        console.log('✅ NoonVision fully initialized');
    }}
    
    // Start after page loads
    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', () => setTimeout(initialize, 1000));
    }} else {{
        setTimeout(initialize, 1000);
    }}
    
    // Also initialize on first click (for browsers that block autoplay)
    document.addEventListener('click', function firstClick() {{
        document.removeEventListener('click', firstClick);
        if (!isInitialized) initialize();
    }});
    
    // Expose for debugging
    window.nv = {{
        initialize,
        performDetection,
        captureFrame,
        startListening,
        stopListening
    }};
    
    console.log('💡 Debug: window.nv.performDetection() to test manually');
}})();
</script>
'''

# CSS for the Gradio interface
CSS = """
footer {display: none !important;}
.results-section {margin-top: 20px;}
#hidden-trigger {position: absolute; left: -9999px; opacity: 0;}
"""

# Build the Gradio interface
with gr.Blocks(css=CSS, title="NoonVision - Hands-Free") as demo:
    
    # Header
    gr.HTML("""
    <div style="text-align:center;padding:25px;background:linear-gradient(135deg,#667eea,#764ba2);color:white;border-radius:15px;margin-bottom:20px">
        <h1 style="margin:0;font-size:2.5em">🦾 NoonVision</h1>
        <p style="margin:10px 0 0 0;font-size:1.2em;opacity:0.9">100% Hands-Free AI Vision Assistant</p>
    </div>
    """)
    
    # Custom HTML with video stream and voice recognition
    gr.HTML(CUSTOM_HTML)
    
    # Hidden textbox to receive base64 image from JavaScript
    # Using visible=False to hide it while keeping it functional
    hidden_input = gr.Textbox(
        label="Image Data",
        elem_id="hidden-trigger",
        visible=False
    )
    
    # Results section
    gr.HTML('<div class="results-section"><h3>🎯 Detection Results</h3></div>')
    
    with gr.Row():
        result_image = gr.Image(type="pil", label="Detected Objects", show_label=True)
        with gr.Column():
            result_status = gr.Textbox(label="Status", value="Waiting for detection...", interactive=False)
            result_audio = gr.Audio(type="filepath", label="Audio Result", autoplay=True)
    
    # Hidden button for triggering detection
    detect_btn = gr.Button("Detect", visible=False, elem_id="detect-btn")
    
    # Connect the hidden input to detection function
    hidden_input.change(
        fn=process_detection,
        inputs=[hidden_input],
        outputs=[result_image, result_audio, result_status]
    )
    
    # Also connect button as fallback
    detect_btn.click(
        fn=lambda: process_detection(""),
        outputs=[result_image, result_audio, result_status]
    )
    
    # Footer
    gr.HTML("""
    <div style="text-align:center;color:#666;padding:20px;margin-top:20px;border-top:1px solid #e5e7eb">
        <p><b>🎯 80+ object types</b> • <b>⚡ Real-time detection</b> • <b>🔊 Audio feedback</b></p>
        <p style="font-size:0.9em">Works best in Chrome or Edge • No buttons needed!</p>
    </div>
    """)

# Launch
if __name__ == "__main__":
    demo.launch()