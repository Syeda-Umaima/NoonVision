import gradio as gr
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import time
from collections import Counter
import os
import base64

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

# Trigger phrases for voice detection
TRIGGER_PHRASES = ["detect", "what do you see", "what's in front", "identify", "scan", "look", "what is in front", "what's this", "what is this"]

def get_audio_base64(filepath):
    """Convert audio file to base64 for embedding in HTML"""
    try:
        if filepath and os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except Exception as e:
        print(f"⚠️ Base64 conversion error: {e}")
    return ""

def generate_startup_audio():
    """Generate startup announcement audio"""
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
    """Generate processing indicator audio"""
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
    """Generate natural speech for detected objects with listening prompt"""
    try:
        if not labels:
            text = "I don't see any recognizable objects at the moment. Try pointing the camera at something else, or move a bit closer to the object."
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
        
        # Add listening prompt at the end
        text += " ... Listening. Say detect when ready."
        
        filename = f"result_{int(time.time()*1000)}.mp3"
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filename)
        return filename
    except Exception as e:
        print(f"⚠️ Audio generation error: {e}")
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

def detect_objects(image):
    """Main detection function - returns annotated image and audio"""
    if image is None:
        error_text = "I cannot see anything right now. Please make sure the camera is working and try again."
        error_audio = generate_error_audio(error_text)
        return None, error_audio, "⚠️ No image received from camera"
    
    if model is None:
        error_text = "The detection system is not ready yet. Please wait a moment and try again."
        error_audio = generate_error_audio(error_text)
        return None, error_audio, "⚠️ Model not loaded"
    
    try:
        start = time.time()
        
        # Convert image
        if isinstance(image, Image.Image):
            img_np = np.array(image)
            img_pil = image.copy()
        else:
            img_np = image
            img_pil = Image.fromarray(image)
        
        # Run YOLO detection
        results = model(img_np, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        labels = results.names
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        
        # Draw bounding boxes on image
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
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)
            text = f"{label} {conf:.2f}"
            bbox = draw.textbbox((x1, y1 - 25), text, font=font)
            draw.rectangle(bbox, fill=BOX_COLOR)
            draw.text((x1, y1 - 25), text, fill="black", font=font)
        
        # Generate result audio
        audio = generate_result_audio(detected_labels)
        
        elapsed = time.time() - start
        
        if detected_labels:
            status = f"✅ Found {len(detected_labels)} object(s) in {elapsed:.2f}s: {', '.join(set(detected_labels))}"
        else:
            status = f"🔍 No objects detected ({elapsed:.2f}s) - Try adjusting camera angle or lighting"
        
        return img_pil, audio, status
        
    except Exception as e:
        error_text = "Sorry, something went wrong during detection. Please try again."
        error_audio = generate_error_audio(error_text)
        return None, error_audio, f"❌ Error: {str(e)}"

# Pre-generate audio files at startup
print("🔊 Generating startup audio files...")
startup_audio_file = generate_startup_audio()
processing_audio_file = generate_processing_audio()
startup_audio_base64 = get_audio_base64(startup_audio_file)
processing_audio_base64 = get_audio_base64(processing_audio_file)
print("✅ Audio files ready")

# Custom CSS for accessibility and styling
CUSTOM_CSS = """
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

.result-image {
    border: 3px solid #22c55e;
    border-radius: 12px;
}

/* Make sure webcam is prominent */
.webcam-container {
    border: 2px solid #667eea;
    border-radius: 12px;
    overflow: hidden;
}

/* Ensure audio player is accessible */
audio {
    width: 100%;
    margin-top: 10px;
}

/* Browser support warning */
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
"""

# Build the Gradio interface
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
    
    # Browser compatibility warning (hidden by default, shown if needed)
    gr.HTML('''
    <div id="browser-warning" class="browser-warning">
        <strong>⚠️ Browser Compatibility:</strong> Voice recognition works best in <strong>Google Chrome</strong> or <strong>Microsoft Edge</strong>. 
        If voice commands aren't working, please switch to one of these browsers.
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
            <li><strong>Repeat</strong> - NoonVision automatically listens again after each detection</li>
        </ol>
        <p style="margin-bottom: 0; color: #166534;"><strong>💡 Tips:</strong> Speak clearly • Good lighting helps • Hold objects 2-6 feet from camera</p>
    </div>
    ''')
    
    # Status indicators
    gr.HTML('''
    <div id="listening-indicator" class="listening-active">
        🎤 <span style="color: #22c55e; font-weight: bold;">Initializing...</span> Please allow microphone access
    </div>
    ''')
    
    gr.HTML('<div id="status-display">🚀 Starting NoonVision... Please allow camera and microphone permissions.</div>')
    gr.HTML('<div id="heard-text">🗣️ Waiting for voice command...</div>')
    
    # Main content area
    with gr.Row():
        with gr.Column(scale=1):
            webcam = gr.Image(
                sources=["webcam"],
                type="pil",
                label="📷 Live Camera Feed",
                streaming=True,
                mirror_webcam=True,
                elem_classes=["webcam-container"]
            )
        
        with gr.Column(scale=1):
            result_img = gr.Image(
                type="pil",
                label="🎯 Detection Results",
                elem_classes=["result-image"]
            )
            
            status_text = gr.Textbox(
                label="Detection Status",
                value="Ready - Say 'Detect' to identify objects",
                lines=2,
                interactive=False
            )
            
            audio_out = gr.Audio(
                type="filepath",
                label="🔊 Audio Result (plays automatically)",
                autoplay=True
            )
    
    # Hidden detect button (triggered by JavaScript)
    with gr.Row(visible=False):
        hidden_btn = gr.Button("Detect", elem_id="hidden-detect-btn")
    
    # Audio elements and JavaScript
    gr.HTML(f'''
    <!-- Startup and processing audio -->
    <audio id="startup-audio" preload="auto">
        <source src="data:audio/mp3;base64,{startup_audio_base64}" type="audio/mp3">
    </audio>
    <audio id="processing-audio" preload="auto">
        <source src="data:audio/mp3;base64,{processing_audio_base64}" type="audio/mp3">
    </audio>
    
    <script>
    (function() {{
        let recognition = null;
        let isProcessing = false;
        let isListening = false;
        let hasInteracted = false;
        
        const TRIGGER_PHRASES = ["detect", "what do you see", "what's in front", "identify", "scan", "look", "what is in front", "what's this", "what is this"];
        
        function containsTrigger(text) {{
            const lowerText = text.toLowerCase();
            return TRIGGER_PHRASES.some(phrase => lowerText.includes(phrase));
        }}
        
        function updateUI(state, message) {{
            const indicator = document.getElementById('listening-indicator');
            const status = document.getElementById('status-display');
            
            if (indicator) {{
                if (state === 'listening') {{
                    indicator.className = 'listening-active';
                    indicator.innerHTML = '🎤 <span style="color: #22c55e; font-weight: bold;">Listening...</span> Say "Detect" or "What do you see?"';
                }} else if (state === 'processing') {{
                    indicator.className = 'processing-active';
                    indicator.innerHTML = '🔍 <span style="color: #3b82f6; font-weight: bold;">Processing...</span> Please wait';
                }} else if (state === 'paused') {{
                    indicator.className = 'listening-paused';
                    indicator.innerHTML = '⏸️ <span style="color: #f59e0b;">Paused</span> - ' + message;
                }} else if (state === 'error') {{
                    indicator.className = 'listening-paused';
                    indicator.innerHTML = '⚠️ <span style="color: #ef4444;">' + message + '</span>';
                }}
            }}
            
            if (status && message) {{
                status.innerHTML = message;
            }}
        }}
        
        function initSpeechRecognition() {{
            // Check browser support
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {{
                console.error('Speech recognition not supported');
                document.getElementById('browser-warning').classList.add('show');
                updateUI('error', 'Speech recognition not supported. Please use Chrome or Edge browser.');
                return false;
            }}
            
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = false;
            recognition.lang = 'en-US';
            recognition.maxAlternatives = 1;
            
            recognition.onstart = function() {{
                isListening = true;
                console.log('🎤 Voice recognition started');
                if (!isProcessing) {{
                    updateUI('listening', '🎤 Listening for voice commands...');
                }}
            }};
            
            recognition.onresult = function(event) {{
                if (isProcessing) {{
                    console.log('⏳ Already processing, ignoring input');
                    return;
                }}
                
                const last = event.results.length - 1;
                const text = event.results[last][0].transcript;
                const confidence = event.results[last][0].confidence;
                
                console.log('Heard:', text, 'Confidence:', confidence);
                
                const heardEl = document.getElementById('heard-text');
                if (heardEl) {{
                    heardEl.innerHTML = '🗣️ Heard: "' + text + '"';
                }}
                
                if (containsTrigger(text)) {{
                    console.log('🎯 Trigger word detected!');
                    triggerDetection();
                }}
            }};
            
            recognition.onerror = function(event) {{
                console.error('Speech recognition error:', event.error);
                
                if (event.error === 'not-allowed') {{
                    updateUI('error', 'Microphone access denied. Please allow microphone permission and refresh the page.');
                    document.getElementById('browser-warning').classList.add('show');
                }} else if (event.error === 'no-speech') {{
                    // Normal timeout, just restart
                    if (!isProcessing) {{
                        setTimeout(startListening, 100);
                    }}
                }} else if (event.error === 'audio-capture') {{
                    updateUI('error', 'No microphone found. Please connect a microphone and refresh.');
                }} else {{
                    // Other errors - try to restart
                    console.log('Restarting after error:', event.error);
                    setTimeout(startListening, 1000);
                }}
            }};
            
            recognition.onend = function() {{
                isListening = false;
                console.log('🎤 Voice recognition ended');
                
                // Auto-restart if not processing
                if (!isProcessing) {{
                    setTimeout(startListening, 300);
                }}
            }};
            
            return true;
        }}
        
        function startListening() {{
            if (!recognition) {{
                if (!initSpeechRecognition()) {{
                    return;
                }}
            }}
            
            if (!isListening && !isProcessing) {{
                try {{
                    recognition.start();
                    console.log('🎤 Starting voice recognition...');
                }} catch (e) {{
                    // Already started
                    console.log('Recognition already running');
                }}
            }}
        }}
        
        function stopListening() {{
            if (recognition && isListening) {{
                try {{
                    recognition.stop();
                }} catch (e) {{
                    console.log('Error stopping recognition:', e);
                }}
            }}
        }}
        
        function triggerDetection() {{
            if (isProcessing) {{
                console.log('Already processing, ignoring');
                return;
            }}
            
            isProcessing = true;
            updateUI('processing', '🔍 Analyzing image... Please wait.');
            
            // Stop listening during processing
            stopListening();
            
            // Play processing sound
            const processingAudio = document.getElementById('processing-audio');
            if (processingAudio) {{
                processingAudio.currentTime = 0;
                processingAudio.play().catch(e => console.log('Processing audio blocked:', e));
            }}
            
            // Find and click the hidden Gradio button
            setTimeout(() => {{
                // Try multiple selectors to find the button
                let detectBtn = document.querySelector('#hidden-detect-btn');
                if (!detectBtn) {{
                    detectBtn = document.querySelector('button#hidden-detect-btn');
                }}
                if (!detectBtn) {{
                    // Look for the button inside Gradio's structure
                    const allButtons = document.querySelectorAll('button');
                    allButtons.forEach(btn => {{
                        if (btn.id === 'hidden-detect-btn' || (btn.textContent && btn.textContent.includes('Detect'))) {{
                            detectBtn = btn;
                        }}
                    }});
                }}
                
                if (detectBtn) {{
                    console.log('🔘 Clicking detect button');
                    detectBtn.click();
                }} else {{
                    console.error('Could not find detect button');
                    detectionComplete();
                }}
            }}, 100);
        }}
        
        function detectionComplete() {{
            console.log('✅ Detection complete');
            isProcessing = false;
            
            // Resume listening after audio finishes (give time for result audio)
            setTimeout(() => {{
                updateUI('listening', '✅ Detection complete! Listening for next command...');
                startListening();
            }}, 500);
        }}
        
        function playStartupAndBegin() {{
            const startupAudio = document.getElementById('startup-audio');
            if (startupAudio) {{
                startupAudio.play()
                    .then(() => {{
                        console.log('🔊 Startup audio playing');
                        // Start listening after startup audio
                        startupAudio.onended = () => {{
                            console.log('🔊 Startup audio ended, beginning listening');
                            startListening();
                        }};
                    }})
                    .catch(e => {{
                        console.log('🔇 Auto-play blocked, starting without audio');
                        startListening();
                    }});
            }} else {{
                startListening();
            }}
        }}
        
        // Handle first user interaction (needed for audio autoplay policy)
        function handleFirstInteraction() {{
            if (!hasInteracted) {{
                hasInteracted = true;
                console.log('👆 First interaction detected');
                playStartupAndBegin();
            }}
        }}
        
        // Expose for Gradio callback
        window.detectionComplete = detectionComplete;
        window.noonvisionStart = playStartupAndBegin;
        
        // Initialize on page load
        function init() {{
            console.log('🚀 NoonVision initializing...');
            
            updateUI('paused', 'Click anywhere or speak to activate NoonVision');
            
            // Set up first interaction handlers
            document.addEventListener('click', handleFirstInteraction, {{ once: false }});
            document.addEventListener('keydown', handleFirstInteraction, {{ once: false }});
            document.addEventListener('touchstart', handleFirstInteraction, {{ once: false }});
            
            // Try to start immediately (may work if user has interacted before)
            setTimeout(() => {{
                if (!hasInteracted) {{
                    // Try to init speech recognition to prompt for permission
                    if (initSpeechRecognition()) {{
                        startListening();
                        hasInteracted = true;
                        
                        // Try to play startup audio
                        const startupAudio = document.getElementById('startup-audio');
                        if (startupAudio) {{
                            startupAudio.play().catch(() => {{}});
                        }}
                    }}
                }}
            }}, 1500);
        }}
        
        // Wait for DOM
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', init);
        }} else {{
            setTimeout(init, 500);
        }}
    }})();
    </script>
    ''')
    
    # Event handler for detection button
    hidden_btn.click(
        fn=detect_objects,
        inputs=webcam,
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
        <p><strong>🎯 Detects 80+ objects:</strong> People, furniture, electronics, food, animals, vehicles, and more</p>
        <p><strong>⚡ Response time:</strong> 1-2 seconds</p>
        <p style="margin-top: 15px; font-size: 0.9em;">
            Built with YOLOv8 + Web Speech API<br>
            <strong>Best experience:</strong> Chrome or Edge browser | Good lighting | Objects 2-6 feet away<br>
            Made with ❤️ for accessibility
        </p>
    </div>
    ''')

# Launch
if __name__ == "__main__":
    demo.launch()