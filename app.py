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
model = None
try:
    model = YOLO("yolov8m.pt")
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    _ = model(dummy, verbose=False)
    print("✅ YOLOv8m loaded successfully")
except Exception as e:
    print(f"❌ YOLO failed: {e}")

def get_audio_base64(filepath):
    try:
        if filepath and os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except:
        pass
    return ""

def generate_audio(text):
    try:
        filename = f"audio_{int(time.time()*1000)}.mp3"
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filename)
        return filename
    except Exception as e:
        print(f"Audio error: {e}")
        return None

def detect_from_base64(image_base64):
    """Detect objects from base64 image"""
    
    if not image_base64 or len(str(image_base64)) < 100:
        text = "I cannot see anything right now. Please check the camera. Listening. Say detect when ready."
        return None, generate_audio(text), "⚠️ No image received"
    
    if model is None:
        text = "Detection system not ready. Please wait. Listening. Say detect when ready."
        return None, generate_audio(text), "⚠️ Model not loaded"
    
    try:
        start = time.time()
        
        # Decode base64
        if ',' in str(image_base64):
            image_base64 = str(image_base64).split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        img_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_np = np.array(img_pil)
        
        # Run YOLO
        results = model(img_np, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        labels = results.names
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        
        # Draw boxes
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
        except:
            font = ImageFont.load_default()
        
        detected_labels = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = labels[int(class_ids[i])]
            conf = confidences[i]
            detected_labels.append(label)
            
            draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)
            text_label = f"{label} {conf:.2f}"
            bbox = draw.textbbox((x1, y1 - 25), text_label, font=font)
            draw.rectangle(bbox, fill=BOX_COLOR)
            draw.text((x1, y1 - 25), text_label, fill="black", font=font)
        
        # Generate speech
        if not detected_labels:
            speech = "I don't see any recognizable objects. Try pointing the camera at something else."
        else:
            counts = Counter(detected_labels)
            if len(counts) == 1:
                obj, count = list(counts.items())[0]
                speech = f"I can see {count} {obj}{'s' if count > 1 else ''} in front of you."
            else:
                items = [f"{count} {obj}{'s' if count > 1 else ''}" for obj, count in counts.items()]
                if len(items) == 2:
                    speech = f"I can see {items[0]} and {items[1]}."
                else:
                    speech = f"I can see {', '.join(items[:-1])}, and {items[-1]}."
        
        speech += " ... Listening. Say detect when ready."
        audio = generate_audio(speech)
        
        elapsed = time.time() - start
        status = f"✅ Found {len(detected_labels)} object(s) in {elapsed:.2f}s"
        
        return img_pil, audio, status
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        text = "Something went wrong. Please try again. Listening. Say detect when ready."
        return None, generate_audio(text), f"❌ Error: {str(e)}"

# Generate startup audio
print("🔊 Generating audio...")
startup_audio = generate_audio("NoonVision ready. Say detect to identify objects around you.")
startup_base64 = get_audio_base64(startup_audio)
processing_audio = generate_audio("Processing.")
processing_base64 = get_audio_base64(processing_audio)
print("✅ Ready")

# CSS
CSS = """
#video-box {
    width: 100%;
    max-width: 640px;
    margin: 0 auto;
    border-radius: 12px;
    overflow: hidden;
    border: 3px solid #667eea;
    position: relative;
    background: #000;
}
#webcam-video {
    width: 100%;
    display: block;
    transform: scaleX(-1);
}
#frozen-frame {
    width: 100%;
    display: none;
    transform: scaleX(-1);
}
#hidden-canvas { display: none; }
#cam-status {
    position: absolute;
    top: 10px;
    left: 10px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: bold;
    color: white;
    background: #22c55e;
}
#cam-status.recording { background: #22c55e; }
#cam-status.paused { background: #f59e0b; }
#cam-status.error { background: #ef4444; }

.status-box {
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    text-align: center;
}
#listening-box {
    background: linear-gradient(90deg, #22c55e20, #22c55e10);
    border-left: 4px solid #22c55e;
    animation: pulse 2s infinite;
}
#listening-box.processing {
    background: linear-gradient(90deg, #3b82f620, #3b82f610);
    border-left-color: #3b82f6;
    animation: none;
}
#heard-box {
    background: #fefce8;
    border-left: 4px solid #eab308;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}
"""

# JavaScript
JS = """
async function() {
    // Wait for elements
    await new Promise(r => setTimeout(r, 1000));
    
    let video, canvas, ctx, frozenImg;
    let isProcessing = false;
    let isListening = false;
    let recognition = null;
    let stream = null;
    
    const TRIGGERS = ["detect", "what do you see", "what's in front", "identify", "scan", "look"];
    
    function $(id) { return document.getElementById(id); }
    
    function hasTrigger(text) {
        const t = text.toLowerCase();
        return TRIGGERS.some(p => t.includes(p));
    }
    
    function setStatus(id, html) {
        const el = $(id);
        if (el) el.innerHTML = html;
    }
    
    function setCamStatus(text, cls) {
        const el = $('cam-status');
        if (el) {
            el.textContent = text;
            el.className = cls;
        }
    }
    
    // Initialize camera
    async function initCamera() {
        video = $('webcam-video');
        canvas = $('hidden-canvas');
        frozenImg = $('frozen-frame');
        
        if (!video || !canvas) {
            setTimeout(initCamera, 500);
            return;
        }
        
        ctx = canvas.getContext('2d');
        
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }
            });
            video.srcObject = stream;
            video.onloadedmetadata = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                setCamStatus('🔴 LIVE', 'recording');
                console.log('Camera ready');
            };
        } catch (e) {
            console.error('Camera error:', e);
            setCamStatus('❌ ERROR', 'error');
        }
    }
    
    // Capture and freeze frame
    function captureFrame() {
        if (!video || !ctx) return null;
        
        // Draw to canvas (flip)
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(video, -canvas.width, 0);
        ctx.restore();
        
        // Get base64
        const data = canvas.toDataURL('image/jpeg', 0.85);
        
        // Show frozen frame, hide video
        frozenImg.src = data;
        frozenImg.style.display = 'block';
        video.style.display = 'none';
        setCamStatus('⏸️ PAUSED', 'paused');
        
        return data;
    }
    
    // Resume live video
    function resumeVideo() {
        if (frozenImg) frozenImg.style.display = 'none';
        if (video) video.style.display = 'block';
        setCamStatus('🔴 LIVE', 'recording');
    }
    
    // Speech recognition
    function initSpeech() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            setStatus('listening-box', '⚠️ Speech not supported. Use Chrome or Edge.');
            return false;
        }
        
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SR();
        recognition.continuous = true;
        recognition.interimResults = false;
        recognition.lang = 'en-US';
        
        recognition.onstart = () => {
            isListening = true;
            if (!isProcessing) {
                $('listening-box').className = 'status-box';
                $('listening-box').id = 'listening-box';
                setStatus('listening-box', '🎤 <b style="color:#22c55e">Listening...</b> Say "Detect"');
            }
        };
        
        recognition.onresult = (e) => {
            if (isProcessing) return;
            const text = e.results[e.results.length - 1][0].transcript;
            console.log('Heard:', text);
            setStatus('heard-box', '🗣️ Heard: "' + text + '"');
            
            if (hasTrigger(text)) {
                triggerDetection();
            }
        };
        
        recognition.onerror = (e) => {
            if (e.error !== 'no-speech' && e.error !== 'aborted') {
                console.error('Speech error:', e.error);
            }
            if (!isProcessing) setTimeout(startListening, 500);
        };
        
        recognition.onend = () => {
            isListening = false;
            if (!isProcessing) setTimeout(startListening, 300);
        };
        
        return true;
    }
    
    function startListening() {
        if (!recognition) initSpeech();
        if (recognition && !isListening && !isProcessing) {
            try { recognition.start(); } catch(e) {}
        }
    }
    
    function stopListening() {
        if (recognition && isListening) {
            try { recognition.stop(); } catch(e) {}
        }
    }
    
    // Main detection trigger
    async function triggerDetection() {
        if (isProcessing) return;
        isProcessing = true;
        
        // Update UI
        $('listening-box').className = 'status-box processing';
        setStatus('listening-box', '🔍 <b style="color:#3b82f6">Processing...</b> Please wait');
        stopListening();
        
        // Play processing sound
        const procAudio = $('processing-audio');
        if (procAudio) { procAudio.currentTime = 0; procAudio.play().catch(()=>{}); }
        
        // Capture frame (pauses video)
        const imageData = captureFrame();
        if (!imageData) {
            setStatus('listening-box', '⚠️ Failed to capture. Try again.');
            isProcessing = false;
            resumeVideo();
            startListening();
            return;
        }
        
        try {
            // Method 1: Try using Gradio's API endpoint
            let apiBase = window.location.origin + window.location.pathname;
            if (!apiBase.endsWith('/')) apiBase += '/';
            
            // Try the Gradio API endpoint
            const response = await fetch(apiBase + 'api/detect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ data: [imageData] })
            });
            
            if (!response.ok) {
                throw new Error('API error: ' + response.status);
            }
            
            const result = await response.json();
            console.log('Result:', result);
            
            // Handle response
            if (result.data) {
                // Update result image
                const resultImgContainer = document.querySelector('#result-image');
                if (resultImgContainer) {
                    const img = resultImgContainer.querySelector('img') || document.createElement('img');
                    if (result.data[0] && result.data[0].url) {
                        img.src = result.data[0].url;
                    } else if (result.data[0] && result.data[0].path) {
                        img.src = apiBase + 'file=' + result.data[0].path;
                    }
                }
                
                // Play audio
                if (result.data[1]) {
                    const audioContainer = document.querySelector('#audio-output');
                    let audioEl = audioContainer ? audioContainer.querySelector('audio') : null;
                    if (!audioEl) {
                        audioEl = document.createElement('audio');
                        audioEl.controls = true;
                        audioEl.autoplay = true;
                        if (audioContainer) audioContainer.appendChild(audioEl);
                    }
                    
                    let audioSrc = '';
                    if (result.data[1].url) {
                        audioSrc = result.data[1].url;
                    } else if (result.data[1].path) {
                        audioSrc = apiBase + 'file=' + result.data[1].path;
                    } else if (typeof result.data[1] === 'string') {
                        audioSrc = result.data[1];
                    }
                    
                    if (audioSrc) {
                        audioEl.src = audioSrc;
                        audioEl.play().catch(e => console.log('Audio play error:', e));
                    }
                }
                
                // Update status
                if (result.data[2]) {
                    const statusTextarea = document.querySelector('#status-text textarea');
                    if (statusTextarea) statusTextarea.value = result.data[2];
                }
            }
            
            // Resume after delay
            setTimeout(() => {
                resumeVideo();
                isProcessing = false;
                setStatus('listening-box', '🎤 <b style="color:#22c55e">Listening...</b> Say "Detect"');
                $('listening-box').className = 'status-box';
                startListening();
            }, 4000);
            
        } catch (error) {
            console.error('Error:', error);
            
            // Fallback: try clicking the Gradio button directly
            try {
                const inputEl = document.querySelector('#image-data-input textarea');
                const btnEl = document.querySelector('#detect-api-btn');
                
                if (inputEl && btnEl) {
                    inputEl.value = imageData;
                    inputEl.dispatchEvent(new Event('input', { bubbles: true }));
                    setTimeout(() => btnEl.click(), 100);
                    
                    setTimeout(() => {
                        resumeVideo();
                        isProcessing = false;
                        setStatus('listening-box', '🎤 <b style="color:#22c55e">Listening...</b>');
                        startListening();
                    }, 5000);
                } else {
                    throw new Error('Components not found');
                }
            } catch (e2) {
                setStatus('listening-box', '⚠️ Error: ' + error.message);
                resumeVideo();
                isProcessing = false;
                startListening();
            }
        }
    }
    
    // Initialize
    console.log('🚀 NoonVision starting...');
    await initCamera();
    
    // Play startup audio and start listening
    const startupAudio = $('startup-audio');
    if (startupAudio) {
        startupAudio.play().then(() => {
            startupAudio.onended = () => {
                initSpeech();
                startListening();
            };
        }).catch(() => {
            initSpeech();
            startListening();
        });
    } else {
        initSpeech();
        startListening();
    }
    
    // Global function for manual trigger
    window.triggerDetection = triggerDetection;
}
"""

# Build interface
with gr.Blocks(title="NoonVision", theme=gr.themes.Soft(), css=CSS) as demo:
    
    # Header
    gr.HTML('''
    <div style="text-align:center; padding:25px; background:linear-gradient(135deg,#667eea,#764ba2); color:white; border-radius:12px; margin-bottom:20px;">
        <h1 style="margin:0; font-size:2.5em;">🦾 NoonVision</h1>
        <h2 style="margin:10px 0; font-weight:normal;">Hands-Free AI Vision Assistant</h2>
        <p style="opacity:0.9;">✨ Voice-Activated • No Buttons Required</p>
    </div>
    ''')
    
    # Instructions
    gr.HTML('''
    <div style="background:#ecfdf5; padding:20px; border-radius:12px; margin-bottom:20px; border:2px solid #22c55e;">
        <h3 style="margin-top:0; color:#166534;">🎤 How to Use:</h3>
        <ol style="font-size:1.1em; line-height:1.8;">
            <li><b>Allow</b> camera and microphone when prompted</li>
            <li><b>Say "Detect"</b> to capture and analyze</li>
            <li><b>Listen</b> to the audio results</li>
            <li><b>Repeat</b> - camera resumes automatically!</li>
        </ol>
    </div>
    ''')
    
    # Status displays
    gr.HTML('<div id="listening-box" class="status-box">🎤 Initializing...</div>')
    gr.HTML('<div id="heard-box" class="status-box">🗣️ Waiting for voice command...</div>')
    
    # Main content
    with gr.Row():
        with gr.Column(scale=1):
            # Custom video element
            gr.HTML('''
            <div id="video-box">
                <div id="cam-status">📷 Loading...</div>
                <video id="webcam-video" autoplay playsinline muted></video>
                <img id="frozen-frame" alt="Captured frame">
                <canvas id="hidden-canvas"></canvas>
            </div>
            ''')
        
        with gr.Column(scale=1):
            result_img = gr.Image(type="pil", label="🎯 Detection Results", elem_id="result-image")
            status_text = gr.Textbox(label="Status", value="Ready", lines=2, elem_id="status-text")
            audio_out = gr.Audio(type="filepath", label="🔊 Audio", autoplay=True, elem_id="audio-output")
    
    # API endpoint - this creates /api/detect endpoint
    image_input = gr.Textbox(visible=False, elem_id="image-data-input")
    detect_btn = gr.Button("Detect", visible=False, elem_id="detect-api-btn")
    detect_btn.click(
        fn=detect_from_base64, 
        inputs=image_input, 
        outputs=[result_img, audio_out, status_text],
        api_name="detect"
    )
    
    # Audio elements
    gr.HTML(f'''
    <audio id="startup-audio" preload="auto" src="data:audio/mp3;base64,{startup_base64}"></audio>
    <audio id="processing-audio" preload="auto" src="data:audio/mp3;base64,{processing_base64}"></audio>
    ''')
    
    # Load JavaScript
    demo.load(fn=None, js=JS)
    
    # Footer
    gr.HTML('''
    <div style="text-align:center; color:#666; padding:20px; border-top:1px solid #e5e7eb; margin-top:20px;">
        <p><b>🎯 80+ objects</b> • <b>⚡ 1-2s response</b> • <b>🌐 Chrome/Edge recommended</b></p>
    </div>
    ''')

if __name__ == "__main__":
    demo.launch()