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

# Configuration
CONF_THRESHOLD = 0.30
IMG_SIZE = 640
BOX_COLOR = (0, 255, 0)
BOX_WIDTH = 4
FONT_SIZE = 18

print("🚀 Loading NoonVision...")

# Load YOLO model
model = None
try:
    model = YOLO("yolov8m.pt")
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    _ = model(dummy, verbose=False)
    print("✅ YOLOv8m loaded")
except Exception as e:
    print(f"❌ YOLO error: {e}")

def generate_audio(text):
    """Generate TTS audio file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(f.name)
            return f.name
    except Exception as e:
        print(f"Audio error: {e}")
        return None

def save_frame(image, current_state):
    """Save the latest webcam frame to state"""
    if image is not None:
        return image
    return current_state

def detect_objects(stored_image):
    """Main detection function using stored frame"""
    
    print(f"[DEBUG] detect_objects called, image type: {type(stored_image)}")
    
    if stored_image is None:
        print("[DEBUG] No stored image!")
        text = "I cannot see anything. Please make sure the camera is working. Say detect when ready."
        return None, generate_audio(text), "⚠️ No image captured - make sure camera is active"
    
    if model is None:
        text = "Detection system not ready. Please wait and try again."
        return None, generate_audio(text), "⚠️ Model not loaded"
    
    try:
        start = time.time()
        
        # Convert to numpy if needed
        if isinstance(stored_image, Image.Image):
            img_pil = stored_image.copy()
            img_np = np.array(stored_image)
        else:
            img_np = np.array(stored_image) if not isinstance(stored_image, np.ndarray) else stored_image
            img_pil = Image.fromarray(img_np.astype('uint8'))
        
        print(f"[DEBUG] Image shape: {img_np.shape}")
        
        # Run detection
        results = model(img_np, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        
        # Get detections
        boxes = results.boxes.xyxy.cpu().numpy()
        labels = results.names
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        
        # Draw on image
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
        except:
            font = ImageFont.load_default()
        
        detected = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = labels[int(class_ids[i])]
            conf = confidences[i]
            detected.append(label)
            
            draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)
            text_str = f"{label} {conf:.2f}"
            bbox = draw.textbbox((x1, y1-25), text_str, font=font)
            draw.rectangle(bbox, fill=BOX_COLOR)
            draw.text((x1, y1-25), text_str, fill="black", font=font)
        
        # Generate speech
        if not detected:
            speech = "I don't see any recognizable objects. Try moving the camera or improving lighting."
        else:
            counts = Counter(detected)
            items = []
            for obj, count in counts.items():
                if count == 1:
                    items.append(f"a {obj}")
                else:
                    items.append(f"{count} {obj}s")
            
            if len(items) == 1:
                speech = f"I can see {items[0]} in front of you."
            elif len(items) == 2:
                speech = f"I can see {items[0]} and {items[1]} in front of you."
            else:
                speech = f"I can see {', '.join(items[:-1])}, and {items[-1]} in front of you."
        
        speech += " ... Listening. Say detect when ready."
        audio = generate_audio(speech)
        
        elapsed = time.time() - start
        status = f"✅ Found {len(detected)} object(s) in {elapsed:.2f}s"
        if detected:
            status += f": {', '.join(set(detected))}"
        
        print(f"[DEBUG] Detection complete: {status}")
        return img_pil, audio, status
        
    except Exception as e:
        print(f"[DEBUG] Detection error: {e}")
        import traceback
        traceback.print_exc()
        text = "Something went wrong. Please try again. Say detect when ready."
        return None, generate_audio(text), f"❌ Error: {str(e)}"

# Generate startup audio
print("🔊 Generating startup audio...")
startup_audio = generate_audio("NoonVision ready. Say detect to identify objects.")
startup_b64 = ""
if startup_audio and os.path.exists(startup_audio):
    with open(startup_audio, 'rb') as f:
        startup_b64 = base64.b64encode(f.read()).decode()

processing_audio = generate_audio("Processing.")
processing_b64 = ""
if processing_audio and os.path.exists(processing_audio):
    with open(processing_audio, 'rb') as f:
        processing_b64 = base64.b64encode(f.read()).decode()
print("✅ Audio ready")

# CSS
CSS = """
.status-listening {
    background: linear-gradient(90deg, rgba(34,197,94,0.2), rgba(34,197,94,0.1));
    padding: 15px; border-radius: 10px; border-left: 4px solid #22c55e;
    animation: pulse 2s infinite; margin: 10px 0;
}
.status-processing {
    background: linear-gradient(90deg, rgba(59,130,246,0.2), rgba(59,130,246,0.1));
    padding: 15px; border-radius: 10px; border-left: 4px solid #3b82f6;
    margin: 10px 0;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.7} }
#heard-box { background: #fefce8; padding: 10px; border-radius: 8px; margin: 5px 0; text-align: center; }
"""

# Build interface
with gr.Blocks(title="NoonVision", theme=gr.themes.Soft(), css=CSS) as demo:
    
    # State to store latest frame
    frame_state = gr.State(value=None)
    
    # Header
    gr.HTML('''
    <div style="text-align:center; padding:20px; background:linear-gradient(135deg,#667eea,#764ba2); color:white; border-radius:12px; margin-bottom:15px;">
        <h1 style="margin:0;">🦾 NoonVision</h1>
        <p style="margin:5px 0; opacity:0.9;">Hands-Free AI Vision Assistant</p>
    </div>
    ''')
    
    # Instructions
    gr.HTML('''
    <div style="background:#ecfdf5; padding:15px; border-radius:10px; border:2px solid #22c55e; margin-bottom:15px;">
        <b>🎤 How to Use:</b> Allow camera & mic → Say "Detect" → Listen to results → Repeat!
    </div>
    ''')
    
    # Status displays
    gr.HTML('<div id="status-box" class="status-listening">🎤 Initializing...</div>')
    gr.HTML('<div id="heard-box">🗣️ Waiting for voice...</div>')
    
    # Main layout
    with gr.Row():
        with gr.Column():
            webcam = gr.Image(
                sources=["webcam"], 
                type="pil", 
                label="📷 Camera - Live Feed",
                streaming=True,  # Enable streaming to continuously get frames
                mirror_webcam=True
            )
        
        with gr.Column():
            result_img = gr.Image(type="pil", label="🎯 Detection Results")
            status_txt = gr.Textbox(label="Status", value="Ready - Say 'Detect'", lines=2)
            audio_out = gr.Audio(type="filepath", label="🔊 Audio", autoplay=True)
    
    # Detect button
    detect_btn = gr.Button("🔍 Detect Objects", variant="primary", size="lg", elem_id="detect-btn")
    
    # When webcam streams, save latest frame to state
    webcam.stream(
        fn=save_frame,
        inputs=[webcam, frame_state],
        outputs=frame_state
    )
    
    # Detect button uses the stored frame
    detect_btn.click(
        fn=detect_objects,
        inputs=frame_state,
        outputs=[result_img, audio_out, status_txt]
    )
    
    # Audio elements and JavaScript
    gr.HTML(f'''
    <audio id="startup-audio" src="data:audio/mp3;base64,{startup_b64}"></audio>
    <audio id="processing-audio" src="data:audio/mp3;base64,{processing_b64}"></audio>
    
    <script>
    (function() {{
        let recognition = null;
        let isProcessing = false;
        let isListening = false;
        
        const TRIGGERS = ["detect", "what do you see", "what's in front", "identify", "scan", "look"];
        
        function hasTrigger(text) {{
            return TRIGGERS.some(t => text.toLowerCase().includes(t));
        }}
        
        function setStatus(html, cls) {{
            const el = document.getElementById('status-box');
            if (el) {{ el.innerHTML = html; el.className = cls; }}
        }}
        
        function setHeard(text) {{
            const el = document.getElementById('heard-box');
            if (el) el.innerHTML = '🗣️ Heard: "' + text + '"';
        }}
        
        function initSpeech() {{
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {{
                setStatus('⚠️ Use Chrome or Edge for voice commands', 'status-processing');
                return false;
            }}
            
            const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SR();
            recognition.continuous = true;
            recognition.interimResults = false;
            recognition.lang = 'en-US';
            
            recognition.onstart = () => {{
                isListening = true;
                if (!isProcessing) setStatus('🎤 <b>Listening...</b> Say "Detect"', 'status-listening');
            }};
            
            recognition.onresult = (e) => {{
                if (isProcessing) return;
                const text = e.results[e.results.length-1][0].transcript;
                console.log('Heard:', text);
                setHeard(text);
                
                if (hasTrigger(text)) {{
                    doDetect();
                }}
            }};
            
            recognition.onerror = (e) => {{
                console.log('Speech error:', e.error);
                if (e.error !== 'no-speech' && e.error !== 'aborted') {{
                    setTimeout(startListening, 1000);
                }}
            }};
            
            recognition.onend = () => {{
                isListening = false;
                if (!isProcessing) setTimeout(startListening, 300);
            }};
            
            return true;
        }}
        
        function startListening() {{
            if (!recognition && !initSpeech()) return;
            if (!isListening && !isProcessing) {{
                try {{ recognition.start(); }} catch(e) {{}}
            }}
        }}
        
        function stopListening() {{
            if (recognition && isListening) {{
                try {{ recognition.stop(); }} catch(e) {{}}
            }}
        }}
        
        function doDetect() {{
            if (isProcessing) return;
            isProcessing = true;
            
            setStatus('🔍 <b>Processing...</b> Analyzing image...', 'status-processing');
            stopListening();
            
            // Play processing sound
            const pa = document.getElementById('processing-audio');
            if (pa) {{ pa.currentTime = 0; pa.play().catch(()=>{{}}); }}
            
            // Click detect button
            setTimeout(() => {{
                const btn = document.getElementById('detect-btn');
                if (btn) {{
                    btn.click();
                    console.log('✅ Detect button clicked');
                }}
                
                // Resume listening after processing
                setTimeout(() => {{
                    isProcessing = false;
                    setStatus('🎤 <b>Listening...</b> Say "Detect" for next scan', 'status-listening');
                    startListening();
                }}, 5000);
            }}, 100);
        }}
        
        // Initialize
        function init() {{
            console.log('🚀 NoonVision initializing...');
            
            const sa = document.getElementById('startup-audio');
            if (sa) {{
                sa.play().then(() => {{
                    sa.onended = () => {{ initSpeech(); startListening(); }};
                }}).catch(() => {{
                    initSpeech();
                    startListening();
                }});
            }} else {{
                initSpeech();
                startListening();
            }}
        }}
        
        // Start when page loads
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', () => setTimeout(init, 1500));
        }} else {{
            setTimeout(init, 1500);
        }}
        
        // Also init on first click (for audio autoplay)
        document.addEventListener('click', function firstClick() {{
            init();
            document.removeEventListener('click', firstClick);
        }}, {{ once: true }});
        
        window.doDetect = doDetect;
    }})();
    </script>
    ''')
    
    # Footer
    gr.HTML('''
    <div style="text-align:center; color:#666; padding:15px; border-top:1px solid #eee; margin-top:15px;">
        <b>🎯 80+ objects</b> • <b>⚡ 1-2s response</b> • Chrome/Edge recommended
    </div>
    ''')

# Launch without SSR
if __name__ == "__main__":
    demo.launch(ssr_mode=False)