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

# Store latest frame globally
latest_frame = None

def update_frame(image):
    """Update the stored frame"""
    global latest_frame
    if image is not None:
        latest_frame = image
    return image

def detect_objects():
    """Detect objects in the latest stored frame"""
    global latest_frame
    
    print(f"[DEBUG] detect called, latest_frame: {type(latest_frame)}")
    
    if latest_frame is None:
        text = "I cannot see anything. Please make sure the camera is active. Say detect when ready."
        return None, generate_audio(text), "⚠️ No image - activate camera first"
    
    if model is None:
        text = "Detection system not ready. Please wait."
        return None, generate_audio(text), "⚠️ Model not loaded"
    
    try:
        start = time.time()
        
        # Use the latest frame
        image = latest_frame
        
        if isinstance(image, Image.Image):
            img_pil = image.copy()
            img_np = np.array(image)
        else:
            img_np = np.array(image) if not isinstance(image, np.ndarray) else image
            img_pil = Image.fromarray(img_np.astype('uint8'))
        
        print(f"[DEBUG] Processing image: {img_np.shape}")
        
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
        
        detected = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = labels[int(class_ids[i])]
            conf = confidences[i]
            detected.append(label)
            
            draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=BOX_WIDTH)
            txt = f"{label} {conf:.2f}"
            bbox = draw.textbbox((x1, y1-25), txt, font=font)
            draw.rectangle(bbox, fill=BOX_COLOR)
            draw.text((x1, y1-25), txt, fill="black", font=font)
        
        # Generate speech
        if not detected:
            speech = "I don't see any recognizable objects. Try adjusting the camera."
        else:
            counts = Counter(detected)
            items = []
            for obj, cnt in counts.items():
                items.append(f"{'a ' + obj if cnt == 1 else str(cnt) + ' ' + obj + 's'}")
            
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
        
        print(f"[DEBUG] Done: {status}")
        return img_pil, audio, status
        
    except Exception as e:
        print(f"[DEBUG] Error: {e}")
        import traceback
        traceback.print_exc()
        text = "Something went wrong. Please try again."
        return None, generate_audio(text), f"❌ Error: {str(e)}"

# Generate startup audio
print("🔊 Generating audio...")
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
print("✅ Ready")

# CSS
CSS = """
.status-listening {
    background: linear-gradient(90deg, rgba(34,197,94,0.2), rgba(34,197,94,0.1));
    padding: 15px; border-radius: 10px; border-left: 4px solid #22c55e;
    animation: pulse 2s infinite; margin: 10px 0;
}
.status-processing {
    background: linear-gradient(90deg, rgba(59,130,246,0.2), rgba(59,130,246,0.1));
    padding: 15px; border-radius: 10px; border-left: 4px solid #3b82f6; margin: 10px 0;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.7} }
#heard-box { background:#fefce8; padding:10px; border-radius:8px; margin:5px 0; text-align:center; }
"""

# Build interface
with gr.Blocks(title="NoonVision", theme=gr.themes.Soft(), css=CSS) as demo:
    
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
        <b>🎤 How to Use:</b> Allow camera & mic → Say "Detect" → Listen → Repeat!
    </div>
    ''')
    
    # Status
    gr.HTML('<div id="status-box" class="status-listening">🎤 Initializing...</div>')
    gr.HTML('<div id="heard-box">🗣️ Waiting for voice...</div>')
    
    # Main layout
    with gr.Row():
        with gr.Column():
            webcam = gr.Image(
                sources=["webcam"], 
                type="pil", 
                label="📷 Camera",
                streaming=True,
                mirror_webcam=True
            )
        
        with gr.Column():
            result_img = gr.Image(type="pil", label="🎯 Results")
            status_txt = gr.Textbox(label="Status", value="Ready", lines=2)
            audio_out = gr.Audio(type="filepath", label="🔊 Audio", autoplay=True)
    
    # Detect button
    detect_btn = gr.Button("🔍 Detect Objects", variant="primary", size="lg", elem_id="detect-btn")
    
    # Stream frames to update latest_frame
    webcam.stream(fn=update_frame, inputs=webcam, outputs=webcam)
    
    # Button triggers detection
    detect_btn.click(fn=detect_objects, inputs=None, outputs=[result_img, audio_out, status_txt])
    
    # JavaScript
    gr.HTML(f'''
    <audio id="startup-audio" src="data:audio/mp3;base64,{startup_b64}"></audio>
    <audio id="processing-audio" src="data:audio/mp3;base64,{processing_b64}"></audio>
    
    <script>
    (function() {{
        let recognition = null;
        let isProcessing = false;
        let isListening = false;
        
        const TRIGGERS = ["detect", "what do you see", "what's in front", "identify", "scan", "look"];
        
        function hasTrigger(t) {{ return TRIGGERS.some(x => t.toLowerCase().includes(x)); }}
        
        function setStatus(h, c) {{
            const e = document.getElementById('status-box');
            if (e) {{ e.innerHTML = h; e.className = c; }}
        }}
        
        function setHeard(t) {{
            const e = document.getElementById('heard-box');
            if (e) e.innerHTML = '🗣️ Heard: "' + t + '"';
        }}
        
        function initSpeech() {{
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {{
                setStatus('⚠️ Use Chrome/Edge for voice', 'status-processing');
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
                const t = e.results[e.results.length-1][0].transcript;
                console.log('Heard:', t);
                setHeard(t);
                if (hasTrigger(t)) doDetect();
            }};
            
            recognition.onerror = (e) => {{
                if (e.error !== 'no-speech' && e.error !== 'aborted') setTimeout(startListening, 1000);
            }};
            
            recognition.onend = () => {{
                isListening = false;
                if (!isProcessing) setTimeout(startListening, 300);
            }};
            return true;
        }}
        
        function startListening() {{
            if (!recognition && !initSpeech()) return;
            if (!isListening && !isProcessing) try {{ recognition.start(); }} catch(e) {{}}
        }}
        
        function stopListening() {{
            if (recognition && isListening) try {{ recognition.stop(); }} catch(e) {{}}
        }}
        
        function doDetect() {{
            if (isProcessing) return;
            isProcessing = true;
            setStatus('🔍 <b>Processing...</b>', 'status-processing');
            stopListening();
            
            const pa = document.getElementById('processing-audio');
            if (pa) {{ pa.currentTime = 0; pa.play().catch(()=>{{}}); }}
            
            setTimeout(() => {{
                const btn = document.getElementById('detect-btn');
                if (btn) btn.click();
                
                setTimeout(() => {{
                    isProcessing = false;
                    setStatus('🎤 <b>Listening...</b> Say "Detect"', 'status-listening');
                    startListening();
                }}, 5000);
            }}, 100);
        }}
        
        function init() {{
            const sa = document.getElementById('startup-audio');
            if (sa) {{
                sa.play().then(() => {{
                    sa.onended = () => {{ initSpeech(); startListening(); }};
                }}).catch(() => {{ initSpeech(); startListening(); }});
            }} else {{ initSpeech(); startListening(); }}
        }}
        
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', () => setTimeout(init, 1500));
        }} else {{ setTimeout(init, 1500); }}
        
        document.addEventListener('click', init, {{ once: true }});
        window.doDetect = doDetect;
    }})();
    </script>
    ''')
    
    gr.HTML('<div style="text-align:center;color:#666;padding:15px;border-top:1px solid #eee;margin-top:15px;"><b>🎯 80+ objects</b> • <b>⚡ 1-2s</b> • Chrome/Edge recommended</div>')

if __name__ == "__main__":
    demo.launch(ssr=False)