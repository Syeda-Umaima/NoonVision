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
        fd, filepath = tempfile.mkstemp(suffix='.mp3')
        os.close(fd)
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filepath)
        return filepath
    except Exception as e:
        print(f"Audio error: {e}")
        return None

def detect_objects(image):
    """Main detection function"""
    print(f"[DEBUG] detect_objects called")
    
    if image is None:
        print("[DEBUG] No image")
        text = "I cannot see anything. Please make sure the camera is working. Say detect when ready."
        return None, generate_audio(text), "⚠️ No image"
    
    if model is None:
        text = "Detection system not ready. Please wait."
        return None, generate_audio(text), "⚠️ Model not loaded"
    
    try:
        start = time.time()
        
        # Convert image
        if isinstance(image, Image.Image):
            img_pil = image.copy()
            img_np = np.array(image)
        else:
            img_np = image
            img_pil = Image.fromarray(image)
        
        print(f"[DEBUG] Image shape: {img_np.shape}")
        
        # Run YOLO
        results = model(img_np, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        labels = results.names
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        
        print(f"[DEBUG] Found {len(boxes)} objects")
        
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
            speech = "I don't see any recognizable objects. Try moving the camera."
        else:
            counts = Counter(detected)
            items = []
            for obj, count in counts.items():
                items.append(f"{count} {obj}{'s' if count > 1 else ''}" if count > 1 else f"a {obj}")
            
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
print("✅ Audio ready")

# CSS
custom_css = """
.status-box {
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    text-align: center;
    font-size: 16px;
}
.listening {
    background: linear-gradient(90deg, rgba(34,197,94,0.2), rgba(34,197,94,0.1));
    border-left: 4px solid #22c55e;
    animation: pulse 2s infinite;
}
.processing {
    background: linear-gradient(90deg, rgba(59,130,246,0.2), rgba(59,130,246,0.1));
    border-left: 4px solid #3b82f6;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}
#heard-text {
    background: #fefce8;
    padding: 10px;
    border-radius: 8px;
    text-align: center;
    margin: 5px 0;
}
"""

# JavaScript
voice_js = f"""
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
        if (el) {{
            el.innerHTML = html;
            el.className = 'status-box ' + cls;
        }}
    }}
    
    function setHeard(text) {{
        const el = document.getElementById('heard-text');
        if (el) el.innerHTML = '🗣️ Heard: "' + text + '"';
    }}
    
    function initSpeech() {{
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {{
            setStatus('⚠️ Use Chrome/Edge for voice', 'processing');
            return false;
        }}
        
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SR();
        recognition.continuous = true;
        recognition.interimResults = false;
        recognition.lang = 'en-US';
        
        recognition.onstart = () => {{
            isListening = true;
            if (!isProcessing) setStatus('🎤 <b>Listening...</b> Say "Detect"', 'listening');
        }};
        
        recognition.onresult = (e) => {{
            if (isProcessing) return;
            const text = e.results[e.results.length-1][0].transcript;
            console.log('Heard:', text);
            setHeard(text);
            if (hasTrigger(text)) doDetect();
        }};
        
        recognition.onerror = (e) => {{
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
        
        setStatus('🔍 <b>Processing...</b>', 'processing');
        stopListening();
        
        // Play sound
        const pa = document.getElementById('processing-audio');
        if (pa) {{ pa.currentTime = 0; pa.play().catch(()=>{{}}); }}
        
        // Find and click button
        setTimeout(() => {{
            // Find all buttons and look for the Detect one
            const buttons = document.querySelectorAll('button');
            for (const btn of buttons) {{
                if (btn.innerText && btn.innerText.includes('Detect')) {{
                    console.log('Clicking detect button');
                    btn.click();
                    break;
                }}
            }}
            
            // Resume after delay
            setTimeout(() => {{
                isProcessing = false;
                setStatus('🎤 <b>Listening...</b> Say "Detect"', 'listening');
                startListening();
            }}, 6000);
        }}, 200);
    }}
    
    function init() {{
        console.log('NoonVision init');
        const sa = document.getElementById('startup-audio');
        if (sa) {{
            sa.play().then(() => {{
                sa.onended = () => {{ initSpeech(); startListening(); }};
            }}).catch(() => {{ initSpeech(); startListening(); }});
        }} else {{
            initSpeech(); startListening();
        }}
    }}
    
    // Start
    setTimeout(init, 1500);
    document.addEventListener('click', function once() {{
        document.removeEventListener('click', once);
        if (!recognition) init();
    }});
    
    window.doDetect = doDetect;
}})();
</script>
"""

# Build interface
with gr.Blocks(title="NoonVision", theme=gr.themes.Soft(), css=custom_css) as demo:
    
    # Audio + JS
    gr.HTML(voice_js)
    
    # Header
    gr.HTML('''
    <div style="text-align:center; padding:20px; background:linear-gradient(135deg,#667eea,#764ba2); color:white; border-radius:12px; margin-bottom:15px;">
        <h1 style="margin:0;">🦾 NoonVision</h1>
        <p style="margin:5px 0;">Hands-Free AI Vision Assistant</p>
    </div>
    ''')
    
    # Instructions
    gr.HTML('''
    <div style="background:#ecfdf5; padding:15px; border-radius:10px; border:2px solid #22c55e; margin-bottom:15px;">
        <b>🎤 How to Use:</b> Allow camera & mic → Say "Detect" → Listen → Repeat!
    </div>
    ''')
    
    # Status
    gr.HTML('<div id="status-box" class="status-box listening">🎤 Initializing...</div>')
    gr.HTML('<div id="heard-text">🗣️ Waiting for voice...</div>')
    
    # Main layout
    with gr.Row():
        with gr.Column():
            webcam = gr.Image(sources=["webcam"], type="pil", label="📷 Camera")
        with gr.Column():
            result_img = gr.Image(type="pil", label="🎯 Results")
            status_txt = gr.Textbox(label="Status", value="Ready")
            audio_out = gr.Audio(type="filepath", label="🔊 Audio", autoplay=True)
    
    # Button
    detect_btn = gr.Button("🔍 Detect Objects", variant="primary")
    detect_btn.click(fn=detect_objects, inputs=webcam, outputs=[result_img, audio_out, status_txt])
    
    # Footer
    gr.HTML('<div style="text-align:center; color:#666; padding:10px;">🎯 80+ objects • ⚡ 1-2s • Chrome/Edge</div>')

if __name__ == "__main__":
    demo.launch(share=False)