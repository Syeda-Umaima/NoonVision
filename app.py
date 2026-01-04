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

print("=" * 50)
print("🚀 NoonVision Starting...")
print("=" * 50)

CONF_THRESHOLD = 0.30
IMG_SIZE = 640

model = None
try:
    model = YOLO("yolov8m.pt")
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    _ = model(dummy, verbose=False)
    print("✅ YOLOv8m model loaded")
except Exception as e:
    print(f"❌ Model error: {e}")

def make_audio(text):
    try:
        fd, path = tempfile.mkstemp(suffix='.mp3')
        os.close(fd)
        gTTS(text=text, lang='en', slow=False).save(path)
        return path
    except:
        return None

def audio_b64(text):
    path = make_audio(text)
    if path and os.path.exists(path):
        with open(path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        os.remove(path)
        return b64
    return ""

def detect(img):
    print(f"[DETECT] Called with: {type(img)}")
    
    if img is None:
        audio = make_audio("No image captured. Please try again. Say detect.")
        return None, audio, "⚠️ No image"
    
    if model is None:
        audio = make_audio("Model not ready.")
        return None, audio, "⚠️ Model not loaded"
    
    try:
        t0 = time.time()
        
        if hasattr(img, 'convert'):
            pil = img.convert('RGB')
            arr = np.array(pil)
        else:
            arr = np.array(img)
            pil = Image.fromarray(arr).convert('RGB')
        
        print(f"[DETECT] Shape: {arr.shape}")
        
        res = model(arr, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy()
        names = res.names
        
        print(f"[DETECT] Found: {len(boxes)} objects")
        
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except:
            font = ImageFont.load_default()
        
        objects = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = names[int(cls_ids[i])]
            conf = confs[i]
            objects.append(label)
            
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            txt = f"{label} {conf:.0%}"
            tb = draw.textbbox((x1, y1-22), txt, font=font)
            draw.rectangle(tb, fill=(0, 255, 0))
            draw.text((x1, y1-22), txt, fill="black", font=font)
        
        if not objects:
            speech = "I cannot see anything clearly in front of the camera. Try adjusting the camera or lighting."
        else:
            counts = Counter(objects)
            parts = []
            for obj, n in counts.items():
                parts.append(f"{n} {obj}{'s' if n > 1 else ''}" if n > 1 else f"a {obj}")
            
            if len(parts) == 1:
                speech = f"I can see {parts[0]} in front of you."
            elif len(parts) == 2:
                speech = f"I can see {parts[0]} and {parts[1]} in front of you."
            else:
                speech = f"I can see {', '.join(parts[:-1])}, and {parts[-1]} in front of you."
        
        speech += " ... Say detect when ready for next scan."
        audio = make_audio(speech)
        
        dt = time.time() - t0
        status = f"✅ {len(objects)} object(s) in {dt:.1f}s: {', '.join(set(objects)) if objects else 'none'}"
        
        print(f"[DETECT] Done: {status}")
        return pil, audio, status
        
    except Exception as e:
        print(f"[DETECT] Error: {e}")
        import traceback
        traceback.print_exc()
        audio = make_audio("Error. Please try again.")
        return None, audio, f"❌ {e}"

print("🔊 Creating audio...")
STARTUP_B64 = audio_b64("NoonVision ready. Say detect to identify objects. This is completely hands free. No buttons needed.")
PROC_B64 = audio_b64("Processing.")
print("✅ Audio ready")

CSS = """
.status-box{padding:18px;border-radius:12px;margin:12px 0;font-size:18px;font-weight:500}
.listening{background:linear-gradient(90deg,rgba(34,197,94,0.25),rgba(34,197,94,0.1));border-left:5px solid #22c55e;animation:pulse 2s infinite}
.processing{background:linear-gradient(90deg,rgba(59,130,246,0.25),rgba(59,130,246,0.1));border-left:5px solid #3b82f6}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.6}}
#heard{background:#fef9c3;padding:12px;border-radius:8px;text-align:center;border-left:5px solid #eab308;margin:8px 0;font-size:16px}
footer{display:none!important}
#detect-btn{font-size:20px!important;padding:15px 30px!important}
"""

JS = f'''
<audio id="snd-start" src="data:audio/mp3;base64,{STARTUP_B64}"></audio>
<audio id="snd-proc" src="data:audio/mp3;base64,{PROC_B64}"></audio>
<script>
(function(){{

let recog, busy=false, listening=false, started=false;
const triggers = ["detect","what do you see","identify","scan","look","what is","check"];

function hasTrigger(text) {{
    return triggers.some(x => text.toLowerCase().includes(x));
}}

function setStatus(html, cls) {{
    const e = document.getElementById('status-box');
    if(e) {{ e.innerHTML = html; e.className = 'status-box ' + cls; }}
}}

function setHeard(text) {{
    const e = document.getElementById('heard');
    if(e) e.innerHTML = '🗣️ Heard: "' + text + '"';
}}

function sleep(ms) {{
    return new Promise(r => setTimeout(r, ms));
}}

function simulateClick(element) {{
    if (!element) return false;
    
    // Try multiple click methods
    try {{
        // Method 1: Direct click
        element.click();
        
        // Method 2: Dispatch mouse events
        ['mousedown', 'mouseup', 'click'].forEach(eventType => {{
            const event = new MouseEvent(eventType, {{
                view: window,
                bubbles: true,
                cancelable: true,
                buttons: 1
            }});
            element.dispatchEvent(event);
        }});
        
        return true;
    }} catch(e) {{
        console.error('Click failed:', e);
        return false;
    }}
}}

function initRecognition() {{
    if(!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {{
        setStatus('⚠️ Voice not supported. Use Chrome or Edge.', 'processing');
        return false;
    }}
    
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    recog = new SR();
    recog.continuous = true;
    recog.interimResults = false;
    recog.lang = 'en-US';
    
    recog.onstart = () => {{
        listening = true;
        if(!busy) setStatus('🎤 <b>Listening...</b> Say "Detect" to scan', 'listening');
    }};
    
    recog.onresult = (e) => {{
        if(busy) return;
        const text = e.results[e.results.length - 1][0].transcript;
        console.log('🎤 Heard:', text);
        setHeard(text);
        if(hasTrigger(text)) {{
            console.log('🎯 TRIGGER DETECTED!');
            doFullDetection();
        }}
    }};
    
    recog.onerror = (e) => {{
        listening = false;
        if(e.error !== 'no-speech' && e.error !== 'aborted') {{
            setTimeout(startListening, 1000);
        }}
    }};
    
    recog.onend = () => {{
        listening = false;
        if(!busy) setTimeout(startListening, 300);
    }};
    
    return true;
}}

function startListening() {{
    if(!recog && !initRecognition()) return;
    if(!listening && !busy) {{
        try {{ recog.start(); }} catch(e) {{}}
    }}
}}

function stopListening() {{
    if(recog && listening) {{
        try {{ recog.stop(); }} catch(e) {{}}
    }}
}}

// FIND WEBCAM CAPTURE BUTTON - Multiple strategies
function findCaptureButton() {{
    console.log('🔍 Looking for capture button...');
    
    // Log all buttons for debugging
    const allBtns = document.querySelectorAll('button');
    console.log('Total buttons found:', allBtns.length);
    
    // Strategy 1: Find button near video element
    const video = document.querySelector('video');
    if (video) {{
        console.log('Found video element');
        const container = video.closest('div[class*="wrap"], div[class*="container"], div[class*="image"]');
        if (container) {{
            const btns = container.querySelectorAll('button');
            console.log('Buttons in video container:', btns.length);
            for (const btn of btns) {{
                const aria = (btn.getAttribute('aria-label') || '').toLowerCase();
                // Skip clear/remove buttons
                if (aria.includes('remove') || aria.includes('clear') || aria.includes('close')) continue;
                console.log('Found capture button candidate:', btn, 'aria:', aria);
                return btn;
            }}
        }}
    }}
    
    // Strategy 2: Look for button with specific aria labels
    for (const btn of allBtns) {{
        const aria = (btn.getAttribute('aria-label') || '').toLowerCase();
        const title = (btn.getAttribute('title') || '').toLowerCase();
        
        if (aria.includes('capture') || aria.includes('camera') || aria.includes('webcam') ||
            aria.includes('take photo') || aria.includes('photograph') ||
            title.includes('capture') || title.includes('camera')) {{
            console.log('Found by aria/title:', btn);
            return btn;
        }}
    }}
    
    // Strategy 3: Find button with camera SVG icon (not clear/x)
    for (const btn of allBtns) {{
        const aria = (btn.getAttribute('aria-label') || '').toLowerCase();
        if (aria.includes('remove') || aria.includes('clear') || aria.includes('upload')) continue;
        
        const svg = btn.querySelector('svg');
        if (svg) {{
            const parent = btn.closest('[class*="image"], [class*="upload"], [class*="webcam"]');
            if (parent) {{
                console.log('Found SVG button in image area:', btn);
                return btn;
            }}
        }}
    }}
    
    // Strategy 4: First button in image component that's not detect
    const imageComponent = document.querySelector('[class*="image"]');
    if (imageComponent) {{
        const btns = imageComponent.querySelectorAll('button');
        for (const btn of btns) {{
            const text = (btn.innerText || '').toLowerCase();
            const aria = (btn.getAttribute('aria-label') || '').toLowerCase();
            if (!text.includes('detect') && !aria.includes('remove') && !aria.includes('clear')) {{
                console.log('Found first non-detect button:', btn);
                return btn;
            }}
        }}
    }}
    
    console.log('❌ Capture button NOT found');
    return null;
}}

// FIND DETECT BUTTON
function findDetectButton() {{
    console.log('🔍 Looking for detect button...');
    const allBtns = document.querySelectorAll('button');
    
    for (const btn of allBtns) {{
        const text = (btn.innerText || btn.textContent || '').toLowerCase();
        if (text.includes('detect')) {{
            console.log('✅ Found detect button:', btn);
            return btn;
        }}
    }}
    
    // Also try by ID
    const byId = document.getElementById('detect-btn');
    if (byId) {{
        console.log('✅ Found detect button by ID');
        return byId;
    }}
    
    console.log('❌ Detect button NOT found');
    return null;
}}

// FIND CLEAR BUTTON
function findClearButton() {{
    console.log('🔍 Looking for clear button...');
    const allBtns = document.querySelectorAll('button');
    
    // Strategy 1: aria-label contains clear/remove
    for (const btn of allBtns) {{
        const aria = (btn.getAttribute('aria-label') || '').toLowerCase();
        const title = (btn.getAttribute('title') || '').toLowerCase();
        
        if (aria.includes('clear') || aria.includes('remove') || aria.includes('close') ||
            title.includes('clear') || title.includes('remove')) {{
            // Make sure it's in the image area
            const parent = btn.closest('[class*="image"], [class*="upload"]');
            if (parent) {{
                console.log('✅ Found clear button:', btn);
                return btn;
            }}
        }}
    }}
    
    // Strategy 2: Button with × or ✕ text
    for (const btn of allBtns) {{
        const text = (btn.innerText || '').trim();
        if (text === '×' || text === '✕' || text === 'x' || text === '✖') {{
            console.log('✅ Found X button:', btn);
            return btn;
        }}
    }}
    
    // Strategy 3: Small button with SVG in image area
    const imageArea = document.querySelector('[class*="image"]');
    if (imageArea) {{
        const btns = imageArea.querySelectorAll('button');
        for (const btn of btns) {{
            const aria = (btn.getAttribute('aria-label') || '').toLowerCase();
            if (aria.includes('remove') || aria.includes('clear')) {{
                console.log('✅ Found clear in image area:', btn);
                return btn;
            }}
        }}
    }}
    
    console.log('❌ Clear button NOT found');
    return null;
}}

// MAIN DETECTION FUNCTION
async function doFullDetection() {{
    if (busy) {{
        console.log('Already busy, ignoring');
        return;
    }}
    
    busy = true;
    stopListening();
    
    console.log('========================================');
    console.log('🚀 STARTING HANDS-FREE DETECTION');
    console.log('========================================');
    
    // Play processing sound
    const proc = document.getElementById('snd-proc');
    if (proc) {{ proc.currentTime = 0; proc.play().catch(()=>{{}}); }}
    
    // STEP 1: Click capture button
    setStatus('📸 <b>Step 1: Capturing image...</b>', 'processing');
    await sleep(300);
    
    const captureBtn = findCaptureButton();
    if (captureBtn) {{
        console.log('📸 CLICKING CAPTURE BUTTON');
        simulateClick(captureBtn);
    }} else {{
        console.log('⚠️ No capture button - may already have image');
    }}
    
    // Wait for capture
    await sleep(1200);
    
    // STEP 2: Click detect button
    setStatus('🔍 <b>Step 2: Analyzing...</b>', 'processing');
    
    const detectBtn = findDetectButton();
    if (detectBtn) {{
        console.log('🔍 CLICKING DETECT BUTTON');
        simulateClick(detectBtn);
    }} else {{
        console.log('❌ DETECT BUTTON NOT FOUND!');
        setStatus('⚠️ Error: Detect button not found', 'processing');
    }}
    
    // Wait for detection + audio playback
    console.log('⏳ Waiting for detection and audio...');
    await sleep(7000);
    
    // STEP 3: Clear image
    setStatus('🔄 <b>Step 3: Clearing...</b>', 'processing');
    
    const clearBtn = findClearButton();
    if (clearBtn) {{
        console.log('🗑️ CLICKING CLEAR BUTTON');
        simulateClick(clearBtn);
    }} else {{
        console.log('⚠️ Clear button not found');
    }}
    
    // Resume
    await sleep(500);
    busy = false;
    console.log('========================================');
    console.log('✅ READY FOR NEXT DETECTION');
    console.log('========================================');
    setStatus('🎤 <b>Listening...</b> Say "Detect" to scan again', 'listening');
    startListening();
}}

function init() {{
    if (started) return;
    started = true;
    console.log('🚀 NoonVision initializing...');
    
    const startSnd = document.getElementById('snd-start');
    if (startSnd) {{
        startSnd.play().then(() => {{
            startSnd.onended = () => {{
                console.log('✅ Startup audio complete');
                initRecognition();
                startListening();
            }};
        }}).catch(() => {{
            console.log('Audio blocked, starting anyway');
            initRecognition();
            startListening();
        }});
    }} else {{
        initRecognition();
        startListening();
    }}
}}

// Initialize
setTimeout(init, 2500);
document.addEventListener('click', function f() {{
    document.removeEventListener('click', f);
    init();
}});

// Debug functions - can call from console
window.noonvision = {{
    doFullDetection,
    findCaptureButton,
    findDetectButton,
    findClearButton,
    test: function() {{
        console.log('=== TESTING BUTTON DETECTION ===');
        console.log('Capture:', findCaptureButton());
        console.log('Detect:', findDetectButton());
        console.log('Clear:', findClearButton());
    }}
}};

console.log('💡 Debug: Run noonvision.test() in console to check button detection');

}})();
</script>
'''

HEADER = """
<div style="text-align:center;padding:25px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;border-radius:15px;margin-bottom:15px;box-shadow:0 4px 15px rgba(0,0,0,0.2)">
<h1 style="margin:0;font-size:2.5em">🦾 NoonVision</h1>
<p style="margin:8px 0;font-size:1.2em;opacity:0.95">Hands-Free AI Vision Assistant</p>
<p style="margin:0;font-size:0.95em;opacity:0.85">✨ 100% Voice Controlled</p>
</div>

<div style="background:linear-gradient(135deg,#ecfdf5,#d1fae5);padding:20px;border-radius:12px;border:2px solid #22c55e;margin-bottom:15px">
<h3 style="margin:0 0 12px 0;color:#166534">🎤 Just Say "Detect" - Everything is Automatic!</h3>
<p style="margin:0;color:#166534">Voice → Auto Capture → Auto Analyze → Audio Result → Auto Reset</p>
</div>

<div id="status-box" class="status-box listening">🎤 Starting... Please allow camera and microphone.</div>
<div id="heard">🗣️ Waiting for voice command...</div>
"""

with gr.Blocks(css=CSS, title="NoonVision") as demo:
    gr.HTML(JS)
    gr.HTML(HEADER)
    
    with gr.Row():
        with gr.Column():
            inp_img = gr.Image(sources=["webcam"], type="pil", label="📷 Live Camera")
        with gr.Column():
            out_img = gr.Image(type="pil", label="🎯 Detection Results")
            out_status = gr.Textbox(label="Status", value="Ready - Say 'Detect'", interactive=False)
            out_audio = gr.Audio(type="filepath", label="🔊 Audio", autoplay=True)
    
    btn = gr.Button("🔍 Detect Objects", variant="primary", size="lg", elem_id="detect-btn")
    btn.click(fn=detect, inputs=inp_img, outputs=[out_img, out_audio, out_status])
    
    gr.HTML("""
    <div style="text-align:center;color:#666;padding:15px;margin-top:15px;border-top:1px solid #e5e7eb">
        <p><b>🐛 Debug:</b> Open browser console (F12) and run <code>noonvision.test()</code> to check button detection</p>
    </div>
    """)

demo.launch()