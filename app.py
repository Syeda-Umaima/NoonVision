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

print("🚀 Loading NoonVision...")

# Load model
model = None
try:
    model = YOLO("yolov8m.pt")
    model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    print("✅ YOLOv8m loaded")
except Exception as e:
    print(f"❌ YOLO error: {e}")

# Global frame storage
current_frame = None

def make_audio(text):
    try:
        f = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        gTTS(text=text, lang='en').save(f.name)
        return f.name
    except:
        return None

def store_frame(img):
    global current_frame
    if img is not None:
        current_frame = img
    return img

def detect():
    global current_frame
    
    if current_frame is None:
        return None, make_audio("No image. Check camera. Say detect when ready."), "⚠️ No image"
    
    if model is None:
        return None, make_audio("Model not ready."), "⚠️ No model"
    
    try:
        img = current_frame
        if isinstance(img, Image.Image):
            arr = np.array(img)
            pil = img.copy()
        else:
            arr = img
            pil = Image.fromarray(img)
        
        res = model(arr, imgsz=640, conf=0.3, verbose=False)[0]
        
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        found = []
        for i, box in enumerate(res.boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            label = res.names[int(res.boxes.cls[i])]
            conf = float(res.boxes.conf[i])
            found.append(label)
            draw.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=3)
            draw.text((x1, y1-20), f"{label} {conf:.0%}", fill=(0,255,0), font=font)
        
        if not found:
            speech = "I don't see any objects. Try adjusting the camera."
        else:
            counts = Counter(found)
            parts = [f"{c} {o}{'s' if c>1 else ''}" for o,c in counts.items()]
            speech = "I see " + (" and ".join(parts) if len(parts)<=2 else ", ".join(parts[:-1])+", and "+parts[-1])
        
        speech += ". Say detect when ready."
        
        return pil, make_audio(speech), f"✅ Found: {', '.join(set(found)) or 'nothing'}"
    
    except Exception as e:
        return None, make_audio("Error. Try again."), f"❌ {e}"

# Startup audio
print("🔊 Making audio...")
startup = make_audio("NoonVision ready. Say detect.")
startup_b64 = ""
if startup and os.path.exists(startup):
    with open(startup, 'rb') as f:
        startup_b64 = base64.b64encode(f.read()).decode()
proc = make_audio("Processing.")
proc_b64 = ""
if proc and os.path.exists(proc):
    with open(proc, 'rb') as f:
        proc_b64 = base64.b64encode(f.read()).decode()
print("✅ Ready")

# UI
with gr.Blocks(title="NoonVision", theme=gr.themes.Soft()) as demo:
    gr.HTML('''<div style="text-align:center;padding:20px;background:linear-gradient(135deg,#667eea,#764ba2);color:white;border-radius:12px;margin-bottom:15px">
        <h1>🦾 NoonVision</h1><p>Say "Detect" to identify objects</p></div>''')
    
    gr.HTML('<div id="status" style="padding:15px;background:#d1fae5;border-radius:10px;border-left:4px solid #22c55e;margin:10px 0">🎤 Initializing...</div>')
    gr.HTML('<div id="heard" style="padding:10px;background:#fef9c3;border-radius:8px;margin:5px 0;text-align:center">🗣️ Waiting...</div>')
    
    with gr.Row():
        cam = gr.Image(sources=["webcam"], type="pil", label="📷 Camera", streaming=True, mirror_webcam=True)
        with gr.Column():
            out_img = gr.Image(type="pil", label="🎯 Results")
            out_status = gr.Textbox(label="Status", value="Ready")
            out_audio = gr.Audio(type="filepath", label="🔊", autoplay=True)
    
    btn = gr.Button("🔍 Detect", variant="primary", elem_id="detect-btn")
    
    cam.stream(store_frame, cam, cam)
    btn.click(detect, None, [out_img, out_audio, out_status])
    
    gr.HTML(f'''
    <audio id="startup" src="data:audio/mp3;base64,{startup_b64}"></audio>
    <audio id="proc" src="data:audio/mp3;base64,{proc_b64}"></audio>
    <script>
    (()=>{{
        let rec,proc=false,list=false;
        const trig=["detect","what do you see","identify","scan","look"];
        const has=t=>trig.some(x=>t.toLowerCase().includes(x));
        const stat=(h,c)=>{{const e=document.getElementById("status");if(e){{e.innerHTML=h;e.style.background=c}}}}
        const heard=t=>{{const e=document.getElementById("heard");if(e)e.innerHTML='🗣️ "'+t+'"'}}
        
        function initS(){{
            if(!('webkitSpeechRecognition'in window)&&!('SpeechRecognition'in window))return false;
            const S=window.SpeechRecognition||window.webkitSpeechRecognition;
            rec=new S();rec.continuous=true;rec.interimResults=false;rec.lang='en-US';
            rec.onstart=()=>{{list=true;if(!proc)stat('🎤 <b>Listening...</b> Say "Detect"','#d1fae5')}};
            rec.onresult=e=>{{if(proc)return;const t=e.results[e.results.length-1][0].transcript;heard(t);if(has(t))doD()}};
            rec.onerror=e=>{{if(e.error!='no-speech'&&e.error!='aborted')setTimeout(startL,1000)}};
            rec.onend=()=>{{list=false;if(!proc)setTimeout(startL,300)}};
            return true
        }}
        function startL(){{if(!rec&&!initS())return;if(!list&&!proc)try{{rec.start()}}catch(e){{}}}}
        function stopL(){{if(rec&&list)try{{rec.stop()}}catch(e){{}}}}
        function doD(){{
            if(proc)return;proc=true;
            stat('🔍 <b>Processing...</b>','#dbeafe');stopL();
            document.getElementById('proc')?.play().catch(()=>{{}});
            setTimeout(()=>{{document.getElementById('detect-btn')?.click();setTimeout(()=>{{proc=false;stat('🎤 <b>Listening...</b>','#d1fae5');startL()}},5000)}},100)
        }}
        function init(){{document.getElementById('startup')?.play().then(()=>{{setTimeout(()=>{{initS();startL()}},2000)}}).catch(()=>{{initS();startL()}})}}
        setTimeout(init,1500);document.addEventListener('click',init,{{once:true}});window.doD=doD
    }})()
    </script>''')

demo.launch()