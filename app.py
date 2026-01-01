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

# ===== CONFIGURATION =====
CONF_THRESHOLD = 0.30
IMG_SIZE = 640

# ===== LOAD YOLO MODEL =====
model = None
try:
    model = YOLO("yolov8m.pt")
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    _ = model(dummy, verbose=False)
    print("✅ YOLOv8m model loaded")
except Exception as e:
    print(f"❌ Model error: {e}")

# ===== AUDIO FUNCTIONS =====
def make_audio(text):
    """Create audio file from text"""
    try:
        fd, path = tempfile.mkstemp(suffix='.mp3')
        os.close(fd)
        gTTS(text=text, lang='en', slow=False).save(path)
        return path
    except:
        return None

def audio_to_b64(text):
    """Create audio and convert to base64"""
    path = make_audio(text)
    if path and os.path.exists(path):
        with open(path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        os.remove(path)
        return b64
    return ""

# ===== DETECTION FUNCTION =====
def detect(img):
    """Run object detection on image"""
    print(f"\n[DETECT] Input type: {type(img)}")
    
    if img is None:
        print("[DETECT] No image!")
        audio = make_audio("No image captured. Please click the camera to take a photo first, then say detect.")
        return None, audio, "⚠️ No image - click camera first"
    
    if model is None:
        audio = make_audio("Model not ready. Please wait.")
        return None, audio, "⚠️ Model not loaded"
    
    try:
        t0 = time.time()
        
        # Convert to numpy
        if hasattr(img, 'convert'):
            pil_img = img.convert('RGB')
            np_img = np.array(pil_img)
        else:
            np_img = np.array(img)
            pil_img = Image.fromarray(np_img).convert('RGB')
        
        print(f"[DETECT] Shape: {np_img.shape}")
        
        # Run YOLO
        results = model(np_img, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy()
        names = results.names
        
        print(f"[DETECT] Found {len(boxes)} objects")
        
        # Draw boxes
        draw = ImageDraw.Draw(pil_img)
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
        
        # Create speech
        if not objects:
            speech = "I don't see any objects. Try moving closer or adjusting lighting."
        else:
            counts = Counter(objects)
            parts = []
            for obj, n in counts.items():
                parts.append(f"{n} {obj}{'s' if n > 1 else ''}" if n > 1 else f"a {obj}")
            
            if len(parts) == 1:
                speech = f"I can see {parts[0]} in front of you."
            elif len(parts) == 2:
                speech = f"I can see {parts[0]} and {parts[1]}."
            else:
                speech = f"I can see {', '.join(parts[:-1])}, and {parts[-1]}."
        
        speech += " Listening. Say detect when ready."
        audio = make_audio(speech)
        
        dt = time.time() - t0
        status = f"✅ {len(objects)} object(s) in {dt:.1f}s"
        if objects:
            status += f": {', '.join(set(objects))}"
        
        print(f"[DETECT] Done: {status}")
        return pil_img, audio, status
        
    except Exception as e:
        print(f"[DETECT] Error: {e}")
        import traceback
        traceback.print_exc()
        audio = make_audio("Error occurred. Please try again.")
        return None, audio, f"❌ {e}"

# ===== STARTUP AUDIO =====
print("🔊 Creating audio...")
STARTUP_B64 = audio_to_b64("NoonVision ready. Say detect to identify objects.")
PROCESSING_B64 = audio_to_b64("Processing.")
print("✅ Audio ready")

# ===== CSS =====
CSS = """
.status-box { padding:15px; border-radius:10px; margin:10px 0; font-size:16px; }
.listening { background:rgba(34,197,94,0.15); border-left:4px solid #22c55e; animation:pulse 2s infinite; }
.processing { background:rgba(59,130,246,0.15); border-left:4px solid #3b82f6; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.7} }
#heard { background:#fefce8; padding:10px; border-radius:8px; text-align:center; border-left:4px solid #eab308; margin:5px 0; }
"""

# ===== JAVASCRIPT =====
JS = f"""
<audio id="snd-start" src="data:audio/mp3;base64,{STARTUP_B64}"></audio>
<audio id="snd-proc" src="data:audio/mp3;base64,{PROCESSING_B64}"></audio>
<script>
(function(){{
  let recog=null, busy=false, listening=false, started=false;
  const triggers=["detect","what do you see","what's in front","identify","scan","look"];
  
  function hasTrig(t){{ return triggers.some(x=>t.toLowerCase().includes(x)); }}
  
  function setBox(h,c){{
    const e=document.getElementById('status-box');
    if(e){{ e.innerHTML=h; e.className='status-box '+c; }}
  }}
  
  function setHeard(t){{
    const e=document.getElementById('heard');
    if(e) e.innerHTML='🗣️ Heard: "'+t+'"';
  }}
  
  function initRec(){{
    if(!('webkitSpeechRecognition' in window)&&!('SpeechRecognition' in window)){{
      setBox('⚠️ Use Chrome or Edge for voice','processing');
      return false;
    }}
    const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
    recog=new SR();
    recog.continuous=true;
    recog.interimResults=false;
    recog.lang='en-US';
    
    recog.onstart=()=>{{
      listening=true;
      if(!busy) setBox('🎤 <b>Listening...</b> Say "Detect"','listening');
    }};
    
    recog.onresult=(e)=>{{
      if(busy)return;
      const t=e.results[e.results.length-1][0].transcript;
      console.log('Heard:',t);
      setHeard(t);
      if(hasTrig(t)) doDetect();
    }};
    
    recog.onerror=(e)=>{{
      console.log('Recog error:',e.error);
      listening=false;
      if(e.error!=='no-speech'&&e.error!=='aborted') setTimeout(startRec,1000);
    }};
    
    recog.onend=()=>{{
      listening=false;
      if(!busy) setTimeout(startRec,300);
    }};
    
    return true;
  }}
  
  function startRec(){{
    if(!recog&&!initRec())return;
    if(!listening&&!busy){{
      try{{ recog.start(); }}catch(e){{}}
    }}
  }}
  
  function stopRec(){{
    if(recog&&listening){{
      try{{ recog.stop(); }}catch(e){{}}
    }}
  }}
  
  function doDetect(){{
    if(busy)return;
    busy=true;
    setBox('🔍 <b>Processing...</b>','processing');
    stopRec();
    
    // Play processing sound
    const ps=document.getElementById('snd-proc');
    if(ps){{ ps.currentTime=0; ps.play().catch(()=>{{}}); }}
    
    // Click button after short delay
    setTimeout(()=>{{
      // Find button by looking for primary/detect button
      let btn=null;
      const buttons=document.querySelectorAll('button');
      for(const b of buttons){{
        const txt=(b.innerText||b.textContent||'').toLowerCase();
        if(txt.includes('detect')){{
          btn=b;
          break;
        }}
      }}
      
      if(btn){{
        console.log('Clicking button');
        btn.click();
      }}else{{
        console.log('Button not found');
      }}
      
      // Resume after audio plays
      setTimeout(()=>{{
        busy=false;
        setBox('🎤 <b>Listening...</b> Say "Detect"','listening');
        startRec();
      }},6000);
    }},300);
  }}
  
  function init(){{
    if(started)return;
    started=true;
    console.log('NoonVision init');
    
    const sa=document.getElementById('snd-start');
    if(sa){{
      sa.play().then(()=>{{
        sa.onended=()=>{{ initRec(); startRec(); }};
      }}).catch(()=>{{ initRec(); startRec(); }});
    }}else{{
      initRec(); startRec();
    }}
  }}
  
  // Start on load
  setTimeout(init,1500);
  
  // Also on click for browsers that block autoplay
  document.addEventListener('click',function f(){{
    document.removeEventListener('click',f);
    init();
  }});
  
  window.doDetect=doDetect;
}})();
</script>
"""

# ===== BUILD UI =====
with gr.Blocks(title="NoonVision", theme=gr.themes.Soft(), css=CSS) as demo:
    
    gr.HTML(JS)
    
    gr.HTML("""
    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#667eea,#764ba2);color:white;border-radius:12px;margin-bottom:15px">
        <h1 style="margin:0">🦾 NoonVision</h1>
        <p style="margin:5px 0;opacity:0.9">Hands-Free AI Vision Assistant</p>
    </div>
    """)
    
    gr.HTML("""
    <div style="background:#ecfdf5;padding:15px;border-radius:10px;border:2px solid #22c55e;margin-bottom:15px">
        <b>🎤 How to Use:</b> Allow camera & mic → Say "Detect" → Listen → Repeat!
    </div>
    """)
    
    gr.HTML('<div id="status-box" class="status-box listening">🎤 Starting up...</div>')
    gr.HTML('<div id="heard">🗣️ Waiting for voice...</div>')
    
    with gr.Row():
        with gr.Column():
            cam = gr.Image(sources=["webcam"], type="pil", label="📷 Camera")
        with gr.Column():
            out_img = gr.Image(type="pil", label="🎯 Results")
            out_status = gr.Textbox(label="Status", value="Ready")
            out_audio = gr.Audio(type="filepath", label="🔊 Audio", autoplay=True)
    
    btn = gr.Button("🔍 Detect Objects", variant="primary", size="lg")
    btn.click(fn=detect, inputs=[cam], outputs=[out_img, out_audio, out_status])
    
    gr.HTML('<div style="text-align:center;color:#666;padding:10px;margin-top:10px">🎯 80+ objects • ⚡ Fast • 🌐 Chrome/Edge</div>')

if __name__ == "__main__":
    demo.launch()