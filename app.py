# app.py
# FastAPI backend — CPU-optimized, multi-mode YOLO selection

import io
import time
import base64
from typing import Tuple, List, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# CONFIG: Choose MODE HERE
# Options: "ultrafast", "balanced", "accurate"
# "ultrafast" -> yolov8n.pt (best for speed on CPU)
# "balanced"  -> yolov8s.pt
# "accurate"  -> yolov8m.pt
# If you have yolo11n or other small models you may set paths accordingly
# -------------------------
MODEL_MODE = "ultrafast"   # change to "balanced" or "accurate" as needed

# Map modes to model filenames and image sizes
MODEL_MAP = {
    "ultrafast": {"path": "yolov8n.pt", "img_size": 416},
    "balanced":  {"path": "yolov8s.pt", "img_size": 640},
    "accurate":  {"path": "yolov8m.pt", "img_size": 640},
}

if MODEL_MODE not in MODEL_MAP:
    MODEL_MODE = "ultrafast"

MODEL_PATH = MODEL_MAP[MODEL_MODE]["path"]
IMG_SIZE = MODEL_MAP[MODEL_MODE]["img_size"]

# Confidence threshold (can tune)
CONF_THRESHOLD = 0.25

# Load model (CPU)
print(f"Loading model {MODEL_PATH} on CPU (mode={MODEL_MODE}) ...")
try:
    model = YOLO(MODEL_PATH)
    # Force CPU (explicit)
    model.to("cpu")
    # Some speed improvements for CPU:
    try:
        model.fuse()  # fuse conv+bce where possible
    except Exception:
        pass
    # Warm up with a small dummy
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    _ = model(dummy, imgsz=IMG_SIZE, verbose=False)
    print("Model loaded and warmed up.")
except Exception as e:
    print("Failed to load model:", e)
    model = None

# Input model for POST
class ImageData(BaseModel):
    image_base64: str

# Utility: resize while keeping aspect ratio and pad
def preprocess_pil_image(pil_img: Image.Image, target_size: int) -> Tuple[np.ndarray, Image.Image]:
    # Convert to RGB and resize (letterbox)
    img = pil_img.convert("RGB")
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    # create padded canvas
    canvas = Image.new("RGB", (target_size, target_size), (114, 114, 114))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return np.array(canvas), canvas

def format_detections(results, orig_size: Tuple[int, int], pad_info: Tuple[int,int,int,int], conf_threshold=CONF_THRESHOLD):
    # results: Ultralytics results object (boxes)
    # pad_info: (paste_x, paste_y, new_w, new_h) used if needed — in this simplified version we return coarse boxes
    detections = []
    if results is None:
        return detections
    try:
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        names = results.names
        for i, box in enumerate(boxes):
            conf = float(confs[i])
            if conf < conf_threshold:
                continue
            cls = int(cls_ids[i])
            label = names[cls]
            x1, y1, x2, y2 = map(float, box.tolist())
            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "box": [round(x1,2), round(y1,2), round(x2,2), round(y2,2)]
            })
    except Exception:
        pass
    return detections

@app.post("/detect")
async def detect_endpoint(data: ImageData):
    start = time.time()
    if model is None:
        return {"error": "Model not loaded", "detections": [], "speech_text": "Model not available."}
    try:
        # decode base64
        header, b64 = data.image_base64.split(",", 1) if "," in data.image_base64 else ("", data.image_base64)
        img_bytes = base64.b64decode(b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        orig_w, orig_h = pil_img.size

        # preprocess (letterbox)
        in_np, canvas = preprocess_pil_image(pil_img, IMG_SIZE)

        # inference (CPU) — minimal overhead
        # Use model.predict to specify imgsz and device explicitly
        results = model.predict(source=in_np, imgsz=IMG_SIZE, device="cpu", conf=CONF_THRESHOLD, verbose=False)[0]

        detections = format_detections(results, (orig_w, orig_h), (0,0,0,0), conf_threshold=CONF_THRESHOLD)

        # Build speech_text with counts (short and clean)
        if len(detections) == 0:
            speech_text = "No objects detected."
        else:
            # count labels
            labels = [d["label"] for d in detections]
            from collections import Counter
            cnt = Counter(labels)
            parts = []
            for k,v in cnt.items():
                if v == 1:
                    parts.append(f"one {k}")
                else:
                    parts.append(f"{v} {k}s")
            # keep description short
            if len(parts) == 1:
                speech_text = f"I see {parts[0]}."
            elif len(parts) == 2:
                speech_text = f"I see {parts[0]} and {parts[1]}."
            else:
                speech_text = "I see " + ", ".join(parts[:-1]) + ", and " + parts[-1] + "."

        elapsed = time.time() - start
        return {
            "detections": detections,
            "speech_text": speech_text,
            "mode": MODEL_MODE,
            "elapsed": round(elapsed, 3),
            "img_size": IMG_SIZE
        }

    except Exception as e:
        return {"error": str(e), "detections": [], "speech_text": "Error during detection."}
