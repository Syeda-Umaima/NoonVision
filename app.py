import base64
import io
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("yolov8m.pt")

class ImageData(BaseModel):
    image_base64: str

@app.post("/detect")
async def detect_objects(data: ImageData):
    img_bytes = base64.b64decode(data.image_base64.split(",")[1])
    img = Image.open(io.BytesIO(img_bytes))
    
    results = model.predict(img)[0]

    detections = []
    spoken_list = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = results.names[cls_id]
        conf = float(box.conf[0])
        detections.append({"label": label, "confidence": round(conf, 2)})
        spoken_list.append(label)

    if len(spoken_list) == 0:
        speech = "No objects detected."
    else:
        speech = "Detected objects are: " + ", ".join(spoken_list)

    return {
        "detections": detections,
        "speech_text": speech
    }
