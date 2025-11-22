from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import cv2
import numpy as np
import pyttsx3
import base64
import threading
import speech_recognition as sr

app = FastAPI()

# Load YOLOv8m
model = YOLO("yolov8m.pt")

# Text-to-speech
engine = pyttsx3.init()
engine.setProperty("rate", 150)

def speak(text):
    """Speak automatically without button press."""
    engine.say(text)
    engine.runAndWait()


def listen_for_voice_command():
    """Continuously listen for word 'detect'."""
    recognizer = sr.Recognizer()

    while True:
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                print("Listening for voice command: Say 'detect'...")
                audio = recognizer.listen(source)

            command = recognizer.recognize_google(audio).lower()
            print("Heard:", command)

            if "detect" in command or "what's in front of me" in command:
                print("Voice detected → Capturing image now")
                capture_and_detect()

        except:
            pass


def capture_and_detect():
    """Capture from webcam → Detect → Auto speak."""
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        speak("Camera not working")
        return {"error": "camera failure"}

    # Run YOLO
    results = model(frame)[0]

    # Extract labels & accuracy
    detections = []
    for box in results.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        label = results.names[cls]
        detections.append((label, conf))

    # If nothing found
    if not detections:
        speak("No objects detected in front of you")
        return {"output": "None"}

    # Prepare spoken output without accuracy
    spoken = "Detected objects are: " + ", ".join([d[0] for d in detections])
    speak(spoken)

    # Prepare accuracy text for screen
    accuracy_text = "\n".join([f"{d[0]} ➤ {round(d[1]*100, 2)}%" for d in detections])

    # Encode frame for display
    _, buffer = cv2.imencode(".jpg", frame)
    encoded = base64.b64encode(buffer).decode()

    return {
        "spoken": spoken,
        "accuracy": accuracy_text,
        "image": encoded
    }


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Manual image upload version."""
    image_data = await file.read()
    np_img = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results = model(img)[0]

    detections = []
    for box in results.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        label = results.names[cls]
        detections.append((label, conf))

    if not detections:
        speak("No objects detected in the image")
        return {"output": "None"}

    spoken = "Detected objects are: " + ", ".join([d[0] for d in detections])
    speak(spoken)

    accuracy_text = "\n".join([f"{d[0]} ➤ {round(d[1]*100, 2)}%" for d in detections])

    return {
        "spoken": spoken,
        "accuracy": accuracy_text
    }


@app.get("/")
def home():
    return HTMLResponse("""
    <h2>Vision App Running</h2>
    <p>Say <b>“Detect”</b> to automatically capture image and hear results.</p>
    """)


# Start voice listener in background
threading.Thread(target=listen_for_voice_command, daemon=True).start()
