import gradio as gr
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import os
from transformers import pipeline

# --- GLOBAL SETUP AND MODEL LOADING ---
# Load YOLOv8 medium model for better detection accuracy (Member 3 choice)
model = YOLO("yolov8m.pt")

# Load a small ASR model for fast transcription (Member 3 choice)
# The 'transformers' library handles model downloading automatically.
try:
    stt_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")
except Exception as e:
    stt_pipe = None
    print(f"STT model failed to load: {e}")

# --- DETECTION AND AUDIO GENERATION FUNCTION ---
def detect_objects_with_voice(image):
    """
    Performs object detection, annotates the image, prepares the text summary,
    and generates gTTS audio output.
    """
    # Use a faster, slightly less sensitive detection setting for real-time (conf=0.15)
    conf_threshold = 0.15 

    if image is None:
        return None, "No image provided.", None

    img_np = np.array(image)
    # Run inference with a defined confidence threshold
    results = model(img_np, conf=conf_threshold)[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    labels = results.names
    confidences = results.boxes.conf.cpu().numpy()

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    detected_objects = []
    detected_objects_audio = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cls_id = int(results.boxes.cls[i])
        label = labels[cls_id]
        conf = confidences[i]

        detected_objects.append(f"{label}: {conf:.2f}")
        detected_objects_audio.append(label)

        # Draw bounding box + confidence
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3) # Increased width for visibility
        draw.text((x1, y1 - 15), f"{label} {conf:.2f}", fill="red", font=font)

    # --- Prepare description and TTS audio ---
    description_text = "\n".join(detected_objects) if detected_objects else "No objects detected."
    
    audio_file = "detected_objects.mp3"
    
    if detected_objects_audio:
        # Generate spoken description
        tts_text = "I see the following objects: " + ", ".join(detected_objects_audio)
        tts = gTTS(text=tts_text, lang='en')
        tts.save(audio_file)
    else:
        # Generate spoken message for no objects
        tts = gTTS(text="I couldn't detect any common objects. Please try again.", lang='en')
        tts.save(audio_file)
        
    return image, description_text, audio_file

# --- SPEECH-TO-TEXT AND TRIGGER FUNCTION (Member 3/4) ---
def transcribe_audio_and_check_trigger(audio_file_path):
    """
    Transcribes audio and checks for trigger phrases to initiate detection.
    Returns: (transcribed_text, trigger_button_click_signal_bool)
    """
    if not stt_pipe or audio_file_path is None:
        return "STT service unavailable or no audio recorded.", False
    
    # Gradio passes the file path to the recorded audio
    result = stt_pipe(audio_file_path)
    transcribed_text = result["text"].strip().lower()
    
    # Define trigger phrases for automatic capture
    is_trigger = False
    if "detect" in transcribed_text or "what is in front of me" in transcribed_text or "what's this" in transcribed_text:
        is_trigger = True
    
    if is_trigger:
        status_message = f"✅ Command recognized: '{transcribed_text}'. Capturing image..."
    else:
        status_message = f"❌ Command not recognized: '{transcribed_text}'. Say 'Detect' or 'What is in front of me'."
        
    # The return boolean triggers the hidden button's click event
    return status_message, is_trigger

# --- GRADIO BLOCKS INTERFACE (Member 1/2/4) ---
with gr.Blocks(title="NoonVision – Voice-Activated Detection") as demo:
    gr.Markdown("## 🎤 NoonVision – Voice-Activated Object Identifier")
    gr.Markdown(
        """
        **Instructions:** Click the microphone 🎙️ below and say a command like **'Detect'** or **'What is in front of me?'** The system will automatically capture the current frame from the webcam, run the detection, and speak the results.
        """
    )
    
    # Hidden button used as a signal/trigger to initiate the image capture and detection flow
    detection_trigger_btn = gr.Button("Auto Detect Trigger", visible=False)
    
    # 1. Main Display and Input Area
    with gr.Row():
        
        # Column 1: Camera Input (M2) and Detection Output (M4)
        with gr.Column(scale=2):
            # The 'streaming=True' is crucial for showing a live feed
            image_input = gr.Image(type="pil", source="webcam", streaming=True, label="Live Webcam Feed")
            image_output = gr.Image(type="pil", label="Annotated Detection Result")
            
        # Column 2: Voice Input and Text/Audio Results (M1/M6)
        with gr.Column(scale=1):
            
            gr.Markdown("### 🔊 Voice Command")
            voice_input = gr.Audio(sources=["microphone"], type="filepath", label="Record your command here")
            
            transcribed_text_output = gr.Textbox(label="Status / Transcribed Command")

            gr.Markdown("### 💬 Results")
            text_output = gr.Textbox(lines=5, label="Object Details (for helpers)")
            audio_output = gr.Audio(type="filepath", autoplay=True, label="Spoken Objects")

    # --- Event Wiring (M4's Critical Task) ---
    
    # 1. Voice Input -> Transcription/Trigger Check
    voice_input.change(
        fn=transcribe_audio_and_check_trigger,
        inputs=[voice_input],
        # The output True/False signal goes to the hidden button to trigger it
        outputs=[transcribed_text_output, detection_trigger_btn], 
        show_progress=True
    )

    # 2. Hidden Button Click -> Detection (This runs the entire ML pipeline)
    detection_trigger_btn.click(
        fn=detect_objects_with_voice,
        inputs=[image_input],
        outputs=[image_output, text_output, audio_output]
    )

demo.launch()