import gradio as gr
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import os
from transformers import pipeline

# --- GLOBAL SETUP AND MODEL LOADING ---
# Load YOLOv8 medium model for good balance of speed and accuracy
model = YOLO("yolov8m.pt")

# Load a small ASR model for fast transcription
try:
    stt_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")
except Exception as e:
    stt_pipe = None
    print(f"STT model failed to load: {e}")

# --- DETECTION AND AUDIO GENERATION FUNCTION ---
def detect_objects_with_voice(image, status_message):
    """
    Performs object detection, annotates the image, and generates gTTS audio output.
    Takes status_message to fulfill the function signature and update status.
    """
    # Using a slightly lower threshold to catch small objects
    conf_threshold = 0.15 

    if image is None:
        # Returns the input image untouched if no image provided
        return None, None, "No image provided.", None, "Error: No image input."

    img_np = np.array(image)
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
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 15), f"{label} {conf:.2f}", fill="red", font=font)

    # --- Prepare description and TTS audio ---
    description_text = "\n".join(detected_objects) if detected_objects else "No objects detected."
    
    audio_file = "detected_objects.mp3"
    
    if detected_objects_audio:
        tts_text = "I see the following objects: " + ", ".join(detected_objects_audio)
    else:
        tts_text = "I couldn't detect any common objects. Please try again."
        
    tts = gTTS(text=tts_text, lang='en')
    tts.save(audio_file)
        
    # Return: (original_image_for_pass_through, annotated_image, description_text, audio_file, final_status)
    return image, image, description_text, audio_file, "Detection complete. Results are spoken."

# --- SPEECH-TO-TEXT AND TRIGGER FUNCTION ---
def transcribe_audio_and_check_trigger(audio_file_path, image_input):
    """
    Transcribes audio and initiates the detect_objects_with_voice function directly 
    if a trigger phrase is found.
    """
    # Start with a default status message
    default_status = "Waiting for command..."
    
    if not stt_pipe or audio_file_path is None:
        return image_input, None, "STT service unavailable or no audio recorded.", None, "Error: STT not ready."

    result = stt_pipe(audio_file_path)
    transcribed_text = result["text"].strip().lower()
    
    # Define trigger phrases
    is_trigger = False
    if "detect" in transcribed_text or "what is in front of me" in transcribed_text or "what's this" in transcribed_text:
        is_trigger = True
    
    if is_trigger:
        status_message = f"✅ Command recognized: '{transcribed_text}'. Capturing image..."
        # If triggered, call the main detection function directly, passing the image input
        # NOTE: The outputs of this call are the final results
        return detect_objects_with_voice(image_input, status_message)
    else:
        status_message = f"❌ Command not recognized: '{transcribed_text}'. Say 'Detect' or 'What is in front of me'."
        # If not triggered, clear results and update status
        # Returns the image input and clears the output images/audio
        return image_input, None, status_message, None, default_status

# --- GRADIO BLOCKS INTERFACE ---
with gr.Blocks(title="NoonVision – Voice-Activated Detection") as demo:
    gr.Markdown("## 🎤 NoonVision – Voice-Activated Object Identifier")
    gr.Markdown(
        """
        **Instructions:** Simply **click the microphone button** and say a command like **'Detect'** or **'What is in front of me?'** The system will automatically capture the current frame, run detection, and speak the results.
        """
    )
    
    # 1. Main Display and Input Area
    with gr.Row():
        
        # Column 1: Camera Input and Detection Output
        with gr.Column(scale=2):
            # Input: Live webcam feed
            image_input = gr.Image(type="pil", source="webcam", streaming=True, label="Live Webcam Feed")
            # Output: Annotated image
            image_output = gr.Image(type="pil", label="Annotated Detection Result")
            
        # Column 2: Voice Input and Text/Audio Results
        with gr.Column(scale=1):
            
            gr.Markdown("### 🔊 Voice Command")
            # Input: Microphone
            voice_input = gr.Audio(sources=["microphone"], type="filepath", label="Click to Record Command")
            
            # Status Box (shows transcription/trigger status)
            transcribed_text_output = gr.Textbox(label="Status / Transcribed Command")

            gr.Markdown("### 💬 Results")
            # Output: Text summary
            text_output = gr.Textbox(lines=5, label="Object Details (for helpers)")
            # Output: Spoken audio
            audio_output = gr.Audio(type="filepath", autoplay=True, label="Spoken Objects")

    # --- Event Wiring ---
    
    # The microphone's 'change' event fires when recording stops.
    voice_input.change(
        fn=transcribe_audio_and_check_trigger,
        inputs=[voice_input, image_input],
        # The outputs map to the function's return values:
        # [image_input (passes current frame), image_output, transcribed_text_output, audio_output, text_output]
        outputs=[image_input, image_output, transcribed_text_output, audio_output, text_output], 
        show_progress=True,
    )

demo.launch(share=False)