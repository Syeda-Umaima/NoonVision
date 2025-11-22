---
title: NoonVision
emoji: 🦾
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "3.43.0"
app_file: app.py
pinned: true
---
# NoonVision – AI-Powered Voice-Activated Object Detection for the Visually Impaired

NoonVision is a web-based AI application designed to assist visually impaired individuals. It uses **voice commands** to trigger image capture, detects objects using **YOLOv8**, and **speaks the results aloud** using AI-generated voice.

### 🚀 Key Features:
* **Voice Trigger:** Say "Detect" or "What is in front of me?" to automatically capture the image.
* **Real-Time Detection:** Uses the powerful **YOLOv8m** model.
* **Spoken Output:** Audio results via `gTTS` for hands-free accessibility.

### 💡 Tech Stack:
* **Application:** Gradio (using `gr.Blocks` for complex flow)
* **Object Detection:** YOLOv8 (Ultralytics)
* **Speech-to-Text (STT):** Hugging Face Transformers (Whisper-tiny.en)
* **Text-to-Speech (TTS):** gTTS
* **Deployment:** Hugging Face Spaces

### 📢 How to Use:
1.  Open the app in your browser and allow camera/microphone access.
2.  Place an object in front of the webcam.
3.  **Click the microphone button and say your command** (e.g., "Detect what's in front of me").
4.  The system will transcribe the command, capture the image, run detection, and play the audio description automatically.