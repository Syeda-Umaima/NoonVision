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

NoonVision is a web-based AI application designed to provide **hands-free object identification** for the visually impaired. It uses a **voice command trigger** to capture the current webcam frame, detects objects, and instantly speaks the results aloud.

### ✨ Key Features:
* **Voice Trigger:** Say **"Detect"** or **"What is in front of me?"** to automatically trigger image capture and analysis.
* **Accessibility Focus:** Fully voice-activated interaction using the microphone for a user-friendly experience.
* **Advanced Detection:** Utilizes the fast and accurate **YOLOv8m** model.
* **Spoken Output:** Audio results via `gTTS` for immediate, non-visual feedback.

### 💡 Technical Stack:
* **Application Framework:** Gradio (using `gr.Blocks` for complex event handling)
* **Object Detection:** YOLOv8 (Ultralytics)
* **Speech-to-Text (STT):** Hugging Face Transformers (Whisper-tiny.en)
* **Text-to-Speech (TTS):** gTTS
* **Deployment:** Hugging Face Spaces

### 📢 How to Use:
1.  Open the app in your browser and allow camera/microphone access.
2.  Place an object in front of the webcam.
3.  **Click the microphone button and say your command** (e.g., "Detect").
4.  The system will transcribe the command, capture the image, run detection, and play the audio description automatically.