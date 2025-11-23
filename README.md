---
title: NoonVision
emoji: 🦾
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.0.0"
app_file: app.py
pinned: true
---
# 🦾 NoonVision – CPU Hands-Free AI Vision Assistant
**CPU-compatible Hands-Free Object Detection and Audio Feedback**
---
## ✨ Features
- Real-time object detection using **YOLOv8m** (CPU)
- Automatic speech recognition using **Whisper-tiny** (CPU)
- Hands-free trigger phrase detection: "detect", "what do you see", etc.
- Audio output describing detected objects using **gTTS**
- Compatible with **Gradio v6+**, fully CPU-only
---
## 💻 Installation
1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/noonvision.git
cd noonvision

Create a virtual environment:

Bashpython -m venv venv

Activate it:

Bash# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

Install dependencies:

Bashpip install --upgrade pip
pip install -r requirements.txt

Download YOLOv8m weights if not auto-downloaded:

Bashwget https://github.com/ultralytics/assets/releases/download/v0.0/yolov8m.pt
🚀 Running the App
Bashpython app.py
Open the link shown in the terminal (usually http://127.0.0.1:7860) in your browser.
Allow microphone and camera permissions. Speak one of the trigger phrases to start detection and hear results automatically.
🎤 Trigger Phrases

"detect"
"what do you see"
"what's in front of me"
"what is in front of me"
"identify objects"
"what's this"
"what is this"
"tell me what you see"
"scan"
"look"

📁 File Structure
textnoonvision/
│
├─ app.py                 # Main application (CPU-compatible)
├─ requirements.txt       # Python dependencies
├─ README.md              # Documentation
├─ yolov8m.pt             # YOLOv8m model (auto-download)
└─ .gitattributes         # For model weights management with Git LFS
⚠️ Notes

Runs entirely on CPU, so detection may be slower than GPU.
First-time model downloads may take a few minutes.
Ensure microphone and webcam are allowed in the browser.
Gradio v6+ fixes previous theme argument errors.

🛠 Troubleshooting

Gradio theme error: Ensure Gradio 6+ and remove theme= argument from gr.Blocks().
Transformers missing: Install transformers>=4.35.0.
Slow CPU detection: Reduce IMG_SIZE in app.py.

📜 License
MIT License – Free for personal and academic use.
🙏 Acknowledgments
Built with:

Ultralytics YOLOv8 – Object detection
OpenAI Whisper – Speech recognition
Google gTTS – Text-to-speech
Gradio – Web interface