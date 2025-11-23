---
title: NoonVision
emoji: 🦾
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: true
---

# 🦾 NoonVision – CPU Hands-Free AI Vision Assistant

**CPU-compatible Hands-Free Object Detection and Audio Feedback**

## ✨ Features

- Real-time object detection using **YOLOv8m** (CPU)
- Automatic speech recognition using **Whisper-tiny** (CPU)
- Hands-free trigger phrase detection
- Audio output describing detected objects using **gTTS**
- Simple and reliable Gradio interface

## 🎯 Trigger Phrases

- "detect"
- "what do you see"
- "what's in front of me"
- "what is in front of me"
- "identify objects"
- "what's this"
- "what is this"
- "tell me what you see"
- "scan"
- "look"

## 🚀 Usage

1. **Allow permissions** - Click "Allow" when prompted for camera and microphone access
2. **Click the microphone** - Press the record button and speak a command
3. **Wait for processing** - The app will detect objects in the camera view
4. **View and listen** - See detected objects with bounding boxes and hear the audio description

## 🔧 Technical Details

- **Models**: YOLOv8m for object detection, Whisper-tiny for speech recognition
- **Platform**: CPU-only optimization
- **Framework**: Gradio 4.44.1+ for stable web interface
- **Compatibility**: Fully compatible with Hugging Face Spaces

## ⚠️ Notes

- First-time model downloads may take 1-2 minutes
- Detection runs on CPU, so allow 2-3 seconds for processing
- Ensure microphone and webcam permissions are granted
- Audio descriptions are generated using Google TTS

## 🛠 Troubleshooting

**Microphone not working:**
- Check browser permissions
- Ensure no other apps are using the microphone
- Click the microphone button to start recording

**Camera not working:**
- Allow camera permissions
- Check if camera is being used by another application

**Slow detection:**
- This is normal on CPU - processing takes 2-3 seconds

## 📜 License

MIT License - Free for personal and academic use.

## 🙏 Acknowledgments

Built with:
- **Ultralytics YOLOv8** - Object detection
- **OpenAI Whisper** - Speech recognition  
- **Google gTTS** - Text-to-speech
- **Gradio** - Web interface
- **Hugging Face** - Deployment platform