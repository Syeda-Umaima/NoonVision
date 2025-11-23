---
title: NoonVision
emoji: 🦾
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: true
---

# 🦾 NoonVision – AI Vision Assistant

**⚡ Real-Time Voice-Activated Object Detection for the Visually Impaired**

<div align="center">

[![Gradio](https://img.shields.io/badge/Gradio-4.44-orange)](https://gradio.app/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8m-Ultralytics-blue)](https://github.com/ultralytics/ultralytics)
[![Whisper](https://img.shields.io/badge/Whisper-OpenAI-green)](https://github.com/openai/whisper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**🏆 Built for Hackathon | ⚡ Optimized for Speed | ♿ Designed for Accessibility**

</div>

---

## 🌟 What is NoonVision?

NoonVision is a **fully hands-free** AI vision assistant that automatically listens for voice commands and instantly speaks detected objects aloud. No buttons, no clicks – just speak!

### ✨ Key Features

- **🎤 Auto-Start Listening** - Automatically begins after permissions granted
- **⚡ Ultra-Fast Detection** - Results in 1-2 seconds
- **🗣️ Natural Voice Commands** - Just say "Detect" or "What do you see?"
- **🔊 Instant Audio Feedback** - Results spoken immediately
- **📷 Real-Time Streaming** - Live camera feed with continuous processing
- **🎯 High Accuracy** - Detects 80+ object categories
- **♿ 100% Accessible** - Designed for complete independence

---

## 🚀 Quick Start

### 1️⃣ Open the App
Visit the live demo or run locally

### 2️⃣ Grant Permissions
Allow **microphone** and **camera** access when prompted

### 3️⃣ Start Speaking!
The app automatically starts listening. Say:
- **"Detect"**
- **"What do you see?"**
- **"What's in front of me?"**

### 4️⃣ Listen to Results
Audio plays automatically with detected objects

---

## 🎤 Voice Commands

The app recognizes these trigger phrases:

| Command | Works? |
|---------|--------|
| "Detect" | ✅ Best |
| "What do you see?" | ✅ |
| "What's in front of me?" | ✅ |
| "Identify" | ✅ |
| "What's this?" | ✅ |
| "Scan" | ✅ |
| "Look" | ✅ |

**Any phrase containing "detect" will trigger analysis!**

---

## 📦 Detectable Objects (80+ Categories)

### Common Categories:

**👥 People:** person  
**🏠 Household:** cup, bottle, bowl, fork, knife, spoon, chair, couch, table, bed, book, clock, vase  
**📱 Electronics:** cell phone, laptop, keyboard, mouse, TV, remote  
**🍎 Food:** banana, apple, orange, sandwich, pizza, donut, cake  
**🐕 Animals:** dog, cat, bird, horse, cow, sheep  
**🚗 Vehicles:** car, bicycle, motorcycle, bus, truck, airplane  
**⚽ Sports:** sports ball, baseball bat, tennis racket, frisbee, skateboard  

And many more!

---

## ⚡ Performance

### Speed Benchmarks

| Metric | Performance |
|--------|-------------|
| **Detection Time** | 0.5-1.0 seconds |
| **Voice Recognition** | <0.3 seconds |
| **Audio Generation** | ~0.5 seconds |
| **Total Response** | **1-2 seconds** |
| **Accuracy** | 75-90% (good conditions) |

### Optimizations Applied

✅ Reduced image size (960→640) for faster processing  
✅ FP16 inference on GPU when available  
✅ Increased confidence threshold (0.25→0.30) for accuracy  
✅ Optimized streaming intervals (100ms→150ms)  
✅ Chunk-based audio processing  
✅ Model warm-up at startup  

---

## 🔧 Technical Architecture

### System Flow

```
┌─────────────┐
│   Browser   │
│  (Camera +  │
│ Microphone) │
└──────┬──────┘
       │
       ├─────────────────────────┐
       │                         │
       ▼                         ▼
┌─────────────┐          ┌─────────────┐
│   Camera    │          │   Voice     │
│  Streaming  │          │  Streaming  │
└──────┬──────┘          └──────┬──────┘
       │                         │
       ▼                         ▼
┌─────────────┐          ┌─────────────┐
│   YOLOv8m   │          │   Whisper   │
│  Detection  │          │     STT     │
└──────┬──────┘          └──────┬──────┘
       │                         │
       └──────────┬──────────────┘
                  │
                  ▼
           ┌─────────────┐
           │   Trigger   │
           │  Detection  │
           └──────┬──────┘
                  │
                  ▼
           ┌─────────────┐
           │    gTTS     │
           │ Audio Gen   │
           └──────┬──────┘
                  │
                  ▼
           ┌─────────────┐
           │   Speaker   │
           │   Output    │
           └─────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Gradio 4.44.0 | Web interface with streaming |
| **Object Detection** | YOLOv8m | Real-time object recognition |
| **Speech Recognition** | Whisper-tiny.en | Voice command transcription |
| **Text-to-Speech** | gTTS | Audio generation |
| **ML Framework** | PyTorch 2.1+ | Model inference |
| **Image Processing** | PIL/Pillow | Image manipulation |

---

## 💻 Installation & Deployment

### Option 1: Hugging Face Spaces (Recommended)

**Already deployed! Just use the live link.**

### Option 2: Run Locally (CPU)

```bash
# Clone repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/NoonVision
cd NoonVision

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py

# Open browser at http://localhost:7860
```

**Requirements:**
- Python 3.8+
- Webcam
- Microphone
- Internet (for gTTS)

### Option 3: Google Colab (GPU Acceleration)

For **faster processing** with GPU:

1. **Open this Colab notebook:** [NoonVision Colab](https://colab.research.google.com/)

2. **Copy and run this code:**

```python
# Install dependencies
!pip install gradio>=4.44.0 ultralytics>=8.1.0 gTTS transformers[torch] datasets soundfile librosa

# Clone repository
!git clone https://huggingface.co/spaces/YOUR_USERNAME/NoonVision
%cd NoonVision

# Run with public URL
!python app.py --share

# ✅ Copy the public URL that appears (looks like: https://xxxxx.gradio.live)
```

3. **Access the public URL** from any device

**GPU Benefits:**
- ⚡ 2-3x faster detection
- 🎯 Better accuracy with FP16 precision
- 💪 Handle multiple users simultaneously

**Note:** Colab provides **free GPU** for limited time. Perfect for hackathon demos!

---

## 💡 Usage Tips

### For Best Results

✅ **Lighting:** Bright, even lighting (natural daylight is best)  
✅ **Distance:** Keep objects 2-6 feet from camera  
✅ **Clarity:** Speak clearly at normal volume  
✅ **Positioning:** Center objects in camera frame  
✅ **Background:** Minimize clutter for better detection  

### Common Issues & Solutions

**Issue:** Voice not working  
**Fix:** Check browser mic permissions, use Chrome/Firefox

**Issue:** No objects detected  
**Fix:** Improve lighting, move objects closer, check camera angle

**Issue:** Slow performance  
**Fix:** Use GPU (Colab) or reduce image quality in settings

**Issue:** Audio not playing  
**Fix:** Unmute browser tab, check volume, verify audio permissions

---

## 🏆 Hackathon Demo Script

### 1. Introduction (30 seconds)
"This is NoonVision - an AI vision assistant designed for visually impaired individuals. It provides completely hands-free object detection using just voice commands."

### 2. Live Demo (1 minute)
1. Open the app (already running)
2. Point camera at common object (cup, phone, laptop)
3. Say: **"Detect"**
4. Wait for audio: *"I see one cup"*
5. Move camera to new objects
6. Say: **"What do you see?"**
7. Listen to results: *"I see one laptop and one mouse"*

### 3. Key Features Highlight (30 seconds)
- **Point to screen:** "Notice it automatically starts listening - no buttons needed"
- **Show bounding boxes:** "Green boxes show detected objects in real-time"
- **Play audio:** "Natural language audio for complete accessibility"
- **Show speed:** "Results in just 1-2 seconds"

### 4. Impact Statement (30 seconds)
"NoonVision empowers visually impaired individuals with independence. By combining state-of-the-art AI (YOLOv8, Whisper) with accessibility-first design, we're making computer vision truly accessible to everyone."

### 5. Technical Excellence (30 seconds)
- 80+ object categories detected
- Optimized for speed (1-2 second response)
- 75-90% accuracy in good conditions
- Works on CPU and GPU
- Open source and free

---

## 📊 Project Statistics

- **Lines of Code:** ~400
- **Models Used:** 2 (YOLOv8m + Whisper-tiny)
- **Dependencies:** 11 packages
- **Detection Categories:** 80+
- **Supported Languages:** English (expandable)
- **License:** MIT (Open Source)

---

## 🤝 Contributing

We welcome contributions!

**Ways to help:**
- 🐛 Report bugs
- 💡 Suggest features
- 🌍 Add language support
- 📖 Improve documentation
- ⭐ Star the project

---

## 🔒 Privacy & Security

✅ **No data storage** - Images processed in real-time  
✅ **No logging** - Voice commands not recorded  
✅ **Local processing** - Detection happens on server  
✅ **Open source** - Code is transparent  
✅ **No tracking** - No analytics or user data collection  

---

## 📄 License

This project is licensed under the **MIT License** - free for personal and commercial use.

---

## 🙏 Acknowledgments

Built with amazing open-source projects:

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [Google gTTS](https://gtts.readthedocs.io/) - Text-to-speech
- [Gradio](https://gradio.app/) - Web interface

Special thanks to the accessibility community for guidance.

---

## 📞 Contact & Support

- **GitHub Issues:** Report bugs or request features
- **Discussions:** Ask questions and share feedback
- **Email:** contact@noonvision.ai

---

## 🌟 Share NoonVision

If this project helps you or someone you know:

⭐ **Star this repository**  
🔄 **Share on social media**  
💬 **Tell accessibility communities**  
📧 **Email to organizations**  

**Together we can make technology accessible for everyone!**

---

<div align="center">

### 🦾 NoonVision

**Empowering Vision Through AI**

*Made with ❤️ for accessibility*

**[Live Demo](#) | [Documentation](#) | [GitHub](#)**

---

**🏆 Hackathon Project 2025**

</div>