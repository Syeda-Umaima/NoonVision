---
title: NoonVision
emoji: 🦾
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: true
---

# 🦾 NoonVision – Hands-Free AI Vision Assistant

**⚡ 100% Voice-Activated Object Detection for the Visually Impaired**

<div align="center">

[![Gradio](https://img.shields.io/badge/Gradio-4.20-orange)](https://gradio.app/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8m-Ultralytics-blue)](https://github.com/ultralytics/ultralytics)
[![Web Speech API](https://img.shields.io/badge/Web_Speech_API-Browser-green)](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**🏆 Built for Accessibility | ⚡ No Buttons Required | ♿ Completely Hands-Free**

</div>

---

## 🌟 What is NoonVision?

NoonVision is a **completely hands-free** AI vision assistant designed specifically for visually impaired users. It requires **zero button clicks** - just speak and listen!

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎤 **Always Listening** | Continuous voice recognition - no button presses needed |
| 📷 **Auto-Start Camera** | Camera activates automatically when page loads |
| 🔊 **Audio Feedback** | Startup announcement, processing beep, and spoken results |
| ⚡ **Fast Detection** | Results in 1-2 seconds |
| 🔄 **Auto-Resume** | Automatically listens again after each detection |
| 🎯 **80+ Objects** | Detects people, furniture, electronics, food, animals, vehicles |
| ♿ **100% Accessible** | Designed for complete independence |

---

## 🚀 How to Use

### Step 1: Open the App
Visit the [NoonVision Space](https://huggingface.co/spaces/SyedaUmaima56/noonvision)

### Step 2: Allow Permissions
When prompted, allow **camera** and **microphone** access

### Step 3: Listen for Ready
You'll hear: *"NoonVision ready. Say detect to identify objects around you."*

### Step 4: Speak a Command
Say any of these:
- **"Detect"** ✅ (Best)
- **"What do you see?"** ✅
- **"What's in front of me?"** ✅
- **"Identify"** ✅
- **"Scan"** ✅
- **"Look"** ✅

### Step 5: Listen to Results
Example: *"I can see a laptop and a cup in front of you. Listening. Say detect when ready."*

### Step 6: Repeat!
NoonVision automatically resumes listening after each detection.

---

## 🎤 Voice Commands

| Phrase | Status |
|--------|--------|
| "Detect" | ✅ Best trigger |
| "What do you see?" | ✅ Works great |
| "What's in front of me?" | ✅ Works great |
| "Identify" | ✅ Works |
| "Scan" | ✅ Works |
| "Look" | ✅ Works |
| "What's this?" | ✅ Works |

**Any phrase containing these keywords will trigger detection!**

---

## 🔊 Audio Feedback System

| Event | Audio |
|-------|-------|
| **App Ready** | "NoonVision ready. Say detect to identify objects around you." |
| **Processing** | "Processing." (short beep) |
| **Objects Found** | "I can see a [object] in front of you." |
| **No Objects** | "I don't see any recognizable objects at the moment..." |
| **After Results** | "Listening. Say detect when ready." |

---

## 📦 Detectable Objects (80+ Categories)

### Categories:

| Category | Objects |
|----------|---------|
| 👥 **People** | person |
| 🏠 **Furniture** | chair, couch, table, bed, desk |
| 📱 **Electronics** | cell phone, laptop, keyboard, mouse, TV, remote |
| 🍎 **Food** | banana, apple, orange, sandwich, pizza, cake |
| 🥤 **Kitchen** | cup, bottle, bowl, fork, knife, spoon |
| 🐕 **Animals** | dog, cat, bird, horse, cow, sheep |
| 🚗 **Vehicles** | car, bicycle, motorcycle, bus, truck |
| ⚽ **Sports** | sports ball, baseball bat, tennis racket |
| 📚 **Objects** | book, clock, vase, scissors, teddy bear |

---

## ⚡ Performance

| Metric | Value |
|--------|-------|
| **Detection Time** | 0.5-1.0 seconds |
| **Voice Recognition** | < 0.3 seconds |
| **Audio Generation** | ~0.5 seconds |
| **Total Response** | **1-2 seconds** |
| **Accuracy** | 75-90% (good conditions) |

---

## 🔧 Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER'S BROWSER                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐              ┌──────────────────────┐ │
│  │   Camera     │              │   Web Speech API     │ │
│  │  (Auto-On)   │              │  (Always Listening)  │ │
│  └──────┬───────┘              └──────────┬───────────┘ │
│         │                                  │             │
│         │        ┌─────────────────┐      │             │
│         │        │ Trigger Word    │◄─────┘             │
│         │        │ Detection       │                    │
│         │        │ "detect", etc   │                    │
│         │        └────────┬────────┘                    │
│         │                 │                             │
│         ▼                 ▼                             │
│  ┌────────────────────────────────────────────────────┐ │
│  │              GRADIO INTERFACE                       │ │
│  └────────────────────────────────────────────────────┘ │
│                          │                              │
└──────────────────────────┼──────────────────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   HUGGING FACE SPACE    │
              │   (Python Backend)      │
              ├─────────────────────────┤
              │  ┌─────────────────┐    │
              │  │    YOLOv8m      │    │
              │  │   Detection     │    │
              │  └────────┬────────┘    │
              │           │             │
              │  ┌────────▼────────┐    │
              │  │     gTTS        │    │
              │  │  Audio Output   │    │
              │  └────────┬────────┘    │
              └───────────┼─────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │    Audio Playback       │
              │   (Auto-play results)   │
              └─────────────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Gradio 4.20.0 | Web interface |
| **Voice Recognition** | Web Speech API | Browser-based continuous listening |
| **Object Detection** | YOLOv8m | Real-time object recognition |
| **Text-to-Speech** | gTTS | Audio generation |
| **ML Framework** | PyTorch | Model inference |

---

## 💡 Tips for Best Results

### ✅ Do:
- Use **good lighting** (natural daylight is best)
- Keep objects **2-6 feet** from camera
- **Speak clearly** at normal volume
- **Center objects** in camera frame
- Use **Chrome or Edge** browser

### ❌ Avoid:
- Dark or very bright environments
- Objects too close (< 1 foot) or far (> 10 feet)
- Cluttered backgrounds
- Speaking too softly or too fast
- Safari or Firefox (limited Web Speech API support)

---

## 🌐 Browser Compatibility

| Browser | Voice Recognition | Camera | Recommended |
|---------|------------------|--------|-------------|
| **Chrome** | ✅ Full support | ✅ | ⭐ Best |
| **Edge** | ✅ Full support | ✅ | ⭐ Great |
| **Firefox** | ⚠️ Limited | ✅ | Use Chrome |
| **Safari** | ❌ No support | ✅ | Use Chrome |

---

## 🔒 Privacy & Security

- ✅ **No data storage** - Images processed in real-time and discarded
- ✅ **No voice recording** - Speech processed locally in browser
- ✅ **No tracking** - No analytics or user data collection
- ✅ **Open source** - Code is transparent and auditable

---

## 📄 Files in This Project

| File | Purpose |
|------|---------|
| `app.py` | Main application code |
| `requirements.txt` | Python dependencies |
| `yolov8m.pt` | YOLO model weights |
| `README.md` | This documentation |
| `.gitattributes` | Git LFS configuration |

---

## 🤝 Contributing

Contributions are welcome! Ideas for improvement:

- 🌍 Add more languages for voice commands
- 🎵 Better audio feedback sounds
- 📱 Mobile optimization
- 🔍 Add distance estimation
- 📖 Improve documentation

---

## 📞 Support

If you encounter issues:

1. **Voice not working?** Use Chrome or Edge browser
2. **No camera?** Check browser permissions
3. **Slow detection?** Ensure good lighting
4. **No audio?** Unmute browser tab

---

## 🙏 Acknowledgments

Built with amazing open-source projects:

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Web Speech API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API)
- [Google gTTS](https://gtts.readthedocs.io/)
- [Gradio](https://gradio.app/)

---

<div align="center">

### 🦾 NoonVision

**Empowering Vision Through Voice**

*100% Hands-Free • Zero Buttons • Complete Independence*

Made with ❤️ for accessibility

</div>