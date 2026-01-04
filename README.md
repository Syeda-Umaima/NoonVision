<p align="center">
  <img src="https://img.shields.io/badge/🦾-NoonVision-667eea?style=for-the-badge&labelColor=764ba2" alt="NoonVision">
</p>

<h1 align="center">🦾 NoonVision</h1>

<p align="center">
  <strong>Hands-Free AI Vision Assistant for the Visually Impaired</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-Object%20Detection-green?style=flat-square" alt="YOLOv8">
  <img src="https://img.shields.io/badge/Gradio-UI%20Framework-orange?style=flat-square" alt="Gradio">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License">
</p>

<p align="center">
  <em>Empowering independence through voice-controlled computer vision</em>
</p>

---

## 🌟 Overview

**NoonVision** is a revolutionary accessibility tool that enables blind and visually impaired individuals to understand their surroundings using just their voice. No buttons, no complex interfaces — simply say "Detect" and let AI describe the world around you.

Built with state-of-the-art YOLOv8 object detection and natural text-to-speech, NoonVision provides real-time audio descriptions of detected objects, making navigation and daily tasks more accessible than ever.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎤 **100% Voice Controlled** | No buttons required — just speak to interact |
| 👁️ **80+ Object Categories** | Detects people, vehicles, animals, furniture, electronics, and more |
| ⚡ **Real-Time Processing** | Results delivered in 1-2 seconds |
| 🔊 **Natural Audio Feedback** | Human-like speech describes your surroundings |
| 📱 **Works Everywhere** | Browser-based — no installation needed |
| ♿ **Accessibility First** | Designed from the ground up for visually impaired users |

---

## 🎯 How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   👤 User says "Detect"                                     │
│        ↓                                                    │
│   📷 Camera captures current frame                          │
│        ↓                                                    │
│   🤖 YOLOv8 AI analyzes the image                          │
│        ↓                                                    │
│   🔊 "I can see a person and a laptop in front of you"     │
│        ↓                                                    │
│   🎤 System returns to listening mode                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🗣️ Voice Commands

NoonVision responds to natural speech. Try any of these:

- **"Detect"** — Scan your surroundings
- **"What do you see?"** — Same as detect
- **"Scan"** — Quick scan
- **"Look"** — Check what's in front
- **"Identify"** — Identify objects
- **"Check"** — See what's around

---

## 🚀 Quick Start

### Using Hugging Face Spaces (Recommended)

1. **Visit** the live demo at [NoonVision on Hugging Face](https://huggingface.co/spaces/your-username/noonvision)
2. **Click** anywhere on the page to initialize
3. **Allow** camera and microphone permissions
4. **Say** "Detect" and listen!

### Running Locally

```bash
# Clone the repository
git clone https://github.com/your-username/noonvision.git
cd noonvision

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| **Object Detection** | YOLOv8m (Medium) |
| **Web Framework** | Gradio 4.19.2 |
| **Speech Recognition** | Web Speech API |
| **Text-to-Speech** | Google TTS (gTTS) |
| **Camera Access** | HTML5 getUserMedia |
| **Backend** | Python 3.10+ |

---

## 📋 Requirements

### Software
- Python 3.10 or higher
- Modern web browser (Chrome or Edge recommended)

### Hardware
- Webcam or device camera
- Microphone
- Speakers or headphones

### Browser Compatibility

| Browser | Support |
|---------|---------|
| ✅ Chrome | Full support (recommended) |
| ✅ Edge | Full support |
| ⚠️ Firefox | Limited speech recognition |
| ⚠️ Safari | Limited speech recognition |

---

## 🎨 Features in Detail

### 🎤 Voice Recognition
- Continuous listening for trigger words
- Works in noisy environments
- Supports natural language variations

### 📷 Smart Camera
- Auto-starts when page loads
- Optimized for various lighting conditions
- Works with front and rear cameras

### 🤖 AI Detection
- Powered by YOLOv8 — state-of-the-art object detection
- Detects 80+ object categories
- Confidence scoring for accuracy
- Bounding box visualization

### 🔊 Audio Response
- Natural, conversational speech
- Clear pronunciation of object names
- Handles singular/plural correctly
- "I can see **a person** and **two chairs**..."

---

## 🔮 Future Roadmap

- [ ] **Scene Description** — Describe spatial relationships between objects
- [ ] **Distance Estimation** — "There's a chair about 3 feet ahead"
- [ ] **Text Reading (OCR)** — Read signs, labels, and documents
- [ ] **Face Recognition** — Identify known individuals
- [ ] **Offline Mode** — Work without internet connection
- [ ] **Mobile App** — Native iOS/Android applications
- [ ] **Multi-language** — Support for additional languages

---

## 🤝 Contributing

We welcome contributions from the community! Whether it's:

- 🐛 Bug fixes
- ✨ New features
- 📚 Documentation improvements
- 🌍 Translations

Please read our contributing guidelines before submitting a PR.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Ultralytics](https://ultralytics.com/)** — For the incredible YOLOv8 model
- **[Gradio](https://gradio.app/)** — For the intuitive web framework
- **[Hugging Face](https://huggingface.co/)** — For hosting and community support
- **The Accessibility Community** — For invaluable feedback and testing

---

## 📬 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/your-username/noonvision/issues)
- **Discussions:** [GitHub Discussions](https://github.com/your-username/noonvision/discussions)

---

<p align="center">
  <strong>Made with ❤️ for Accessibility</strong>
</p>

<p align="center">
  <em>Because everyone deserves to see the world</em>
</p>

---

<p align="center">
  <img src="https://img.shields.io/badge/⭐-Star%20this%20repo-yellow?style=for-the-badge" alt="Star">
</p>