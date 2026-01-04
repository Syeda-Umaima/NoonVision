---
title: NoonVision
emoji: 🦾
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: "4.19.2"
app_file: app.py
pinned: true
---

# 🦾 NoonVision

**Hands-Free AI Vision Assistant for the Visually Impaired**

> Empowering independence through voice-controlled computer vision

## ✨ What It Does

NoonVision enables blind and visually impaired individuals to understand their surroundings using just their voice. Simply say **"Detect"** and AI describes the world around you.

- 🎤 **100% Voice Controlled** — No buttons required
- 👁️ **80+ Objects** — People, vehicles, animals, furniture & more
- ⚡ **Real-Time** — Results in 1-2 seconds
- 🔊 **Audio Feedback** — Natural speech descriptions

## 🚀 How To Use

1. **Click** anywhere on the page to start
2. **Allow** camera & microphone access
3. **Say** "Detect" (or "scan", "look", "what do you see")
4. **Listen** to the audio description
5. **Repeat** — system auto-resets for next scan

## 🗣️ Voice Commands

| Command | Action |
|---------|--------|
| "Detect" | Scan surroundings |
| "What do you see?" | Same as detect |
| "Scan" / "Look" | Quick scan |
| "Identify" / "Check" | Identify objects |

## 🛠️ Tech Stack

- **Detection:** YOLOv8m
- **UI:** Gradio
- **Speech:** Web Speech API
- **TTS:** Google gTTS

## 🌐 Browser Support

| Browser | Support |
|---------|---------|
| ✅ Chrome | Recommended |
| ✅ Edge | Full support |
| ⚠️ Firefox/Safari | Limited |

## 🐛 Debug

Open browser console (F12) and run:
```javascript
noonvision.status()       // Check state
noonvision.doDetection()  // Manual trigger
```

---

<p align="center">Made with ❤️ for Accessibility</p>