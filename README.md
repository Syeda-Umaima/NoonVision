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

# 🦾 NoonVision – CPU Hands-Free AI Vision Assistant

**CPU-compatible Hands-Free Object Detection with Automatic Audio Feedback**

NoonVision is an accessible, voice-activated application that uses computer vision to identify objects in a live camera feed and speaks the results aloud. It is designed to be **fully CPU-compatible** for broad deployment and accessibility.

---

## ✨ Key Features

| Feature | Technology | Note |
| :--- | :--- | :--- |
| **Object Detection** | YOLOv8m (Ultralytics) | Mid-size model for good accuracy on CPU. |
| **Speech Recognition** | Whisper-tiny (Hugging Face) | Tiny English model for fast transcription. |
| **Audio Feedback** | gTTS | Generates natural-sounding spoken results. |
| **Hands-Free Operation** | Gradio Streaming Events | Automatically listens for voice commands. |
| **Deployment** | Gradio | Simple, modern web interface. |

---

## 💻 Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/YOUR_USERNAME/noonvision.git](https://github.com/YOUR_USERNAME/noonvision.git)
    cd noonvision
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

3.  **Activate it:**

    ```bash
    # Windows
    venv\Scripts\activate

    # Linux/macOS
    source venv/bin/activate
    ```

4.  **Install dependencies:**

    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

5.  **Download YOLOv8m weights (if not auto-downloaded by Ultralytics):**

    ```bash
    wget [https://github.com/ultralytics/assets/releases/download/v0.0/yolov8m.pt](https://github.com/ultralytics/assets/releases/download/v0.0/yolov8m.pt)
    ```

---

## 🚀 Running the App (Local)

Once the environment is set up, run the main application file:

```bash
python app.py