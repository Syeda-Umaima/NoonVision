<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NoonVision – CPU Hands-Free AI Vision Assistant</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f9f9f9;
            color: #333;
            line-height: 1.6;
            margin: 0;
            padding: 0;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 20px;
            text-align: center;
        }
        header h1 {
            margin: 0;
            font-size: 2.5em;
        }
        header p {
            font-size: 1.2em;
            margin-top: 10px;
        }
        main {
            max-width: 900px;
            margin: 20px auto;
            padding: 20px;
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0px 3px 15px rgba(0,0,0,0.1);
        }
        h2, h3, h4 {
            color: #667eea;
        }
        code {
            background-color: #eee;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
        }
        pre {
            background-color: #eee;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        ul, ol {
            margin-left: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        table, th, td {
            border: 1px solid #ddd;
        }
        th, td {
            padding: 8px 12px;
            text-align: left;
        }
        th {
            background-color: #667eea;
            color: white;
        }
        footer {
            text-align: center;
            padding: 20px;
            margin-top: 30px;
            color: #666;
            font-size: 0.9em;
        }
        .highlight {
            color: #764ba2;
            font-weight: bold;
        }
    </style>
</head>
<body>

<header>
    <h1>🦾 NoonVision</h1>
    <p>CPU Hands-Free AI Vision Assistant – Detect & Speak Objects Automatically</p>
</header>

<main>
    <h2>✨ Features</h2>
    <ul>
        <li>Real-time object detection using <span class="highlight">YOLOv8m</span> (CPU).</li>
        <li>Automatic speech recognition using <span class="highlight">Whisper-tiny</span> (CPU).</li>
        <li>Hands-free trigger phrase detection: "detect", "what do you see", etc.</li>
        <li>Audio output describing detected objects using <span class="highlight">gTTS</span>.</li>
        <li>Compatible with Gradio v6+, fully CPU-only.</li>
    </ul>

    <h2>💻 Installation</h2>
    <ol>
        <li>Clone the repository:
            <pre><code>git clone https://github.com/yourusername/noonvision.git
cd noonvision</code></pre>
        </li>
        <li>Create a Python virtual environment:
            <pre><code>python -m venv venv</code></pre>
        </li>
        <li>Activate the virtual environment:
            <pre><code># Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate</code></pre>
        </li>
        <li>Install dependencies:
            <pre><code>pip install --upgrade pip
pip install -r requirements.txt</code></pre>
        </li>
        <li>Download YOLOv8m weights (if not auto-downloaded):
            <pre><code>wget https://github.com/ultralytics/assets/releases/download/v0.0/yolov8m.pt</code></pre>
        </li>
    </ol>

    <h2>🚀 Running the App</h2>
    <pre><code>python app.py</code></pre>
    <p>Open the link shown in the terminal (usually <code>http://127.0.0.1:7860</code>) in your browser.</p>
    <p>Allow <span class="highlight">microphone</span> and <span class="highlight">camera</span> permissions. Speak one of the trigger phrases to start detection and hear the results automatically.</p>

    <h2>🎤 Trigger Phrases</h2>
    <ul>
        <li>"detect"</li>
        <li>"what do you see"</li>
        <li>"what's in front of me"</li>
        <li>"what is in front of me"</li>
        <li>"identify objects"</li>
        <li>"what's this"</li>
        <li>"what is this"</li>
        <li>"tell me what you see"</li>
        <li>"scan"</li>
        <li>"look"</li>
    </ul>

    <h2>📁 File Structure</h2>
    <pre><code>noonvision/
│
├─ app.py                 # Main application (CPU-compatible)
├─ requirements.txt       # Python dependencies
├─ README.md              # Markdown documentation
├─ yolov8m.pt             # YOLOv8m model (auto-download)
└─ .gitattributes         # For model weights management with Git LFS</code></pre>

    <h2>⚠️ Notes</h2>
    <ul>
        <li>Runs entirely on CPU, so detection may be slower than GPU.</li>
        <li>First-time model downloads may take a few minutes.</li>
        <li>Ensure microphone and webcam are allowed in the browser.</li>
        <li>Gradio v6+ fixes previous theme argument errors.</li>
    </ul>

    <h2>🛠 Troubleshooting</h2>
    <ul>
        <li><strong>Gradio theme error:</strong> Ensure Gradio 6+ and remove `theme=` argument from `gr.Blocks()`.</li>
        <li><strong>Transformers missing:</strong> Install `transformers>=4.35.0`.</li>
        <li><strong>Slow CPU detection:</strong> Reduce <code>IMG_SIZE</code> in <code>app.py</code>.</li>
    </ul>

    <h2>📜 License</h2>
    <p>MIT License – Free for personal and academic use.</p>
</main>

<footer>
    Made with ❤️ by NoonVision Team | Fully CPU-Compatible | Accessibility First
</footer>

</body>
</html>
