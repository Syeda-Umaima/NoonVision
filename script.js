const video = document.getElementById("camera");
const detectBtn = document.getElementById("detectBtn");
const resultBox = document.getElementById("results");

// Start camera
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => video.srcObject = stream);

// Capture a frame from camera
function captureFrame() {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    canvas.getContext("2d").drawImage(video, 0, 0);
    return canvas.toDataURL("image/jpeg");
}

// Speak output automatically
function speak(text) {
    const synth = window.speechSynthesis;
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = "en-US";
    synth.speak(utter);
}

// Display detection results
function showResults(data) {
    resultBox.innerHTML = "";

    // Auto speak
    speak(data.speech_text);

    // Show detections with big text
    data.detections.forEach(det => {
        const div = document.createElement("div");
        div.className = "result-item";
        div.innerHTML = `
            <p style="font-size:30px; font-weight:bold">${det.label.toUpperCase()}</p>
            <p style="font-size:24px;">Accuracy: <b>${det.confidence}</b></p>
        `;
        resultBox.appendChild(div);
    });
}

// Send to backend
async function detect() {
    const base64 = captureFrame();

    const response = await fetch("/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: base64 })
    });

    const data = await response.json();
    showResults(data);
}

detectBtn.onclick = detect;

// Voice Detection
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (SpeechRecognition) {
    const recognition = new SpeechRecognition();
    recognition.continuous = true;

    recognition.onresult = function (event) {
        const transcript = event.results[event.results.length - 1][0].transcript.toLowerCase();
        console.log("Heard:", transcript);

        if (transcript.includes("detect")) {
            detect();
        }
    };

    recognition.start();
}
