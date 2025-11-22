const captureButton = document.getElementById("captureBtn");
const resultBox = document.getElementById("results");

let videoStream;

// Start camera
async function startCamera() {
    videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
    document.getElementById("camera").srcObject = videoStream;
}
startCamera();


// ----------------------
// VOICE ACTIVATION LOGIC
// ----------------------
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = new SpeechRecognition();
recognition.continuous = true;
recognition.lang = "en-US";

recognition.onresult = function (event) {
    const text = event.results[event.results.length - 1][0].transcript.toLowerCase();
    console.log("User said:", text);

    if (text.includes("detect")) {
        console.log("Voice command detected: Auto-capturing...");
        captureAndDetect();
    }
};

recognition.start();


// ----------------------
// CAPTURE + SEND TO BACKEND
// ----------------------
async function captureAndDetect() {
    const video = document.getElementById("camera");
    const canvas = document.createElement("canvas");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    const base64Image = canvas.toDataURL("image/jpeg");

    const response = await fetch("https://your-hf-link/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: base64Image })
    });

    const result = await response.json();
    displayResults(result);
}


// ----------------------
// SHOW RESULTS ON SCREEN
// ----------------------
function displayResults(result) {
    resultBox.innerHTML = "";

    result.detections.forEach(det => {
        const item = document.createElement("div");
        item.className = "result-item";
        item.innerHTML = `
            <p style="font-size:28px">
                <b>${det.label.toUpperCase()}</b>
            </p>
            <p style="font-size:22px">Accuracy: <b>${det.confidence}</b></p>
        `;
        resultBox.appendChild(item);
    });
}


// Capture button (manual fallback)
captureButton.addEventListener("click", captureAndDetect);
