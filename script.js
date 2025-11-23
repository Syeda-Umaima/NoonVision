// script.js
const video = document.getElementById("camera");
const detectBtn = document.getElementById("detectBtn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

// start camera
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: true });
    video.srcObject = stream;
    statusEl.textContent = "Camera started. Say 'detect' or press the button.";
  } catch (e) {
    statusEl.textContent = "Camera or microphone permission denied.";
    console.error(e);
  }
}
startCamera();

// capture frame to base64
function captureFrameBase64() {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.8);
}

// speak using browser TTS
function speak(text) {
  try {
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "en-US";
    window.speechSynthesis.cancel(); // cancel any existing
    window.speechSynthesis.speak(u);
  } catch (e) {
    console.warn("TTS error", e);
  }
}

// show results visually (big font for accuracy)
function showResults(data) {
  resultsEl.innerHTML = "";
  if (!data || !data.detections || data.detections.length === 0) {
    resultsEl.innerHTML = "<p style='font-size:24px; font-weight:bold;'>No objects detected</p>";
    return;
  }
  data.detections.forEach(d => {
    const div = document.createElement("div");
    div.className = "det";
    div.innerHTML = `<div style="font-size:28px; font-weight:bold;">${d.label.toUpperCase()}</div>
                     <div style="font-size:22px;">Accuracy: <b>${d.confidence}</b></div>`;
    resultsEl.appendChild(div);
  });
}

async function runDetect() {
  statusEl.textContent = "Capturing image... detecting...";
  const b64 = captureFrameBase64();

  try {
    const res = await fetch("/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64: b64 })
    });
    const data = await res.json();
    if (data.error) {
      statusEl.textContent = "Error: " + data.error;
      speak("An error occurred during detection.");
      return;
    }
    statusEl.textContent = `Mode: ${data.mode} | In ${data.elapsed}s`;
    showResults(data);
    // speak result via browser TTS
    if (data.speech_text) {
      speak(data.speech_text);
    }
  } catch (e) {
    statusEl.textContent = "Network or server error";
    console.error(e);
    speak("Detection failed. Please try again.");
  }
}

// button
detectBtn.addEventListener("click", runDetect);

// voice trigger using browser SpeechRecognition
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (SpeechRecognition) {
  const recog = new SpeechRecognition();
  recog.lang = "en-US";
  recog.continuous = true;
  recog.interimResults = false;

  recog.onresult = (ev) => {
    const text = ev.results[ev.results.length - 1][0].transcript.toLowerCase();
    console.log("heard:", text);
    // if contains keyword "detect" or other synonyms
    if (text.includes("detect") || text.includes("what do you see") || text.includes("what's in front")) {
      runDetect();
    }
  };

  recog.onerror = (e) => {
    console.warn("SpeechRecognition error", e);
  };

  recog.onstart = () => {
    console.log("Speech recognition started");
  };

  try {
    recog.start();
  } catch (e) {
    console.warn(e);
  }
} else {
  console.warn("SpeechRecognition not supported in this browser.");
}
