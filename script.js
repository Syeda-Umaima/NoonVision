function speak(text) {
    const synth = window.speechSynthesis;
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = "en-US";
    synth.speak(utter);
}

function displayResults(result) {
    resultBox.innerHTML = "";

    // Speak automatically
    speak(result.speech_text);

    result.detections.forEach(det => {
        const item = document.createElement("div");
        item.className = "result-item";
        item.innerHTML = `
            <p style="font-size:28px"><b>${det.label.toUpperCase()}</b></p>
            <p style="font-size:22px">Accuracy: <b>${det.confidence}</b></p>
        `;
        resultBox.appendChild(item);
    });
}
