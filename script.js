// Copyright 2023 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//      http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import { GestureRecognizer, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";
const demosSection = document.getElementById("demos");
let gestureRecognizer;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
const videoHeight = "360px";
const videoWidth = "480px";
let gestureText = [];
let lastSavedLetter = '';
let lastGestureTime = 0; // Registro del tiempo del último gesto
const minTimeBetweenGestures = 1000;
// Before we can use HandLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.

const createGestureRecognizer = async (selectedOption) => {
    console.log(selectedOption);
    let country;
    if(selectedOption==="PE"){
        country="abecedario-peru-v2"
    }else{
        if(selectedOption==="EC"){
            country="abecedario-ecuador-v2"
        }else{
            if(selectedOption==="MX"){
                country="abecedario-mx-v2"
            }else{
                country="abecedario-usa"
            }
        }
    }
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `./${country}.task`,
            delegate: "GPU"
        },
        runningMode: runningMode
    });
    demosSection.classList.remove("invisible");
};
createGestureRecognizer("Perú");
/********************************************************************
// Demo 2: Continuously grab image from webcam stream and detect it.
********************************************************************/
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const gestureOutput = document.getElementById("gesture_output");
// Check if webcam access is supported.
function actualizarTextoEnHTML() {
    // Actualiza el contenido del elemento <span> con id "gestureText"
    const spanElement = document.getElementById("gestureText");
    spanElement.textContent = `Frase: ${gestureText.join(' ')}`;
}
const miSelect = document.getElementById("miSelect");

// Agrega un evento de cambio (change) al elemento select
miSelect.addEventListener("change", function () {
    // Obtiene el valor seleccionado
    const valorSeleccionado = miSelect.value;

    // Muestra el valor seleccionado en la consola
    console.log("Opción seleccionada:", valorSeleccionado);
    createGestureRecognizer(valorSeleccionado);
});
// Llama a la función actualizarTextoEnHTML cada 1000 milisegundos (1 segundo)
setInterval(actualizarTextoEnHTML, 1000);
document.getElementById("soundButton")
    .addEventListener("click", function () {
        const synth = window.speechSynthesis;
        const utterance = new SpeechSynthesisUtterance(gestureText.join(''));
        utterance.lang = "es-MX";
        utterance.pitch = 1;
        utterance.rate = 1;
        utterance.volume = 5;
        synth.speak(utterance);
    });
function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}
// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
}
else {
    console.warn("getUserMedia() is not supported by your browser");
}
// Enable the live webcam view and start detection.
function resetTranslation() {
    gestureText = []; // Reinicia el array de texto
    actualizarTextoEnHTML();
}
const resetButton = document.getElementById("resetButton");
resetButton.addEventListener("click", resetTranslation);
function enableCam(event) {
    if (!gestureRecognizer) {
        alert("Please wait for gestureRecognizer to load");
        return;
    }
    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "ACTIVAR PREDICCIONES";
    }
    else {
        webcamRunning = true;
        enableWebcamButton.innerText = "DESACTIVAR PREDICCIONES";
    }
    // getUsermedia parameters.
    const constraints = {
        video: true
    };
    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}
let lastVideoTime = -1;
let results = undefined;
async function predictWebcam() {
    const webcamElement = document.getElementById("webcam");
    // Now let's start detecting the stream.
    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await gestureRecognizer.setOptions({ runningMode: "VIDEO" });
    }
    let nowInMs = Date.now();
    if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        results = gestureRecognizer.recognizeForVideo(video, nowInMs);
    }
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasElement.style.height = videoHeight;
    webcamElement.style.height = videoHeight;
    canvasElement.style.width = videoWidth;
    webcamElement.style.width = videoWidth;
    if (results.landmarks) {
        for (const landmarks of results.landmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
                color: "#00FF00",
                lineWidth: 2
            });
            drawLandmarks(canvasCtx, landmarks, { color: "#FF0000", lineWidth: 0.5 });
        }
    }
    canvasCtx.restore();
    if (results.gestures.length > 0) {
        const currentTime = Date.now(); // Obtener el tiempo actual
        const categoryName = results.gestures[0][0].categoryName;
        const categoryScore = parseFloat(results.gestures[0][0].score * 100).toFixed(2);

        // Verificar si ha pasado suficiente tiempo desde el último gesto similar
        if (currentTime - lastGestureTime >= minTimeBetweenGestures) {
            // Filtrar gestos repetidos
            if (categoryName !== lastSavedLetter || categoryName === 'R') {
                gestureOutput.style.display = "block";
                gestureOutput.style.width = videoWidth;
                gestureOutput.innerText = `LETRA: ${categoryName}\n PROBABILIDAD: ${categoryScore} %\n FRASE: ${gestureText.join('')}`;
                if (categoryScore > 90) {
                    gestureText.push(categoryName);
                    console.log(gestureText);

                    // Actualizar el tiempo del último gesto y la última letra guardada
                    lastGestureTime = currentTime;
                    lastSavedLetter = categoryName;
                }
            }
        }
    }
    else {
        gestureOutput.style.display = "none";
    }
    // Call this function again to keep predicting when the browser is ready.
    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
}