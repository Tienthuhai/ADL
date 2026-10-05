// ============================================================
//  YOLO11 REALTIME WEB APP
// YOLO11n
// Input: 640 x 640
// Output: [1, 9, 8400]
// ============================================================


// ------------------------------------------------------------
// CONFIGURATION
// ------------------------------------------------------------

const MODEL_PATH = "./best.onnx";

const INPUT_SIZE = 320;

const CONF_THRESHOLD = 0.6;

const IOU_THRESHOLD = 0.5;


// ============================================================
// IMPORTANT:
// CHANGE THESE NAMES TO MATCH YOUR MODEL
// ============================================================
//
// Your model output is [1, 9, 8400].
// Therefore:
//
// 9 = 4 bbox values + 5 class scores.
//
// Replace the five names below with the exact names printed by:
//
// print(model.names)
//
// ============================================================

const CLASS_NAMES = [
    "Camera",
    "Emergency_light",
    "Light",
    "Switch",
    "Volume"
];


// ============================================================
// GLOBAL VARIABLES
// ============================================================

let session = null;

let stream = null;

let running = false;

let lastTime = performance.now();

let frameCounter = 0;

let fps = 0;


// ============================================================
// DOM
// ============================================================

const video =
    document.getElementById("video");

const overlay =
    document.getElementById("overlay");

const overlayCtx =
    overlay.getContext("2d");

const processingCanvas =
    document.getElementById("processingCanvas");

const processingCtx =
    processingCanvas.getContext("2d", {
        willReadFrequently: true
    });

const statusElement =
    document.getElementById("status");

const fpsElement =
    document.getElementById("fps");

const objectsElement =
    document.getElementById("objects");

const detectionsElement =
    document.getElementById("detections");

const loadButton =
    document.getElementById("loadButton");

const cameraButton =
    document.getElementById("cameraButton");

const stopButton =
    document.getElementById("stopButton");


// ============================================================
// STATUS
// ============================================================

function setStatus(text) {

    statusElement.textContent = text;

}


// ============================================================
// LOAD MODEL
// ============================================================

async function loadModel() {

    try {

        setStatus("Loading YOLO model...");

        loadButton.disabled = true;

        // Prefer WebGL when available.
        // If WebGL fails, ONNX Runtime can fall back to WASM.

        try {

            session =
                await ort.InferenceSession.create(
                    MODEL_PATH,
                    {
                        executionProviders: [
                            "webgl"
                        ]
                    }
                );

            setStatus("YOLO loaded - WebGL");

        } catch (webglError) {

            console.warn(
                "WebGL failed:",
                webglError
            );

            session =
                await ort.InferenceSession.create(
                    MODEL_PATH,
                    {
                        executionProviders: [
                            "wasm"
                        ]
                    }
                );

            setStatus("YOLO loaded - WASM");

        }

        console.log(
            "Input names:",
            session.inputNames
        );

        console.log(
            "Output names:",
            session.outputNames
        );

        console.log(
            "Input metadata:",
            session.inputMetadata
        );

        console.log(
            "Output metadata:",
            session.outputMetadata
        );

    } catch (error) {

        console.error(error);

        setStatus(
            "Model loading failed"
        );

        alert(
            "Cannot load best.onnx.\n\n" +
            "Check that best.onnx is in the same folder as index.html."
        );

        loadButton.disabled = false;

    }

}


// ============================================================
// CAMERA
// ============================================================

async function startCamera() {

    if (!session) {

        alert(
            "Load the model first."
        );

        return;

    }

    try {

        setStatus(
            "Requesting camera..."
        );

        stream =
            await navigator.mediaDevices.getUserMedia(
                {
                    video: {
                        facingMode: {
                            ideal: "environment"
                        },

                        width: {
                            ideal: 640
                        },

                        height: {
                            ideal: 480
                        }
                    },

                    audio: false
                }
            );

        video.srcObject = stream;

        await video.play();

        setupOverlay();

        running = true;

        setStatus(
            "Camera running"
        );

        requestAnimationFrame(
            inferenceLoop
        );

    } catch (error) {

        console.error(error);

        setStatus(
            "Camera error"
        );

        alert(
            "Camera access failed.\n\n" +
            "Make sure you opened the website using HTTPS and allowed Camera permission."
        );

    }

}


// ============================================================
// STOP CAMERA
// ============================================================

function stopCamera() {

    running = false;

    if (stream) {

        stream
            .getTracks()
            .forEach(
                track => track.stop()
            );

        stream = null;

    }

    video.srcObject = null;

    overlayCtx.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );

    objectsElement.textContent = "0";

    detectionsElement.textContent =
        "No detection";

    setStatus(
        session
            ? "Model loaded"
            : "Model not loaded"
    );

}


// ============================================================
// OVERLAY
// ============================================================

function setupOverlay() {

    overlay.width =
        video.videoWidth;

    overlay.height =
        video.videoHeight;

}


// ============================================================
// PREPROCESS IMAGE
// ============================================================

function preprocess() {

    processingCtx.clearRect(
        0,
        0,
        INPUT_SIZE,
        INPUT_SIZE
    );

    const videoWidth =
        video.videoWidth;

    const videoHeight =
        video.videoHeight;


    // Letterbox scaling

    const scale =
        Math.min(
            INPUT_SIZE / videoWidth,
            INPUT_SIZE / videoHeight
        );


    const newWidth =
        videoWidth * scale;

    const newHeight =
        videoHeight * scale;


    const offsetX =
        (INPUT_SIZE - newWidth) / 2;

    const offsetY =
        (INPUT_SIZE - newHeight) / 2;


    // YOLO letterbox background

    processingCtx.fillStyle =
        "#808080";

    processingCtx.fillRect(
        0,
        0,
        INPUT_SIZE,
        INPUT_SIZE
    );


    processingCtx.drawImage(
        video,

        0,
        0,
        videoWidth,
        videoHeight,

        offsetX,
        offsetY,
        newWidth,
        newHeight
    );


    const imageData =
        processingCtx.getImageData(
            0,
            0,
            INPUT_SIZE,
            INPUT_SIZE
        );


    const data =
        imageData.data;


    const floatData =
        new Float32Array(
            3 *
            INPUT_SIZE *
            INPUT_SIZE
        );


    const channelSize =
        INPUT_SIZE *
        INPUT_SIZE;


    for (
        let i = 0;
        i < channelSize;
        i++
    ) {

        const r =
            data[i * 4] / 255.0;

        const g =
            data[i * 4 + 1] / 255.0;

        const b =
            data[i * 4 + 2] / 255.0;


        floatData[i] =
            r;

        floatData[
            channelSize + i
        ] =
            g;

        floatData[
            channelSize * 2 + i
        ] =
            b;

    }


    return {
        tensor: new ort.Tensor(
            "float32",
            floatData,
            [
                1,
                3,
                INPUT_SIZE,
                INPUT_SIZE
            ]
        ),

        scale: scale,

        offsetX: offsetX,

        offsetY: offsetY

    };

}


// ============================================================
// INFERENCE
// ============================================================

async function runInference() {

    const prep =
        preprocess();


    const inputName =
        session.inputNames[0];


    const feeds = {};

    feeds[inputName] =
        prep.tensor;


    const results =
        await session.run(feeds);


    const outputName =
        session.outputNames[0];


    const output =
        results[outputName];


    return {
        output: output,
        scale: prep.scale,
        offsetX: prep.offsetX,
        offsetY: prep.offsetY
    };

}


// ============================================================
// YOLO POSTPROCESS
// ============================================================

function postprocess(
    output,
    scale,
    offsetX,
    offsetY
) {

    const data =
        output.data;


    const dims =
        output.dims;
    // Expected:
    //
    // [1, 9, 8400]
    //
    // 9 = 4 + 5 classes
    //
    // YOLO output layout:
    //
    // x
    // y
    // w
    // h
    // class0
    // class1
    // ...


    const numChannels =
        dims[1];

    const numBoxes =
        dims[2];

    const numClasses =
        numChannels - 4;


    const candidates = [];


    for (
        let i = 0;
        i < numBoxes;
        i++
    ) {

        const cx =
            data[
                0 * numBoxes + i
            ];

        const cy =
            data[
                1 * numBoxes + i
            ];

        const w =
            data[
                2 * numBoxes + i
            ];

        const h =
            data[
                3 * numBoxes + i
            ];


        let bestClass = -1;

        let bestScore = 0;


        for (
            let c = 0;
            c < numClasses;
            c++
        ) {

            const score =
                data[
                    (4 + c) *
                    numBoxes + i
                ];


            if (
                score > bestScore
            ) {

                bestScore =
                    score;

                bestClass =
                    c;

            }

        }


        if (
            bestScore <
            CONF_THRESHOLD
        ) {

            continue;

        }


        // Convert YOLO xywh
        // to x1,y1,x2,y2

        let x1 =
            cx - w / 2;

        let y1 =
            cy - h / 2;

        let x2 =
            cx + w / 2;

        let y2 =
            cy + h / 2;


        // Undo letterbox

        x1 =
            (x1 - offsetX)
            / scale;

        y1 =
            (y1 - offsetY)
            / scale;

        x2 =
            (x2 - offsetX)
            / scale;

        y2 =
            (y2 - offsetY)
            / scale;


        // Clamp

        x1 =
            Math.max(
                0,
                Math.min(
                    video.videoWidth,
                    x1
                )
            );

        y1 =
            Math.max(
                0,
                Math.min(
                    video.videoHeight,
                    y1
                )
            );

        x2 =
            Math.max(
                0,
                Math.min(
                    video.videoWidth,
                    x2
                )
            );

        y2 =
            Math.max(
                0,
                Math.min(
                    video.videoHeight,
                    y2
                )
            );


        candidates.push({

            x1: x1,

            y1: y1,

            x2: x2,

            y2: y2,

            score: bestScore,

            classId: bestClass

        });

    }


    return nms(
        candidates,
        IOU_THRESHOLD
    );

}


// ============================================================
// IOU
// ============================================================

function calculateIoU(
    a,
    b
) {

    const x1 =
        Math.max(
            a.x1,
            b.x1
        );

    const y1 =
        Math.max(
            a.y1,
            b.y1
        );

    const x2 =
        Math.min(
            a.x2,
            b.x2
        );

    const y2 =
        Math.min(
            a.y2,
            b.y2
        );


    const intersectionWidth =
        Math.max(
            0,
            x2 - x1
        );

    const intersectionHeight =
        Math.max(
            0,
            y2 - y1
        );


    const intersection =
        intersectionWidth *
        intersectionHeight;


    const areaA =
        (a.x2 - a.x1) *
        (a.y2 - a.y1);


    const areaB =
        (b.x2 - b.x1) *
        (b.y2 - b.y1);


    const union =
        areaA +
        areaB -
        intersection;


    if (
        union <= 0
    ) {

        return 0;

    }


    return (
        intersection /
        union
    );

}


// ============================================================
// NMS
// ============================================================

function nms(
    boxes,
    threshold
) {

    boxes.sort(
        (a, b) =>
            b.score -
            a.score
    );


    const selected = [];


    while (
        boxes.length > 0
    ) {

        const current =
            boxes.shift();


        selected.push(
            current
        );


        boxes =
            boxes.filter(
                box => {

                    if (
                        box.classId !==
                        current.classId
                    ) {

                        return true;

                    }


                    return (
                        calculateIoU(
                            current,
                            box
                        ) < threshold
                    );

                }
            );

    }


    return selected;

}


// ============================================================
// DRAW DETECTIONS
// ============================================================

function drawDetections(
    detections
) {

    overlayCtx.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );


    const scaleX =
        overlay.width /
        video.videoWidth;


    const scaleY =
        overlay.height /
        video.videoHeight;


    overlayCtx.lineWidth = 3;

    overlayCtx.font =
        "bold 16px Arial";


    detections.forEach(
        detection => {

            const x =
                detection.x1 *
                scaleX;

            const y =
                detection.y1 *
                scaleY;

            const width =
                (
                    detection.x2 -
                    detection.x1
                ) *
                scaleX;

            const height =
                (
                    detection.y2 -
                    detection.y1
                ) *
                scaleY;


            const className =
                CLASS_NAMES[
                    detection.classId
                ] ||
                `Class ${detection.classId}`;


            const confidence =
                (
                    detection.score *
                    100
                ).toFixed(1);


            const label =
                `${className} ${confidence}%`;


            overlayCtx.strokeStyle =
                "#00ff00";

            overlayCtx.strokeRect(
                x,
                y,
                width,
                height
            );


            const textWidth =
                overlayCtx.measureText(
                    label
                ).width;


            overlayCtx.fillStyle =
                "#00ff00";

            overlayCtx.fillRect(
                x,
                Math.max(
                    0,
                    y - 24
                ),
                textWidth + 12,
                24
            );


            overlayCtx.fillStyle =
                "#000";

            overlayCtx.fillText(
                label,
                x + 6,
                Math.max(
                    17,
                    y - 7
                )
            );

        }
    );


    objectsElement.textContent =
        detections.length;


    if (
        detections.length === 0
    ) {

        detectionsElement.textContent =
            "No detection";

        return;

    }


    detectionsElement.innerHTML =
        detections
            .map(
                detection => {

                    const className =
                        CLASS_NAMES[
                            detection.classId
                        ] ||
                        `Class ${detection.classId}`;


                    const confidence =
                        (
                            detection.score *
                            100
                        ).toFixed(1);


                    return `
                        <div>
                            🥭
                            <strong>
                                ${className}
                            </strong>
                            -
                            ${confidence}%
                        </div>
                    `;

                }
            )
            .join("");

}


// ============================================================
// FPS
// ============================================================

function updateFPS() {

    frameCounter++;

    const now =
        performance.now();


    const elapsed =
        now - lastTime;


    if (
        elapsed >= 1000
    ) {

        fps =
            frameCounter *
            1000 /
            elapsed;


        fpsElement.textContent =
            fps.toFixed(1);


        frameCounter = 0;

        lastTime = now;

    }

}


// ============================================================
// MAIN REALTIME LOOP
// ============================================================

let lastInferenceTime = 0;

// YOLO chạy tối đa khoảng 10 lần/giây
const INFERENCE_INTERVAL = 100;

async function inferenceLoop(timestamp) {

    if (!running) {
        return;
    }

    if (video.readyState >= 2) {

        // Chỉ chạy YOLO mỗi 100 ms
        if (
            timestamp - lastInferenceTime >=
            INFERENCE_INTERVAL
        ) {

            lastInferenceTime = timestamp;

            try {

                const result =
                    await runInference();

                const detections =
                    postprocess(
                        result.output,
                        result.scale,
                        result.offsetX,
                        result.offsetY
                    );

                drawDetections(
                    detections
                );

                updateFPS();

            } catch (error) {

                console.error(
                    "Inference error:",
                    error
                );

                setStatus(
                    "Inference error - see Console"
                );
            }
        }
    }

    requestAnimationFrame(
        inferenceLoop
    );
}


// ============================================================
// BUTTONS
// ============================================================

loadButton.addEventListener(
    "click",
    loadModel
);


cameraButton.addEventListener(
    "click",
    startCamera
);


stopButton.addEventListener(
    "click",
    stopCamera
);


// ============================================================
// INITIAL STATUS
// ============================================================

setStatus(
    "Ready - load model"
);