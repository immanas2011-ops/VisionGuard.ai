👁️ VisionGuard AIReal-time, edge-accelerated drowsiness detection, gaze tracking, and multimodal ocular health screening.VisionGuard AI combines low-latency, on-device computer vision with server-side LLM multimodal intelligence. By leveraging MediaPipe's 478-point face mesh, TensorFlow.js edge models, and Google's Gemini Multimodal API, VisionGuard AI delivers zero-latency local fatigue intervention alongside deep AI-driven ocular health diagnostics.🌟 Key Features⚡ On-Device Low-Latency Tracking: Local landmark estimation via MediaPipe FaceLandmarker running on WebAssembly (WebGL2 GPU / CPU delegate selection).🎯 478-Point Facial & Iris Mapping: High-precision tracking of iris vectors, eye aspect ratio ($EAR$), gaze direction, and ARKit facial blendshapes.📈 Adaptive Baseline Calibration: Dynamic baseline seeding that adapts blink metrics to diverse facial structures, eye shapes, and glasses.📊 Real-Time Biometric Telemetry:Rolling 8-second SVG eye openness waveform ($50\%$ threshold).Rolling Blinks Per Minute ($BPM$) & average blink duration ($ms$).Rolling 60-second Percentage of Eye Closure ($PERCLOS$) and composite Alertness Index ($0 - 100\%$).2D vector compass gaze classification (Focused, Looking Left/Right/Up/Down, Eyes Closed).🚨 Multi-Modal Intervention: Visual HUD banners, edge-pulsing screen borders, Web Audio chimes, and persistent session event logging.🧠 Multimodal AI Diagnostic Reports: Automatic rolling 5-frame capture ($1.25\text{s}$ interval) sent to an Express backend powered by Gemini multimodal model series for full ocular risk scoring and health analysis.🏗️ System Architecture               +-------------------------------------------------------+
               |                  Webcam Input Stream                  |
               |                     (1280 x 720)                      |
               +---------------------------+---------------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
     +------------------------------+             +------------------------------+
     |   On-Device Tracking Loop    |             |    Rolling 5-Frame Buffer    |
     |  (MediaPipe + TF.js WASM)    |             |    (1.25s Frame Intervals)    |
     +--------------+---------------+             +--------------+---------------+
                    |                                             |
        +-----------+-----------+                                 | On-Demand Scan / Burst
        |           |           |                                 v
        v           v           v                 +------------------------------+
     Canvas       Live     Drowsiness             |       Express Backend        |
    Overlay   Telemetry   Intervention            |    (POST /api/diagnostic)    |
   (Vectors)  (BPM/EAR)     (Audio/HUD)           +--------------+---------------+
                                                                 |
                                                                 v
                                                  +------------------------------+
                                                  |    Gemini Multimodal API     |
                                                  |   (Structured Report + JSON) |
                                                  +------------------------------+
⚙️ Core Functional Stages1. Continuous Optical Capture & Background MonitoringResilient Engine: Video feed initializes at $1280 \times 720$ resolution inside a persistent workspace container.Background Operation: The scanner viewport remains mounted underneath UI overlays (How_Works, Impact_Future, Core_Engine, System_Specs, India_Stats), allowing continuous monitoring and live preview telemetry while navigating between app tabs.2. On-Device Real-Time Eye Tracking & CalibrationDynamic $EAR$ Seeding: Automatically seeds open-eye $EAR$ baselines from initial frames and blends eyeBlink blendshape scores for consistent tracking across lighting conditions and eyewear.Edge Classification: Center-cropped frames are evaluated at $\approx 5\text{ Hz}$ via TensorFlow.js to factor image-based drowsiness probability into the local fatigue score.3. Threshold Modes & Fatigue InterventionSensitivity Modes: Toggle between Standard and Strict sensitivity modes on the fly via the scanner HUD.Intervention Cascade: Sustained eye closure or high $PERCLOS$ scores trigger visual warnings, screen-edge pulsing, Web Audio alert chimes, and timestamped terminal logs.4. 5-Frame AI Diagnostic Report & ExportServer-Side Reasoning: Captured 5-frame JPEG bursts are analyzed by Gemini models (gemini-3.8-flash with automatic fallback to gemini-3.1-flash-lite / gemini-flash-latest).Structured Output: Returns schema-validated JSON covering:Overall Eye Health Status & Score ($0 - 100\%$)Drowsiness Risk Level & Sore/Red Eye SeverityEstimated Sensor Metrics ($PERCLOS$, $BPM$, Gaze Stability) & AI Intelligence SummaryExport & Audit: Complete audit trail showing frame thumbnails (01–05) and one-click JSON payload export.🛠️ Tech StackFrontend: HTML5, React / Modern Web APIs, Canvas Rendering, Web Audio API, WebGL2Computer Vision & ML: MediaPipe FaceLandmarker (478-Point Face Mesh), TensorFlow.jsBackend: Node.js, ExpressAI & Multimodal LLM: Google Gemini API (gemini-3.8-flash, gemini-3.1-flash-lite)🚦 Getting StartedPrerequisitesNode.js (v18.x or higher)NPM or YarnModern Web Browser with WebGL2 support & Webcam permissionsGoogle Gemini API KeyEnvironment SetupCreate a .env file in your root backend directory:PORT=5000
GEMINI_API_KEY=your_gemini_api_key_here
InstallationClone the repository:git clone https://github.com/your-username/visionguard-ai.git
cd visionguard-ai
Install backend dependencies:cd backend
npm install
Install frontend dependencies:cd ../frontend
npm install
Running the ApplicationStart the Express Server:cd backend
npm start
Start the Frontend Development Server:cd frontend
npm run dev
Open http://localhost:3000 (or the provided port) in your browser, grant camera permissions, and begin session tracking.📡 API ReferencePOST /api/diagnosticSubmits captured biometric telemetry and a 5-frame image burst for multimodal analysis.Request Body{
  "telemetry": {
    "bpm": 18,
    "perclos": 0.12,
    "alertnessScore": 88,
    "gazeDirection": "Focused"
  },
  "frames": [
    "data:image/jpeg;base64,...",
    "data:image/jpeg;base64,...",
    "data:image/jpeg;base64,...",
    "data:image/jpeg;base64,...",
    "data:image/jpeg;base64,..."
  ]
}
Response Example{
  "eyeHealthScore": 92,
  "drowsinessStatus": "Alert",
  "soreRedEyeSeverity": "Low",
  "metrics": {
    "perclos": "12%",
    "blinkRate": "18 BPM",
    "gazeStability": "High"
  },
  "summary": "User shows high visual focus with healthy eye aspect ratio variations. No indicators of ocular strain or fatigue."
}
