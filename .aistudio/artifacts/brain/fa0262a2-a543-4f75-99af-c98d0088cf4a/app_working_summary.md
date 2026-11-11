# How VisionGuard AI Works

**VisionGuard AI** is a real-time computer vision system designed for drowsiness detection, gaze monitoring, and ocular health screening. It pairs low-latency on-device eye tracking with on-demand multimodal AI analysis on the server.

---

## 1. End-to-End Architecture

```text
Webcam Stream
   ├──► On-Device Tracking Loop (MediaPipe 478-Point Face Mesh + Blendshapes + TF.js)
   │       ├──► Canvas Overlay (Iris rings, eyelid arcs, face brackets, gaze vector)
   │       ├──► Live Telemetry (BPM, PERCLOS, Alertness Score, Gaze, Focus Streak)
   │       └──► Drowsiness Intervention (Visual HUD banner, edge pulse, Web Audio chime)
   │
   └──► Rolling 5-Frame Buffer (1.25s interval / fast burst on scan)
           └──► Express Backend (POST /api/diagnostic)
                   └──► Gemini Multimodal API ──► Structured Biometric Report + JSON Export
```

---

## 2. Core Functional Stages

### A. Continuous Optical Capture & Background Monitoring
- Accessing the webcam initializes a `1280×720` video stream inside the persistent main workspace.
- Because the scanner viewport stays mounted underneath briefing/info overlays (`How_Works`, `Impact_Future`, `Core_Engine`, `System_Specs`, `India_Stats`), eye tracking, alerts, and a **Monitor_Live** mini-preview strip continue operating uninterrupted while browsing other tabs.

### B. On-Device Real-Time Eye Tracking & Calibration
- **478-Point Iris & Facial Mapping**: Runs locally via WebAssembly (`FaceLandmarker`) using bundled model and WASM assets (with automatic WebGL2 GPU or CPU delegate selection).
- **Adaptive Baseline Calibration**: Seeds each user's open-eye Eye Aspect Ratio (`EAR`) baseline from the first open frame and blends facial `eyeBlink` blendshape scores so blink metrics adapt reliably across narrow eyes, wide eyes, and glasses.
- **Live Biometric Telemetry**:
  - **Eye Openness & Waveform**: Left/right eye openness bars paired with an 8-second rolling SVG waveform (`50%` closure threshold).
  - **Blink Rate & Duration**: Tracks total blinks, rolling blinks per minute (`BPM`), and average blink duration (`ms`).
  - **PERCLOS & Alertness Score**: Calculates rolling 60-second percentage of eye closure (`PERCLOS`) and a composite `0–100%` **Alertness** index.
  - **Gaze Tracking & Focus Timer**: Fuses ARKit gaze blendshapes with geometric iris offset to classify gaze direction (`Focused`, `Looking Left/Right/Up/Down`, `Eyes Closed`) on a 2D vector compass while timing active attention and focus streaks.
- **Edge Classification (TensorFlow.js)**: Evaluates center-cropped frames at ~5 Hz using Teachable Machine to factor visual drowsiness probability into local fatigue scoring.

### C. Threshold Modes & Fatigue Intervention
- **Std vs. Strict Sensitivity**: Users can switch fatigue thresholds on the fly from the scanner HUD bar (`Std` for normal operation, `Strict` for higher sensitivity).
- **Multi-Modal Alerts**: Sustained eye closures, high PERCLOS, or high fatigue scores trigger color-coded HUD warning/critical banners, a pulsing red viewport border on critical alerts, optional synthesized Web Audio chimes, and timestamped entries in the session event history and terminal log.
- **Manual Controls**: Includes HUD toggles for iris overlay visibility, video mirroring, audio mute, telemetry baseline reset (`Reset`), and stopping the feed.

### D. 5-Frame AI Diagnostic Report & Export
- **Rolling & On-Demand Frame Capture**: Captures a clean square JPEG frame every `1.25` seconds into a 5-frame rolling buffer. Clicking **Run Full Report** (or **Generate Biometric Health Report**) rapidly fills any missing frames in the sequence and submits the 5-frame burst plus live sensor telemetry to `/api/diagnostic`.
- **Server-Side Gemini Reasoning**: The Express backend keeps Gemini API credentials strictly on the server, queries `gemini-3.8-flash` (with automatic failover to `gemini-3.1-flash-lite` and `gemini-flash-latest`), and returns a schema-validated JSON report covering:
  - **Eye Health Status & Score (`0–100%`)**
  - **Drowsiness Check & Risk Level**
  - **Sore Eyes & Red Eyes Severity**
  - **Estimated Eye Metrics** (`PERCLOS`, `Blink Rate`, `Gaze Stability`) & **Intelligence Summary**
- **Audit Trail & Export**: Displays the 5 captured thumbnails (`01`–`05`) inside the verified report card, offers a one-click **Export** button to download telemetry + diagnostics as JSON, and surfaces clear inline error banners with a retry action if network or API issues occur.
