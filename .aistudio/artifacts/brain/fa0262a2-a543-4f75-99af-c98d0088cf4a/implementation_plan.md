# VisionGuard AI — Real-Time Eye Tracking & Modern Studio Redesign

VisionGuard AI will replace its static simulated reticles with genuine 30–60 FPS on-device iris, eyelid, blink, and gaze tracking powered by computer-vision facial landmarks, paired with live drowsiness telemetry, optional audio alerts, and session history. The entire interface will be redesigned into a sleek Dark Creator Studio aesthetic inspired by YouTube Studio and Instagram Dark while preserving the existing server-side Gemini biometric diagnostic report workflow.

---

## User Review & Critical Decisions

> [!IMPORTANT]
> All Phase 1 preferences have been incorporated below. Because the workspace layout question was skipped, we have selected the **YouTube Studio Theater + Side Live Analytics** layout as the recommended default. Please confirm or request adjustments before we begin implementation.

- **Visual Direction (Confirmed)**: Dark creator studio palette inspired by YouTube Studio and Instagram Dark (`#0F0F0F` canvas, `#18181B` hairline panels, crisp neutral typography, calibrated rose-crimson `#F43F5E` / `#EF4444` accent).
- **Camera Overlay Style (Confirmed)**: Clean real-time iris rings, dynamic eyelid/blink arcs, and a directional gaze vector rendered live on the video canvas.
- **Workspace Arrangement (Recommended Default)**: YouTube Studio-style 16:9 primary camera theater on the left with ambient backlight glow and floating media controls, paired with a live analytics and session-events column on the right and an expandable AI diagnostic report below.
- **Real-Time Tracking Suite (Confirmed)**: Live blink rate, rolling PERCLOS, dual-eye openness waveform chart, real-time gaze direction and focus timer, visual + optional Web Audio drowsiness alert, and a timestamped session history of fatigue events.

---

## 1. Overview & Core Concept

- **What It Does**: Streams live webcam frames through two synchronized on-device models—a 478-point facial/iris landmarker for sub-frame eye geometry and the existing Teachable Machine drowsiness classifier—while retaining on-demand multi-frame deep diagnostic analysis via the server-side Gemini endpoint.
- **Target Audience & Context**: Drivers, safety researchers, students, and everyday computer users monitoring eye fatigue, attention drift, and drowsiness in real time directly in the browser.
- **Key Value**: Transforms eye tracking from a decorative loop into responsive, verifiable biometric telemetry—every reticle tracks the user's real pupils, every metric reflects measured eyelid motion, and every fatigue alert is backed by real time-series data.

---

## 2. User Experience & Visual Design

### Key User Flows

1. **Ready State (Pre-Camera)**: User lands on a clean dark studio workspace with a single-row navigation bar (`Monitor`, `How It Works`, `Impact`, `Safety Data`) and a focused camera theater stage displaying a clear **Start Camera** call to action—no cluttered terminal boilerplate or fake static metrics.
2. **Live Tracking & Overlay**: Clicking **Start Camera** initializes the webcam stream, warms up the facial/iris landmarker and Teachable Machine classifier, and immediately locks crisp iris rings, pupil dots, blink-reactive eyelid curves, and a live gaze vector arrow onto the user's eyes in real time.
3. **Continuous Biometric Telemetry**: As the user blinks, looks around, or gets drowsy, the side analytics panel updates smoothly with live Blink Rate (blinks/min), rolling 60-second PERCLOS (%), Left/Right Eye Openness bars + real-time waveform sparkline, and a Gaze Direction + Focus Timer readout.
4. **Drowsiness Alert & Session History**: When sustained eye closure, elevated PERCLOS, or high fatigue probability occurs, an unmistakable visual warning state activates on the stage and analytics header (with an optional gentle Web Audio chime controlled by a mute toggle in the camera toolbar), and the event is appended to the **Session History** feed.
5. **Deep AI Diagnostic Report**: Once 5 live frames are buffered, clicking **Run AI Report** submits the 5-frame burst to the server-side Gemini diagnostic endpoint and renders the structured biometric report (Eye Health score, Drowsiness risk, Soreness/Redness assessment, PERCLOS/Blink/Gaze estimates, and Summary) directly below the theater stage.
6. **Informational Views**: Selecting `How It Works`, `Impact`, or `Safety Data` opens clean, scannable editorial views styled consistently with the dark studio system.

### Visual Identity & Theme

- **Aesthetic Direction**: Modern Dark Creator Studio (YouTube Studio theater focus + Instagram Dark surface refinement). High-contrast media viewport, disciplined whitespace, zero fake cyberpunk prefixes, zero decorative pulsing dots, and zero pill-badge clutter.
- **Color Palette & Mood** (60-30-10 discipline):
  - *60% Dominant Canvas (`--background`)*: Deep obsidian `#0F0F0F` for the page stage and theater surround.
  - *30% Structural Surfaces (`--surface` / `--border`)*: Elevated charcoal `#18181B` and `#202024` panels bounded by subtle `1px solid rgba(255, 255, 255, 0.08)` hairline borders (`--border`).
  - *10% Calibrated Accents*: Rose-crimson `#F43F5E` / `#EF4444` (`--primary`) reserved for primary actions, live iris lock rings, and high-risk alerts; semantic emerald `#10B981` for alert/focused states and amber `#F59E0B` for fatigue warnings (always paired with explicit text labels).
- **Typography & Hierarchy**:
  - *Display & UI Body*: `Plus Jakarta Sans` (clean geometric legibility, `text-wrap: balance` on headings, single-line `whitespace-nowrap` controls).
  - *Numeric Telemetry & Timers*: `JetBrains Mono` with `font-variant-numeric: tabular-nums` so fluctuating blink counts, PERCLOS percentages, and focus timers never jitter horizontally.
- **Component Styling & Spatial Composition**:
  - Strict 3-zone Top Bar (`VisionGuard` wordmark — 4 single-line links — 1 primary action button).
  - Single-level card depth without nested card-in-card borders or thick left-edge accent stripes (`border-l-4`).
  - Floating translucent control dock inside the bottom edge of the 16:9 video viewport (Play/Pause Camera, Overlay Toggle, Audio Alert Toggle, Run AI Report), using a measured dark gradient scrim for WCAG AA legibility over any lighting condition.

### Interactive Feedback & Motion

- Sub-200ms compositor-only transitions (`transform`, `opacity`) on control hover states, drawer switches, and alert banners.
- Canvas tracking overlay renders smoothly at display refresh rate (`requestAnimationFrame`), interpolating minor landmark jitter lightly so iris rings feel locked and responsive without vibrating.
- Explicit loading, camera-permission-denied, no-face-detected, and API-error states with clear recovery actions.

---

## 3. Key Product Decisions & Trade-Offs

- **Decision 1 — Real-Time Eye Landmark Engine**:
  - *Chosen Approach*: Run **MediaPipe FaceLandmarker** (`outputFaceBlendshapes: true`, 478 landmarks including iris centers/perimeters `468–477`) in `VIDEO` mode on-device in the browser alongside the existing Teachable Machine classifier.
  - *Why*: Provides true 30–60 FPS pupil/iris tracking, native blink blendshapes (`eyeBlinkLeft`, `eyeBlinkRight`), and directional gaze blendshapes (`eyeLookIn/Out/Up/Down`) with zero server round-trip latency and full camera privacy.
  - *Alternatives Considered*: Sending continuous webcam frames to the backend Gemini API (rejected due to latency, bandwidth cost, and inability to track smoothly at 30+ FPS) or pure Teachable Machine classification (cannot localize pupil coordinates on screen).

- **Decision 2 — Gaze & Drowsiness Calculation Strategy**:
  - *Chosen Approach*: Combine geometric Eye Aspect Ratio (EAR) from eyelid contour landmarks with MediaPipe blink/gaze blendshapes, feeding a sliding 60-second window for real PERCLOS (%) and blink-edge detection for BPM.
  - *Why*: Pairing geometric landmark ratios with neural blendshape confidence prevents false blink triggers during head turns or glasses glare.
  - *Alternatives Considered*: Relying solely on raw pixel brightness thresholding (highly brittle under varying room lighting).

- **Decision 3 — Non-Intrusive Audio Fatigue Alert**:
  - *Chosen Approach*: Synthesize a clean two-tone alert chime using the browser's built-in Web Audio API, gated behind a visible mute/unmute toggle in the camera control dock.
  - *Why*: Requires zero external audio files, obeys browser autoplay policies (initialized on user click to start camera), and works reliably offline.
  - *Alternatives Considered*: Hosting external `.mp3` alert files (unnecessary asset overhead and potential loading delay).

---

## 4. Technical Architecture & Data Strategy

### System Architecture & Data Flow

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          Browser Client (React)                           │
│                                                                           │
│  ┌──────────────────────┐       ┌──────────────────────────────────────┐  │
│  │  Webcam Video Feed   │──────▶│      Real-Time Vision Pipeline       │  │
│  │  (getUserMedia 16:9) │       │  • MediaPipe FaceLandmarker (Iris,   │  │
│  └──────────┬───────────┘       │    Eyelid Contours, Blendshapes)     │  │
│             │                   │  • Teachable Machine Classifier      │  │
│             ▼                   └──────────────────┬───────────────────┘  │
│  ┌──────────────────────┐                          │                      │
│  │  Mirrored Viewport   │◀─────────────────────────┤ (30-60 FPS)          │
│  │  Video + HUD Canvas  │   Iris Rings, Eyelid     │                      │
│  │  + Floating Controls │   Arcs & Gaze Vector     ▼                      │
│  └──────────┬───────────┘       ┌──────────────────────────────────────┐  │
│             │                   │     Biometric Telemetry Engine       │  │
│             │ (5-Frame Buffer   │  • Blink Detector & BPM Counter      │  │
│             │  every 1.25s)     │  • 60s Rolling PERCLOS & Openness    │  │
│             │                   │  • Gaze Direction & Focus Timer      │  │
│             │                   │  • Fatigue Alert + Session History   │  │
│             ▼                   └──────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │            AI Diagnostic Report Panel (On-Demand Fetch)             │  │
│  └──────────────────────────────────┬──────────────────────────────────┘  │
└─────────────────────────────────────┼─────────────────────────────────────┘
                                      │ POST /api/diagnostic (5 JPEG frames)
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                     Express Server (Server-Side Gemini)                   │
│  • Validates frame payload                                                │
│  • Calls Gemini API (@google/genai) with structured JSON responseSchema   │
│  • Returns eyeHealth, drowsiness, soreEyes, redEyes, eyeMetrics, summary  │
└───────────────────────────────────────────────────────────────────────────┘
```

### Data Model & State

- **Live Eye Frame Telemetry**: Ephemeral per-frame metrics containing left/right iris coordinates (normalized `0..1`), iris radii, left/right openness (`0..1`), gaze vector `(dx, dy)`, categorical gaze state (`Focused`, `Left`, `Right`, `Up`, `Down`, `Away`), and face-detected flag.
- **Rolling Session Metrics**:
  - `blinkRateBpm`: Calculated from detected blink transitions across a rolling 60-second window.
  - `perclosPercent`: Percentage of samples in the last 60 seconds where average eye openness fell below the closure threshold (`< 0.25`).
  - `opennessHistory`: Ring buffer of recent eye openness samples rendered as a real-time SVG waveform.
  - `focusSeconds` / `totalTrackedSeconds`: Active session timers tracking gaze engagement.
- **Alert & Event Log**: Timestamped `SessionEvent` entries (`eventType`: `'drowsiness' | 'extended-closure' | 'gaze-away' | 'diagnostic'`, `severity`, `label`, `timestamp`) populated strictly from real detected conditions.
- **Diagnostic Report**: Preserved structured report schema returned from `POST /api/diagnostic` (`eyeHealth`, `drowsiness`, `soreEyes`, `redEyes`, `eyeMetrics`, `summary`).

### Interactive Component & State Mapping

- **Top Navigation Bar**: Switches active view state between live `Monitor` and informational overlays/views (`How It Works`, `Impact`, `Safety Data`), plus triggers the primary **Start Camera** or **Run AI Report** action.
- **Theater Video Stage & Floating Control Dock**: Handles camera start/stop (`getUserMedia` + clean track teardown), eye-tracking overlay toggle (iris rings & gaze vector visibility), audio alert mute toggle, and manual **Run AI Report** trigger with progress feedback while buffering/analyzing 5 frames.
- **Live Analytics & Waveform Panel**: Reads smoothed telemetry refs/state at UI cadence (~10–15 Hz for React text/charts while canvas draws at 60 FPS) to keep DOM renders lightweight and free of stutter.
- **Drowsiness Alert Banner & Web Audio Synthesizer**: Activates automatically when eyes remain closed past threshold (`> 1.5s`), PERCLOS crosses risk threshold (`>= 20%`), or Teachable Machine drowsy confidence spikes, playing an optional chime if unmuted and recording a deduplicated entry in Session History.
- **Diagnostic Report Section**: Displays loading skeleton/progress state during `POST /api/diagnostic`, renders the structured report with clear health indicators, handles error states gracefully, and supports resetting to run a new scan.

### Key Sequences & Integrations

- **On-Device Vision Pipeline**: Install `@mediapipe/tasks-vision` and initialize `FaceLandmarker` with WASM vision tasks + `face_landmarker.task` (GPU delegate with CPU fallback if WebGL is unavailable) alongside `@teachablemachine/image`. Detect on each video frame, map coordinates cleanly to the mirrored viewport canvas, and compute blendshape + geometric metrics locally.
- **Server-Side Deep Analysis**: Keep the existing Express `POST /api/diagnostic` integration untouched so all Gemini multimodal reasoning continues executing securely on the server.
