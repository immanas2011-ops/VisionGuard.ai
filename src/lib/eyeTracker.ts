import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import type { FaceLandmarkerResult, NormalizedLandmark } from '@mediapipe/tasks-vision';
import type {
  EyeTelemetry,
  FatigueLevel,
  GazeDirection,
  SensitivityMode,
} from '../types/vision';

const LOCAL_WASM_PATH = '/mediapipe/wasm';
const CDN_WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/wasm';
const LOCAL_MODEL_PATH = '/models/face_landmarker.task';
const CDN_MODEL_PATH =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

// MediaPipe 478-landmark indices (subject's perspective)
const RIGHT_EYE = {
  outer: 33,
  inner: 133,
  upper: 159,
  lower: 145,
  upperA: 158,
  upperB: 160,
  lowerA: 153,
  lowerB: 144,
  irisCenter: 468,
  irisPerimeter: [469, 470, 471, 472],
};

const LEFT_EYE = {
  inner: 362,
  outer: 263,
  upper: 386,
  lower: 374,
  upperA: 385,
  upperB: 387,
  lowerA: 373,
  lowerB: 380,
  irisCenter: 473,
  irisPerimeter: [474, 475, 476, 477],
};

const FACE_BOUNDS = {
  top: 10,
  bottom: 152,
  left: 234,
  right: 454,
};

let landmarkerPromise: Promise<FaceLandmarker> | null = null;

const BENIGN_WASM_LOG_PATTERN =
  /^(?:INFO:\s*.*(?:TensorFlow|TFLite|XNNPACK|MediaPipe|delegate|runtime)|WARNING:\s*.*(?:absl::InitializeLog|Feedback manager|delegate|TensorFlow|MediaPipe)|[IW]\d{4}\s+\d{2}:\d{2}:\d{2}\.\d+|Created TensorFlow Lite XNNPACK delegate)/i;

export function installMediaPipeLogFilter(): void {
  if (typeof window === 'undefined' || typeof console === 'undefined') return;
  const g = globalThis as Record<PropertyKey, unknown>;
  const flagKey = Symbol.for('visionguard.mediapipeLogFilterInstalled');
  if (g[flagKey]) return;
  g[flagKey] = true;

  const originalError = console.error.bind(console);
  console.error = (...args: unknown[]) => {
    const msg = args
      .filter((a): a is string => typeof a === 'string')
      .join(' ')
      .trim();
    if (msg && BENIGN_WASM_LOG_PATTERN.test(msg)) {
      console.debug(...args);
      return;
    }
    originalError(...args);
  };
}

function canUseWebGL2(): boolean {
  if (typeof document === 'undefined') return false;
  try {
    const canvas = document.createElement('canvas');
    return Boolean(canvas.getContext('webgl2'));
  } catch {
    return false;
  }
}

export async function getFaceLandmarker(): Promise<FaceLandmarker> {
  installMediaPipeLogFilter();
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      let vision;
      try {
        vision = await FilesetResolver.forVisionTasks(LOCAL_WASM_PATH);
      } catch {
        vision = await FilesetResolver.forVisionTasks(CDN_WASM_PATH);
      }

      const createWithConfig = async (modelAssetPath: string, delegate: 'GPU' | 'CPU') => {
        return FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath,
            delegate,
          },
          outputFaceBlendshapes: true,
          runningMode: 'VIDEO',
          numFaces: 1,
          minFaceDetectionConfidence: 0.45,
          minFacePresenceConfidence: 0.45,
          minTrackingConfidence: 0.45,
        });
      };

      const delegates: ('GPU' | 'CPU')[] = canUseWebGL2() ? ['GPU', 'CPU'] : ['CPU'];
      const modelPaths = [LOCAL_MODEL_PATH, CDN_MODEL_PATH];
      let lastError: unknown = null;

      for (const modelPath of modelPaths) {
        for (const delegate of delegates) {
          try {
            return await createWithConfig(modelPath, delegate);
          } catch (err) {
            lastError = err;
          }
        }
      }

      throw lastError instanceof Error
        ? lastError
        : new Error('Unable to initialize face landmark tracking engine.');
    })().catch((err) => {
      landmarkerPromise = null;
      throw err;
    });
  }
  return landmarkerPromise;
}

function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

function dist2D(a: NormalizedLandmark, b: NormalizedLandmark): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function getBlendshapeScore(
  categories: { categoryName: string; score: number }[] | undefined,
  name: string
): number | null {
  if (!categories) return null;
  const found = categories.find((c) => c.categoryName === name);
  return found ? found.score : null;
}

function computeRawEar(landmarks: NormalizedLandmark[], eye: typeof RIGHT_EYE): number | null {
  const outer = landmarks[eye.outer];
  const inner = landmarks[eye.inner];
  const upper = landmarks[eye.upper];
  const lower = landmarks[eye.lower];
  const upperA = landmarks[eye.upperA];
  const lowerA = landmarks[eye.lowerA];
  const upperB = landmarks[eye.upperB];
  const lowerB = landmarks[eye.lowerB];

  if (!outer || !inner || !upper || !lower) return null;

  const width = Math.max(dist2D(outer, inner), 1e-4);
  const vCenter = dist2D(upper, lower);
  const vA = upperA && lowerA ? dist2D(upperA, lowerA) : vCenter;
  const vB = upperB && lowerB ? dist2D(upperB, lowerB) : vCenter;
  return (vCenter + vA + vB) / (3 * width);
}

function getCoverTransform(
  videoWidth: number,
  videoHeight: number,
  canvasWidth: number,
  canvasHeight: number,
  mirrored: boolean
) {
  const videoAspect = videoWidth / Math.max(videoHeight, 1);
  const boxAspect = canvasWidth / Math.max(canvasHeight, 1);

  let renderW = canvasWidth;
  let renderH = canvasHeight;
  let offsetX = 0;
  let offsetY = 0;

  if (videoAspect > boxAspect) {
    renderH = canvasHeight;
    renderW = renderH * videoAspect;
    offsetX = (canvasWidth - renderW) / 2;
  } else {
    renderW = canvasWidth;
    renderH = renderW / videoAspect;
    offsetY = (canvasHeight - renderH) / 2;
  }

  return {
    toX: (nx: number) => offsetX + (mirrored ? 1 - nx : nx) * renderW,
    toY: (ny: number) => offsetY + ny * renderH,
    renderW,
    renderH,
  };
}

interface SmoothedEyePoint {
  x: number;
  y: number;
  r: number;
}

export class EyeTrackingProcessor {
  private blinkCount = 0;
  private blinkTimestamps: number[] = [];
  private blinkDurationsMs: number[] = [];
  private isCurrentlyClosed = false;
  private closedStartTime: number | null = null;
  private closureSamples: { t: number; closed: boolean }[] = [];
  private opennessHistory: number[] = [];
  private lastWaveSampleAt = 0;
  private sessionStartTime = 0;
  private lastFrameTime = 0;
  private focusAccumMs = 0;
  private currentFocusStreakMs = 0;
  private bestFocusStreakMs = 0;
  private smoothedFps = 0;
  private baselineEar = 0.26;
  private calibratedSamples = 0;
  private drowsinessEpisodes = 0;
  private wasCriticalFatigue = false;

  private smoothedGazeX = 0;
  private smoothedGazeY = 0;
  private smoothedRightIris: SmoothedEyePoint | null = null;
  private smoothedLeftIris: SmoothedEyePoint | null = null;

  public reset() {
    const now = performance.now();
    this.blinkCount = 0;
    this.blinkTimestamps = [];
    this.blinkDurationsMs = [];
    this.isCurrentlyClosed = false;
    this.closedStartTime = null;
    this.closureSamples = [];
    this.opennessHistory = [];
    this.lastWaveSampleAt = 0;
    this.sessionStartTime = now;
    this.lastFrameTime = now;
    this.focusAccumMs = 0;
    this.currentFocusStreakMs = 0;
    this.bestFocusStreakMs = 0;
    this.smoothedFps = 0;
    this.baselineEar = 0.26;
    this.calibratedSamples = 0;
    this.drowsinessEpisodes = 0;
    this.wasCriticalFatigue = false;
    this.smoothedGazeX = 0;
    this.smoothedGazeY = 0;
    this.smoothedRightIris = null;
    this.smoothedLeftIris = null;
  }

  public processAndDraw(params: {
    result: FaceLandmarkerResult | null;
    video: HTMLVideoElement;
    canvas: HTMLCanvasElement;
    mirrored: boolean;
    showOverlay: boolean;
    sensitivity: SensitivityMode;
    tmDrowsyScore: number;
    now: number;
  }): EyeTelemetry {
    const {
      result,
      video,
      canvas,
      mirrored,
      showOverlay,
      sensitivity,
      tmDrowsyScore,
      now,
    } = params;

    if (this.sessionStartTime === 0) {
      this.sessionStartTime = now;
      this.lastFrameTime = now;
    }

    const dtMs = clamp(now - this.lastFrameTime, 0, 250);
    this.lastFrameTime = now;
    if (dtMs > 0) {
      const instantFps = 1000 / dtMs;
      this.smoothedFps =
        this.smoothedFps === 0 ? instantFps : this.smoothedFps * 0.88 + instantFps * 0.12;
    }

    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    const landmarks = result?.faceLandmarks?.[0];
    const categories = result?.faceBlendshapes?.[0]?.categories;

    if (!landmarks || landmarks.length < 478) {
      this.smoothedRightIris = null;
      this.smoothedLeftIris = null;
      this.isCurrentlyClosed = false;
      this.closedStartTime = null;
      this.currentFocusStreakMs = 0;
      this.wasCriticalFatigue = false;

      if (now - this.lastWaveSampleAt >= 100) {
        this.lastWaveSampleAt = now;
        this.opennessHistory = [...this.opennessHistory, 0].slice(-80);
      }

      const sessionSeconds = Math.max(0, (now - this.sessionStartTime) / 1000);
      const focusSeconds = this.focusAccumMs / 1000;
      const focusRatioPercent =
        sessionSeconds > 0.5 ? clamp((focusSeconds / sessionSeconds) * 100, 0, 100) : 100;

      return {
        faceDetected: false,
        leftOpenness: 0,
        rightOpenness: 0,
        avgOpenness: 0,
        isBlinking: false,
        closedDurationMs: 0,
        blinkCount: this.blinkCount,
        blinkRateBpm: this.computeBpm(now),
        avgBlinkDurationMs: this.computeAvgBlinkDuration(),
        perclosPercent: this.computePerclos(),
        alertnessScore: 100,
        drowsinessEpisodes: this.drowsinessEpisodes,
        gazeX: 0,
        gazeY: 0,
        gazeDirection: 'No Face',
        focusSeconds,
        focusStreakSeconds: 0,
        bestFocusStreakSeconds: this.bestFocusStreakMs / 1000,
        sessionSeconds,
        focusRatioPercent,
        fatigueLevel: 'nominal',
        alertReason: null,
        opennessHistory: this.opennessHistory,
        fps: Math.round(this.smoothedFps),
      };
    }

    // Subject's right/left blink blendshape scores
    const blinkRightScore = getBlendshapeScore(categories, 'eyeBlinkRight');
    const blinkLeftScore = getBlendshapeScore(categories, 'eyeBlinkLeft');
    const rightEar = computeRawEar(landmarks, RIGHT_EYE) ?? this.baselineEar;
    const leftEar = computeRawEar(landmarks, LEFT_EYE) ?? this.baselineEar;
    const avgEar = (rightEar + leftEar) / 2;

    const avgBlinkBlend =
      blinkRightScore !== null && blinkLeftScore !== null
        ? (blinkRightScore + blinkLeftScore) / 2
        : blinkRightScore ?? blinkLeftScore ?? 0;

    // Fast initial calibration followed by slow baseline tracking (works with glasses & narrow/wide eyes)
    if (avgBlinkBlend < 0.28 && avgEar > 0.10 && avgEar < 0.55) {
      if (this.calibratedSamples === 0) {
        this.baselineEar = avgEar;
      } else {
        const blendAlpha = this.calibratedSamples < 12 ? 0.28 : 0.04;
        this.baselineEar = this.baselineEar * (1 - blendAlpha) + avgEar * blendAlpha;
      }
      this.calibratedSamples += 1;
    }

    const computeOpenness = (ear: number, blinkScore: number | null) => {
      const relativeEar = ear / Math.max(this.baselineEar, 0.12);
      const earOpen = clamp((relativeEar - 0.50) / 0.45, 0, 1);
      if (blinkScore === null) return earOpen;
      const blendOpen = clamp(1 - blinkScore, 0, 1);
      return clamp(blendOpen * 0.60 + earOpen * 0.40, 0, 1);
    };

    const rightEyeOpenness = computeOpenness(rightEar, blinkRightScore);
    const leftEyeOpenness = computeOpenness(leftEar, blinkLeftScore);
    const avgOpenness = (rightEyeOpenness + leftEyeOpenness) / 2;
    const relativeAvgEar = avgEar / Math.max(this.baselineEar, 0.12);

    // Responsive blink & closure detection calibrated for 30-60 FPS webcams & glasses
    const closedSignal =
      sensitivity === 'strict'
        ? avgOpenness < 0.55 || avgBlinkBlend >= 0.34 || relativeAvgEar < 0.75
        : avgOpenness < 0.50 || avgBlinkBlend >= 0.38 || relativeAvgEar < 0.71;

    const openSignal =
      sensitivity === 'strict'
        ? avgOpenness >= 0.60 && (avgBlinkBlend < 0.28 || relativeAvgEar >= 0.86)
        : avgOpenness >= 0.56 && (avgBlinkBlend < 0.32 || relativeAvgEar >= 0.84);

    let closedDurationMs = 0;
    if (!this.isCurrentlyClosed && closedSignal) {
      this.isCurrentlyClosed = true;
      this.closedStartTime = now;
    } else if (this.isCurrentlyClosed && openSignal) {
      const duration = this.closedStartTime !== null ? now - this.closedStartTime : 0;
      if (duration >= 15 && duration <= 1100) {
        this.blinkCount += 1;
        this.blinkTimestamps.push(now);
        this.blinkDurationsMs = [...this.blinkDurationsMs, duration].slice(-20);
      }
      this.isCurrentlyClosed = false;
      this.closedStartTime = null;
    }

    if (this.isCurrentlyClosed && this.closedStartTime !== null) {
      closedDurationMs = now - this.closedStartTime;
    }

    // Rolling 60s PERCLOS window (tracks both prolonged closures and blink closure ratios)
    const isPerclosClosed = this.isCurrentlyClosed || closedSignal;
    this.closureSamples.push({ t: now, closed: isPerclosClosed });
    const windowCutoff = now - 60_000;
    while (this.closureSamples.length > 0 && this.closureSamples[0].t < windowCutoff) {
      this.closureSamples.shift();
    }
    while (this.blinkTimestamps.length > 0 && this.blinkTimestamps[0] < windowCutoff) {
      this.blinkTimestamps.shift();
    }

    if (now - this.lastWaveSampleAt >= 100) {
      this.lastWaveSampleAt = now;
      this.opennessHistory = [...this.opennessHistory, avgOpenness].slice(-80);
    }

    // Gaze estimation (ARKit gaze blendshapes + geometric iris displacement)
    const lookInL = getBlendshapeScore(categories, 'eyeLookInLeft') ?? 0;
    const lookOutL = getBlendshapeScore(categories, 'eyeLookOutLeft') ?? 0;
    const lookUpL = getBlendshapeScore(categories, 'eyeLookUpLeft') ?? 0;
    const lookDownL = getBlendshapeScore(categories, 'eyeLookDownLeft') ?? 0;

    const lookInR = getBlendshapeScore(categories, 'eyeLookInRight') ?? 0;
    const lookOutR = getBlendshapeScore(categories, 'eyeLookOutRight') ?? 0;
    const lookUpR = getBlendshapeScore(categories, 'eyeLookUpRight') ?? 0;
    const lookDownR = getBlendshapeScore(categories, 'eyeLookDownRight') ?? 0;

    const subjectRightward = (lookInL + lookOutR - (lookOutL + lookInR)) / 2;
    const blendGazeX = mirrored ? subjectRightward : -subjectRightward;
    const blendGazeY = (lookDownL + lookDownR - (lookUpL + lookUpR)) / 2;

    const geomRight = this.computeGeometricGaze(landmarks, RIGHT_EYE, mirrored);
    const geomLeft = this.computeGeometricGaze(landmarks, LEFT_EYE, mirrored);
    const geomGazeX = (geomRight.gx + geomLeft.gx) / 2;
    const geomGazeY = (geomRight.gy + geomLeft.gy) / 2;

    const targetGazeX = clamp(blendGazeX * 0.7 + geomGazeX * 0.3, -1, 1);
    const targetGazeY = clamp(blendGazeY * 0.72 + geomGazeY * 0.28, -1, 1);

    this.smoothedGazeX = this.smoothedGazeX * 0.52 + targetGazeX * 0.48;
    this.smoothedGazeY = this.smoothedGazeY * 0.52 + targetGazeY * 0.48;

    let gazeDirection: GazeDirection = 'Focused';
    if (this.isCurrentlyClosed && closedDurationMs > 220) {
      gazeDirection = 'Eyes Closed';
    } else if (Math.abs(this.smoothedGazeX) > 0.22 || Math.abs(this.smoothedGazeY) > 0.22) {
      if (Math.abs(this.smoothedGazeX) >= Math.abs(this.smoothedGazeY)) {
        gazeDirection = this.smoothedGazeX < 0 ? 'Looking Left' : 'Looking Right';
      } else {
        gazeDirection = this.smoothedGazeY < 0 ? 'Looking Up' : 'Looking Down';
      }
    }

    if (gazeDirection === 'Focused') {
      this.focusAccumMs += dtMs;
      this.currentFocusStreakMs += dtMs;
      if (this.currentFocusStreakMs > this.bestFocusStreakMs) {
        this.bestFocusStreakMs = this.currentFocusStreakMs;
      }
    } else {
      this.currentFocusStreakMs = 0;
    }

    const sessionSeconds = Math.max(0, (now - this.sessionStartTime) / 1000);
    const focusSeconds = this.focusAccumMs / 1000;
    const focusRatioPercent =
      sessionSeconds > 0.5 ? clamp((focusSeconds / sessionSeconds) * 100, 0, 100) : 100;

    const perclosPercent = this.computePerclos();
    const blinkRateBpm = this.computeBpm(now);
    const avgBlinkDurationMs = this.computeAvgBlinkDuration();

    // Thresholds tuned by SensitivityMode
    const criticalClosureMs = sensitivity === 'strict' ? 850 : 1300;
    const warningClosureMs = sensitivity === 'strict' ? 400 : 550;
    const criticalPerclos = sensitivity === 'strict' ? 15 : 22;
    const warningPerclos = sensitivity === 'strict' ? 8 : 12;

    let fatigueLevel: FatigueLevel = 'nominal';
    let alertReason: string | null = null;

    if (closedDurationMs >= criticalClosureMs) {
      fatigueLevel = 'critical';
      alertReason = `Prolonged eye closure (${(closedDurationMs / 1000).toFixed(1)}s)`;
    } else if (sessionSeconds >= 6 && perclosPercent >= criticalPerclos) {
      fatigueLevel = 'critical';
      alertReason = `Elevated PERCLOS (${perclosPercent.toFixed(1)}%)`;
    } else if (tmDrowsyScore >= 0.84) {
      fatigueLevel = 'critical';
      alertReason = `High fatigue probability (${Math.round(tmDrowsyScore * 100)}%)`;
    } else if (closedDurationMs >= warningClosureMs) {
      fatigueLevel = 'warning';
      alertReason = 'Extended blink detected';
    } else if (sessionSeconds >= 6 && perclosPercent >= warningPerclos) {
      fatigueLevel = 'warning';
      alertReason = `Moderate fatigue drift (PERCLOS ${perclosPercent.toFixed(1)}%)`;
    }

    if (fatigueLevel === 'critical' && !this.wasCriticalFatigue) {
      this.drowsinessEpisodes += 1;
      this.wasCriticalFatigue = true;
    } else if (fatigueLevel !== 'critical') {
      this.wasCriticalFatigue = false;
    }

    // Real-time Alertness Score (0..100)
    const perclosPenalty = clamp(perclosPercent * 1.9, 0, 48);
    const closurePenalty = clamp((closedDurationMs / criticalClosureMs) * 40, 0, 42);
    const gazePenalty = gazeDirection === 'Focused' ? 0 : 8;
    const tmPenalty = tmDrowsyScore > 0.5 ? (tmDrowsyScore - 0.5) * 35 : 0;
    const alertnessScore = Math.round(
      clamp(100 - perclosPenalty - closurePenalty - gazePenalty - tmPenalty, 0, 100)
    );

    // Draw real-time face perimeter brackets + iris rings + eyelid arcs + gaze vector
    if (ctx && showOverlay && video.videoWidth > 0 && video.videoHeight > 0) {
      const transform = getCoverTransform(
        video.videoWidth,
        video.videoHeight,
        canvas.width,
        canvas.height,
        mirrored
      );

      this.drawFaceFrameBrackets(ctx, landmarks, fatigueLevel, transform);

      this.smoothedRightIris = this.drawEyeOverlay({
        ctx,
        landmarks,
        eye: RIGHT_EYE,
        openness: rightEyeOpenness,
        fatigueLevel,
        gazeX: this.smoothedGazeX,
        gazeY: this.smoothedGazeY,
        prevPoint: this.smoothedRightIris,
        transform,
      });

      this.smoothedLeftIris = this.drawEyeOverlay({
        ctx,
        landmarks,
        eye: LEFT_EYE,
        openness: leftEyeOpenness,
        fatigueLevel,
        gazeX: this.smoothedGazeX,
        gazeY: this.smoothedGazeY,
        prevPoint: this.smoothedLeftIris,
        transform,
      });
    }

    const displayLeftOpenness = mirrored ? rightEyeOpenness : leftEyeOpenness;
    const displayRightOpenness = mirrored ? leftEyeOpenness : rightEyeOpenness;

    return {
      faceDetected: true,
      leftOpenness: displayLeftOpenness,
      rightOpenness: displayRightOpenness,
      avgOpenness,
      isBlinking: this.isCurrentlyClosed,
      closedDurationMs,
      blinkCount: this.blinkCount,
      blinkRateBpm,
      avgBlinkDurationMs,
      perclosPercent,
      alertnessScore,
      drowsinessEpisodes: this.drowsinessEpisodes,
      gazeX: this.smoothedGazeX,
      gazeY: this.smoothedGazeY,
      gazeDirection,
      focusSeconds,
      focusStreakSeconds: this.currentFocusStreakMs / 1000,
      bestFocusStreakSeconds: this.bestFocusStreakMs / 1000,
      sessionSeconds,
      focusRatioPercent,
      fatigueLevel,
      alertReason,
      opennessHistory: this.opennessHistory,
      fps: Math.round(this.smoothedFps),
    };
  }

  private computeGeometricGaze(
    landmarks: NormalizedLandmark[],
    eye: typeof RIGHT_EYE,
    mirrored: boolean
  ) {
    const outer = landmarks[eye.outer];
    const inner = landmarks[eye.inner];
    const iris = landmarks[eye.irisCenter];
    if (!outer || !inner || !iris) return { gx: 0, gy: 0 };

    const cx = (outer.x + inner.x) / 2;
    const cy = (outer.y + inner.y) / 2;
    const width = Math.max(dist2D(outer, inner), 1e-4);

    const rawDx = (iris.x - cx) / (width * 0.24);
    const gx = clamp(mirrored ? -rawDx : rawDx, -1, 1);
    const gy = clamp((iris.y - (cy - width * 0.03)) / (width * 0.14), -1, 1);
    return { gx, gy };
  }

  private computeBpm(now: number): number {
    if (this.blinkTimestamps.length === 0) return 0;
    const elapsedSec = Math.max((now - this.sessionStartTime) / 1000, 1);
    const effectiveWindowSec = clamp(elapsedSec, 10, 60);
    return Math.max(1, Math.round((this.blinkTimestamps.length / effectiveWindowSec) * 60));
  }

  private computeAvgBlinkDuration(): number {
    if (this.blinkDurationsMs.length === 0) return 0;
    const sum = this.blinkDurationsMs.reduce((acc, v) => acc + v, 0);
    return Math.round(sum / this.blinkDurationsMs.length);
  }

  private computePerclos(): number {
    if (this.closureSamples.length === 0) return 0;
    let closedCount = 0;
    for (const s of this.closureSamples) {
      if (s.closed) closedCount += 1;
    }
    return (closedCount / this.closureSamples.length) * 100;
  }

  private drawFaceFrameBrackets(
    ctx: CanvasRenderingContext2D,
    landmarks: NormalizedLandmark[],
    fatigueLevel: FatigueLevel,
    transform: ReturnType<typeof getCoverTransform>
  ) {
    const top = landmarks[FACE_BOUNDS.top];
    const bottom = landmarks[FACE_BOUNDS.bottom];
    const left = landmarks[FACE_BOUNDS.left];
    const right = landmarks[FACE_BOUNDS.right];
    if (!top || !bottom || !left || !right) return;

    const xs = [transform.toX(left.x), transform.toX(right.x)];
    const ys = [transform.toY(top.y), transform.toY(bottom.y)];
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const padX = (maxX - minX) * 0.08;
    const padY = (maxY - minY) * 0.08;
    const x1 = minX - padX;
    const x2 = maxX + padX;
    const y1 = minY - padY;
    const y2 = maxY + padY;

    const dprScale = Math.max(1, ctx.canvas.width / Math.max(ctx.canvas.clientWidth || ctx.canvas.width, 1));
    const len = clamp((x2 - x1) * 0.14, 12 * dprScale, 28 * dprScale);

    ctx.save();
    ctx.strokeStyle =
      fatigueLevel === 'critical'
        ? 'rgba(239, 68, 68, 0.85)'
        : fatigueLevel === 'warning'
          ? 'rgba(245, 158, 11, 0.7)'
          : 'rgba(220, 38, 38, 0.45)';
    ctx.lineWidth = 1.5 * dprScale;

    // Top-left
    ctx.beginPath();
    ctx.moveTo(x1, y1 + len);
    ctx.lineTo(x1, y1);
    ctx.lineTo(x1 + len, y1);
    ctx.stroke();

    // Top-right
    ctx.beginPath();
    ctx.moveTo(x2 - len, y1);
    ctx.lineTo(x2, y1);
    ctx.lineTo(x2, y1 + len);
    ctx.stroke();

    // Bottom-left
    ctx.beginPath();
    ctx.moveTo(x1, y2 - len);
    ctx.lineTo(x1, y2);
    ctx.lineTo(x1 + len, y2);
    ctx.stroke();

    // Bottom-right
    ctx.beginPath();
    ctx.moveTo(x2 - len, y2);
    ctx.lineTo(x2, y2);
    ctx.lineTo(x2, y2 - len);
    ctx.stroke();

    ctx.restore();
  }

  private drawEyeOverlay(params: {
    ctx: CanvasRenderingContext2D;
    landmarks: NormalizedLandmark[];
    eye: typeof RIGHT_EYE;
    openness: number;
    fatigueLevel: FatigueLevel;
    gazeX: number;
    gazeY: number;
    prevPoint: SmoothedEyePoint | null;
    transform: ReturnType<typeof getCoverTransform>;
  }): SmoothedEyePoint | null {
    const {
      ctx,
      landmarks,
      eye,
      openness,
      fatigueLevel,
      gazeX,
      gazeY,
      prevPoint,
      transform,
    } = params;

    const irisLm = landmarks[eye.irisCenter];
    const outerLm = landmarks[eye.outer];
    const innerLm = landmarks[eye.inner];
    const upperLm = landmarks[eye.upper];
    const lowerLm = landmarks[eye.lower];

    if (!irisLm || !outerLm || !innerLm || !upperLm || !lowerLm) return prevPoint;

    const rawIx = transform.toX(irisLm.x);
    const rawIy = transform.toY(irisLm.y);

    let radiusSum = 0;
    let radiusCount = 0;
    for (const idx of eye.irisPerimeter) {
      const p = landmarks[idx];
      if (p) {
        radiusSum += Math.hypot(transform.toX(p.x) - rawIx, transform.toY(p.y) - rawIy);
        radiusCount += 1;
      }
    }
    const rawR = clamp(radiusCount > 0 ? radiusSum / radiusCount : 14, 10, 44);

    const alpha = 0.62;
    const ix = prevPoint ? prevPoint.x * (1 - alpha) + rawIx * alpha : rawIx;
    const iy = prevPoint ? prevPoint.y * (1 - alpha) + rawIy * alpha : rawIy;
    const r = prevPoint ? prevPoint.r * (1 - alpha) + rawR * alpha : rawR;

    const isClosed = openness < 0.48;
    const primaryColor =
      fatigueLevel === 'critical' || isClosed
        ? '#EF4444'
        : fatigueLevel === 'warning' || openness < 0.56
          ? '#F59E0B'
          : '#F87171';

    const outerPt = { x: transform.toX(outerLm.x), y: transform.toY(outerLm.y) };
    const innerPt = { x: transform.toX(innerLm.x), y: transform.toY(innerLm.y) };
    const upperPt = { x: transform.toX(upperLm.x), y: transform.toY(upperLm.y) };
    const lowerPt = { x: transform.toX(lowerLm.x), y: transform.toY(lowerLm.y) };

    const dprScale = Math.max(
      1,
      ctx.canvas.width / Math.max(ctx.canvas.clientWidth || ctx.canvas.width, 1)
    );

    ctx.save();

    // 1. Subtle eyelid contour arcs reactive to blink state
    ctx.strokeStyle = isClosed ? 'rgba(239, 68, 68, 0.92)' : 'rgba(255, 255, 255, 0.45)';
    ctx.lineWidth = (isClosed ? 2.2 : 1.35) * dprScale;
    ctx.lineCap = 'round';

    const upperCtrlX = upperPt.x;
    const upperCtrlY = 2 * upperPt.y - 0.5 * (outerPt.y + innerPt.y);
    ctx.beginPath();
    ctx.moveTo(outerPt.x, outerPt.y);
    ctx.quadraticCurveTo(upperCtrlX, upperCtrlY, innerPt.x, innerPt.y);
    ctx.stroke();

    const lowerCtrlX = lowerPt.x;
    const lowerCtrlY = 2 * lowerPt.y - 0.5 * (outerPt.y + innerPt.y);
    ctx.beginPath();
    ctx.moveTo(outerPt.x, outerPt.y);
    ctx.quadraticCurveTo(lowerCtrlX, lowerCtrlY, innerPt.x, innerPt.y);
    ctx.stroke();

    // 2. Iris lock ring & precision ticks
    ctx.shadowColor = '#DC2626';
    ctx.shadowBlur = 10 * dprScale;
    ctx.strokeStyle = primaryColor;
    ctx.lineWidth = 1.9 * dprScale;

    ctx.beginPath();
    ctx.arc(ix, iy, r, 0, Math.PI * 2);
    ctx.stroke();

    ctx.shadowBlur = 0;
    ctx.strokeStyle = isClosed ? 'rgba(239, 68, 68, 0.85)' : 'rgba(255, 255, 255, 0.75)';
    ctx.lineWidth = 1.2 * dprScale;
    const tickInner = r + 2 * dprScale;
    const tickOuter = r + 5.5 * dprScale;
    for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 2) {
      ctx.beginPath();
      ctx.moveTo(ix + Math.cos(angle) * tickInner, iy + Math.sin(angle) * tickInner);
      ctx.lineTo(ix + Math.cos(angle) * tickOuter, iy + Math.sin(angle) * tickOuter);
      ctx.stroke();
    }

    // 3. Pupil center & live gaze vector
    const gazeMag = Math.hypot(gazeX, gazeY);
    ctx.fillStyle = '#FFFFFF';
    ctx.beginPath();
    ctx.arc(ix, iy, 2.4 * dprScale, 0, Math.PI * 2);
    ctx.fill();

    if (!isClosed && gazeMag > 0.14) {
      const vecLen = clamp(gazeMag, 0.14, 1) * r * 2.1;
      const angle = Math.atan2(gazeY, gazeX);
      const ex = ix + Math.cos(angle) * vecLen;
      const ey = iy + Math.sin(angle) * vecLen;

      ctx.strokeStyle = primaryColor;
      ctx.lineWidth = 1.75 * dprScale;
      ctx.beginPath();
      ctx.moveTo(ix, iy);
      ctx.lineTo(ex, ey);
      ctx.stroke();

      ctx.fillStyle = '#FFFFFF';
      ctx.beginPath();
      ctx.arc(ex, ey, 2.5 * dprScale, 0, Math.PI * 2);
      ctx.fill();
    } else if (!isClosed) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
      ctx.lineWidth = 1.1 * dprScale;
      ctx.beginPath();
      ctx.arc(ix, iy, r * 0.42, 0, Math.PI * 2);
      ctx.stroke();
    }

    ctx.restore();
    return { x: ix, y: iy, r };
  }
}
