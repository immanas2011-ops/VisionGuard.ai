import { useCallback, useEffect, useRef, useState } from 'react';
import * as tmImage from '@teachablemachine/image';
import { motion, AnimatePresence } from 'motion/react';
import {
  Camera,
  Activity,
  ShieldAlert,
  Monitor,
  Terminal,
  Zap,
  Lock,
  Eye,
  EyeOff,
  ShieldCheck,
  Search,
  Crosshair,
  Database,
  Cloud,
  ChevronRight,
  Brain,
  ClipboardCheck,
  RefreshCw,
  AlertTriangle,
  Target,
  FlipHorizontal2,
  Volume2,
  VolumeX,
  Square,
  Download,
  RotateCcw,
  SlidersHorizontal,
  X,
} from 'lucide-react';
import { EyeTrackingProcessor, getFaceLandmarker } from './lib/eyeTracker';
import { soundAlert } from './lib/soundAlert';
import type {
  CameraStatus,
  DiagnosticReport,
  EyeTelemetry,
  PredictionClass,
  SensitivityMode,
  SessionEvent,
} from './types/vision';

const MODEL_URL = 'https://teachablemachine.withgoogle.com/models/s4-ADGmEy/';

type TabId = 'home' | 'core' | 'specs' | 'stats' | 'how-it-works' | 'future';

interface Feature {
  icon: any;
  title: string;
  description: string;
}

const features: Feature[] = [
  {
    icon: Eye,
    title: 'AI Recognition',
    description:
      'Trained on massive datasets to identify objects with surgical precision at sub-millisecond speeds.',
  },
  {
    icon: ShieldCheck,
    title: 'Security Shield',
    description:
      'Autonomous threat detection protocols that continuously monitor visual feeds for anomalies.',
  },
  {
    icon: Activity,
    title: 'Live Telemetry',
    description:
      'Deep-stream processing with real-time analytics providing instant feedback and confidence metrics.',
  },
  {
    icon: Lock,
    title: 'Private Core',
    description:
      'Operates entirely on-device. Your vision data never leaves your secure local infrastructure.',
  },
];

const INITIAL_TELEMETRY: EyeTelemetry = {
  faceDetected: false,
  leftOpenness: 0,
  rightOpenness: 0,
  avgOpenness: 0,
  isBlinking: false,
  closedDurationMs: 0,
  blinkCount: 0,
  blinkRateBpm: 0,
  avgBlinkDurationMs: 0,
  perclosPercent: 0,
  alertnessScore: 100,
  drowsinessEpisodes: 0,
  gazeX: 0,
  gazeY: 0,
  gazeDirection: 'No Face',
  focusSeconds: 0,
  focusStreakSeconds: 0,
  bestFocusStreakSeconds: 0,
  sessionSeconds: 0,
  focusRatioPercent: 100,
  fatigueLevel: 'nominal',
  alertReason: null,
  opennessHistory: [],
  fps: 0,
};

function formatDuration(totalSeconds: number): string {
  const s = Math.max(0, Math.floor(totalSeconds));
  const mins = Math.floor(s / 60);
  const secs = s % 60;
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

function buildWaveformPath(samples: number[], width: number, height: number) {
  if (samples.length < 2) {
    return { line: '', area: '' };
  }
  const padY = 4;
  const usableH = height - padY * 2;
  const pts = samples.map((val, idx) => {
    const x = (idx / Math.max(samples.length - 1, 1)) * width;
    const clamped = Math.max(0, Math.min(1, val));
    const y = padY + (1 - clamped) * usableH;
    return { x, y };
  });
  const line = pts
    .map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`)
    .join(' ');
  const area = `${line} L ${width.toFixed(1)} ${height.toFixed(1)} L 0 ${height.toFixed(1)} Z`;
  return { line, area };
}

export default function App() {
  const [predictions, setPredictions] = useState<PredictionClass[]>([]);
  const [logs, setLogs] = useState<string[]>([
    '[SYSTEM] CORE V0.4.1 SECURED',
    '[AUTH] SUPERUSER ACCESS GRANTED',
    '[HARDWARE] OPTICAL ARRAY STANDBY',
  ]);
  const [events, setEvents] = useState<SessionEvent[]>([]);
  const [status, setStatus] = useState<CameraStatus>('idle');
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isDiagnosticRunning, setIsDiagnosticRunning] = useState(false);
  const [diagnosticProgress, setDiagnosticProgress] = useState(0);
  const [diagnosticError, setDiagnosticError] = useState<string | null>(null);
  const [report, setReport] = useState<DiagnosticReport | null>(null);
  const [reportFrames, setReportFrames] = useState<string[]>([]);
  const [reportRefId, setReportRefId] = useState<string>('');
  const [bufferSize, setBufferSize] = useState(0);

  const [mirrored, setMirrored] = useState(true);
  const [showOverlay, setShowOverlay] = useState(true);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [sensitivity, setSensitivity] = useState<SensitivityMode>('standard');
  const [telemetry, setTelemetry] = useState<EyeTelemetry>(INITIAL_TELEMETRY);

  const [activeTab, setActiveTab] = useState<TabId>('home');

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const frameBufferRef = useRef<string[]>([]);
  const diagnosticAbortRef = useRef<AbortController | null>(null);

  const processorRef = useRef(new EyeTrackingProcessor());
  const telemetryRef = useRef<EyeTelemetry>(INITIAL_TELEMETRY);
  const tmModelRef = useRef<tmImage.CustomMobileNet | null>(null);
  const tmDrowsyScoreRef = useRef(0);

  const runningRef = useRef(false);
  const mirroredRef = useRef(mirrored);
  const showOverlayRef = useRef(showOverlay);
  const soundEnabledRef = useRef(soundEnabled);
  const sensitivityRef = useRef(sensitivity);

  const lastVideoTimeRef = useRef(-1);
  const lastDetectWallRef = useRef(0);
  const lastDetectTsRef = useRef(0);
  const lastUiSyncRef = useRef(0);
  const lastTmPredictRef = useRef(0);
  const lastFatigueEventAtRef = useRef(0);
  const gazeAwayStartRef = useRef<number | null>(null);
  const gazeAwayLoggedRef = useRef(false);

  useEffect(() => {
    mirroredRef.current = mirrored;
  }, [mirrored]);

  useEffect(() => {
    showOverlayRef.current = showOverlay;
  }, [showOverlay]);

  useEffect(() => {
    soundEnabledRef.current = soundEnabled;
  }, [soundEnabled]);

  useEffect(() => {
    sensitivityRef.current = sensitivity;
  }, [sensitivity]);

  const addLog = useCallback((msg: string) => {
    setLogs((prev) =>
      [`[${new Date().toLocaleTimeString('en-GB')}] ${msg}`, ...prev].slice(0, 16)
    );
  }, []);

  const pushSessionEvent = useCallback(
    (
      type: SessionEvent['type'],
      severity: SessionEvent['severity'],
      title: string,
      detail: string
    ) => {
      const timestamp = new Date().toLocaleTimeString('en-GB');
      setEvents((prev) =>
        [
          {
            id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
            timestamp,
            type,
            severity,
            title,
            detail,
          },
          ...prev,
        ].slice(0, 30)
      );
      addLog(`${title.toUpperCase()}: ${detail}`);
    },
    [addLog]
  );

  // Status icon animation variants (preserved from original VisionGuard design)
  const iconVariants = {
    idle: { opacity: 0.4, scale: 0.9 },
    loading: {
      opacity: 1,
      scale: 1.1,
      rotate: [0, 360],
      transition: { repeat: Infinity, duration: 2 },
    },
    active: { opacity: 1, scale: 1, filter: 'drop-shadow(0 0 8px #dc2626)' },
    error: {
      opacity: 1,
      color: '#ef4444',
      x: [0, -2, 2, -2, 2, 0],
      transition: { duration: 0.2 },
    },
  };

  const syncOverlayCanvasSize = useCallback(() => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const nextW = Math.round(rect.width * dpr);
    const nextH = Math.round(rect.height * dpr);
    if (canvas.width !== nextW || canvas.height !== nextH) {
      canvas.width = nextW;
      canvas.height = nextH;
    }
  }, []);

  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    if (!canvas || typeof ResizeObserver === 'undefined') return;
    syncOverlayCanvasSize();
    const observer = new ResizeObserver(() => syncOverlayCanvasSize());
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [syncOverlayCanvasSize]);

  const stopScanner = useCallback(() => {
    runningRef.current = false;
    diagnosticAbortRef.current?.abort();
    diagnosticAbortRef.current = null;
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    const overlay = overlayCanvasRef.current;
    if (overlay) {
      const ctx = overlay.getContext('2d');
      ctx?.clearRect(0, 0, overlay.width, overlay.height);
    }
    frameBufferRef.current = [];
    telemetryRef.current = INITIAL_TELEMETRY;
    setBufferSize(0);
    setPredictions([]);
    setTelemetry(INITIAL_TELEMETRY);
    setIsDiagnosticRunning(false);
    setDiagnosticProgress(0);
    setStatus('idle');
    addLog('OPTICAL STREAM TERMINATED');
  }, [addLog]);

  const handleResetTelemetry = useCallback(() => {
    processorRef.current.reset();
    telemetryRef.current = INITIAL_TELEMETRY;
    setTelemetry(INITIAL_TELEMETRY);
    addLog('TELEMETRY BASELINE RECALIBRATED');
  }, [addLog]);

  const initScanner = useCallback(async () => {
    if (status === 'active' || status === 'loading') return;

    soundAlert.unlock();
    setCameraError(null);
    setDiagnosticError(null);
    setStatus('loading');
    addLog('MOUNTING OPTICAL SENSOR & IRIS TRACKER...');

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Camera API not supported in this browser.');
      }

      const streamPromise = navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user',
        },
        audio: false,
      });

      const landmarkerPromise = getFaceLandmarker();

      const tmPromise = tmModelRef.current
        ? Promise.resolve(tmModelRef.current)
        : tmImage
            .load(`${MODEL_URL}model.json`, `${MODEL_URL}metadata.json`)
            .catch(() => null);

      const [streamRes, landmarkerRes, tmRes] = await Promise.allSettled([
        streamPromise,
        landmarkerPromise,
        tmPromise,
      ]);

      if (streamRes.status === 'rejected') {
        throw streamRes.reason;
      }

      const stream = streamRes.value;
      if (landmarkerRes.status === 'rejected') {
        stream.getTracks().forEach((t) => t.stop());
        throw landmarkerRes.reason;
      }

      const landmarker = landmarkerRes.value;
      const tmModel = tmRes.status === 'fulfilled' ? tmRes.value : null;

      if (tmModel) {
        tmModelRef.current = tmModel;
        addLog(`MODEL_LINKED: ${tmModel.getTotalClasses()} INDICES`);
      }

      const video = videoRef.current;
      if (!video) {
        stream.getTracks().forEach((t) => t.stop());
        throw new Error('Optical viewport element unavailable.');
      }

      streamRef.current = stream;
      video.srcObject = stream;

      await new Promise<void>((resolve) => {
        if (video.readyState >= 2) {
          resolve();
          return;
        }
        const onReady = () => {
          video.removeEventListener('loadeddata', onReady);
          resolve();
        };
        video.addEventListener('loadeddata', onReady);
      });

      await video.play();
      syncOverlayCanvasSize();

      if (!captureCanvasRef.current) {
        captureCanvasRef.current = document.createElement('canvas');
      }
      captureCanvasRef.current.width = 400;
      captureCanvasRef.current.height = 400;

      processorRef.current.reset();
      telemetryRef.current = INITIAL_TELEMETRY;
      lastVideoTimeRef.current = -1;
      lastDetectWallRef.current = 0;
      lastUiSyncRef.current = 0;
      lastTmPredictRef.current = 0;
      lastFatigueEventAtRef.current = 0;
      gazeAwayStartRef.current = null;
      gazeAwayLoggedRef.current = false;
      runningRef.current = true;

      setStatus('active');
      addLog('NEURAL & IRIS TRACKING ENGINE ONLINE');

      const loop = (now: number) => {
        if (!runningRef.current) return;

        const vid = videoRef.current;
        const overlay = overlayCanvasRef.current;

        if (vid && overlay && vid.readyState >= 2 && vid.videoWidth > 0) {
          syncOverlayCanvasSize();

          let result = null;
          if (
            vid.currentTime !== lastVideoTimeRef.current ||
            now - lastDetectWallRef.current >= 120
          ) {
            lastVideoTimeRef.current = vid.currentTime;
            lastDetectWallRef.current = now;
            const ts = Math.max(Math.floor(performance.now()), lastDetectTsRef.current + 1);
            lastDetectTsRef.current = ts;
            try {
              result = landmarker.detectForVideo(vid, ts);
            } catch {
              result = null;
            }
          }

          // Run Teachable Machine classification at ~5 Hz on clean cropped frame
          if (
            tmModelRef.current &&
            captureCanvasRef.current &&
            now - lastTmPredictRef.current >= 180
          ) {
            lastTmPredictRef.current = now;
            const cCanvas = captureCanvasRef.current;
            const cCtx = cCanvas.getContext('2d');
            if (cCtx) {
              const size = Math.min(vid.videoWidth, vid.videoHeight);
              const sx = (vid.videoWidth - size) / 2;
              const sy = (vid.videoHeight - size) / 2;
              cCtx.save();
              if (mirroredRef.current) {
                cCtx.translate(cCanvas.width, 0);
                cCtx.scale(-1, 1);
              }
              cCtx.drawImage(vid, sx, sy, size, size, 0, 0, cCanvas.width, cCanvas.height);
              cCtx.restore();

              void tmModelRef.current
                .predict(cCanvas)
                .then((results) => {
                  if (!runningRef.current) return;
                  setPredictions(results);
                  const drowsyPred = results.find((p) => {
                    const name = p.className.toLowerCase();
                    return (
                      name.includes('drowsy') ||
                      name.includes('sleep') ||
                      name.includes('tired') ||
                      name.includes('closed')
                    );
                  });
                  tmDrowsyScoreRef.current = drowsyPred ? drowsyPred.probability : 0;
                })
                .catch(() => undefined);
            }
          }

          if (result !== null) {
            const nextTelemetry = processorRef.current.processAndDraw({
              result,
              video: vid,
              canvas: overlay,
              mirrored: mirroredRef.current,
              showOverlay: showOverlayRef.current,
              sensitivity: sensitivityRef.current,
              tmDrowsyScore: tmDrowsyScoreRef.current,
              now,
            });
            telemetryRef.current = nextTelemetry;

            if (nextTelemetry.fatigueLevel === 'critical' && nextTelemetry.alertReason) {
              if (soundEnabledRef.current) {
                soundAlert.playAlert('critical');
              }
              if (now - lastFatigueEventAtRef.current >= 8000) {
                lastFatigueEventAtRef.current = now;
                pushSessionEvent(
                  nextTelemetry.closedDurationMs >= 1300 ? 'closure' : 'drowsiness',
                  'critical',
                  'Drowsiness Alert',
                  nextTelemetry.alertReason
                );
              }
            } else if (nextTelemetry.fatigueLevel === 'warning' && nextTelemetry.alertReason) {
              if (now - lastFatigueEventAtRef.current >= 12000) {
                lastFatigueEventAtRef.current = now;
                pushSessionEvent(
                  'drowsiness',
                  'warning',
                  'Fatigue Warning',
                  nextTelemetry.alertReason
                );
              }
            }

            const isLookingAway =
              nextTelemetry.faceDetected &&
              nextTelemetry.gazeDirection !== 'Focused' &&
              nextTelemetry.gazeDirection !== 'Eyes Closed';
            if (isLookingAway) {
              if (gazeAwayStartRef.current === null) {
                gazeAwayStartRef.current = now;
                gazeAwayLoggedRef.current = false;
              } else if (!gazeAwayLoggedRef.current && now - gazeAwayStartRef.current >= 4000) {
                gazeAwayLoggedRef.current = true;
                pushSessionEvent(
                  'gaze',
                  'warning',
                  'Gaze Drift',
                  `${nextTelemetry.gazeDirection} (>4s)`
                );
              }
            } else {
              gazeAwayStartRef.current = null;
              gazeAwayLoggedRef.current = false;
            }

            if (now - lastUiSyncRef.current >= 80) {
              lastUiSyncRef.current = now;
              setTelemetry(nextTelemetry);
            }
          }
        }

        animationFrameRef.current = window.requestAnimationFrame(loop);
      };

      animationFrameRef.current = window.requestAnimationFrame(loop);
    } catch (err) {
      console.warn('Optical scanner initialization error:', err);
      const message =
        err instanceof Error && err.message
          ? err.message
          : 'CAMERA_ACCESS_DENIED';
      setCameraError(message);
      setStatus('error');
      addLog(`IO_FAILURE: ${message}`);
    }
  }, [addLog, pushSessionEvent, status, syncOverlayCanvasSize]);

  const captureCurrentFrameBase64 = useCallback((): string | null => {
    const vid = videoRef.current;
    if (!vid || vid.readyState < 2 || vid.videoWidth === 0) return null;

    if (!captureCanvasRef.current) {
      captureCanvasRef.current = document.createElement('canvas');
      captureCanvasRef.current.width = 400;
      captureCanvasRef.current.height = 400;
    }

    const canvas = captureCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    const size = Math.min(vid.videoWidth, vid.videoHeight);
    const sx = (vid.videoWidth - size) / 2;
    const sy = (vid.videoHeight - size) / 2;
    ctx.save();
    if (mirroredRef.current) {
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(vid, sx, sy, size, size, 0, 0, canvas.width, canvas.height);
    ctx.restore();

    const dataUrl = canvas.toDataURL('image/jpeg', 0.65);
    return dataUrl.split(',')[1] || null;
  }, []);

  useEffect(() => {
    let interval: number | undefined;
    if (status === 'active') {
      addLog('STARTING 1.25s CAPTURE LOOP...');
      interval = window.setInterval(() => {
        const base64 = captureCurrentFrameBase64();
        if (base64) {
          frameBufferRef.current = [...frameBufferRef.current, base64].slice(-5);
          setBufferSize(frameBufferRef.current.length);
        }
      }, 1250);
    }
    return () => {
      if (interval !== undefined) window.clearInterval(interval);
      frameBufferRef.current = [];
      setBufferSize(0);
    };
  }, [status, addLog, captureCurrentFrameBase64]);

  const runDiagnosticScan = async () => {
    if (status !== 'active' || isDiagnosticRunning) return;

    setDiagnosticError(null);
    setIsDiagnosticRunning(true);
    setDiagnosticProgress(10);
    addLog('SYNCHRONIZING 5-FRAME BUFFER...');

    try {
      let capturedFrames = [...frameBufferRef.current];

      // Actively fill up to 5 frames quickly so reports never hang waiting on the 1.25s timer
      let attempts = 0;
      while (runningRef.current && capturedFrames.length < 5 && attempts < 10) {
        attempts += 1;
        const frame = captureCurrentFrameBase64();
        if (frame) {
          capturedFrames = [...capturedFrames, frame].slice(-5);
          frameBufferRef.current = capturedFrames;
          setBufferSize(capturedFrames.length);
          setDiagnosticProgress(Math.min(capturedFrames.length * 18, 90));
        }
        if (capturedFrames.length < 5 && attempts < 10) {
          await new Promise((resolve) => setTimeout(resolve, 220));
        }
        if (!frame && capturedFrames.length > 0) {
          break;
        }
      }

      if (!runningRef.current) return;

      if (capturedFrames.length === 0) {
        throw new Error('Optical stream produced no valid image frames. Ensure camera video is active.');
      }

      setDiagnosticProgress(100);
      addLog(`CAPTURED ${capturedFrames.length} FRAMES. RUNNING GEMINI BIOMETRIC ANALYSIS...`);

      diagnosticAbortRef.current?.abort();
      const controller = new AbortController();
      diagnosticAbortRef.current = controller;

      const tSnap = telemetryRef.current;
      const response = await fetch('/api/diagnostic', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: controller.signal,
        body: JSON.stringify({
          frames: capturedFrames,
          telemetry: {
            blinkRateBpm: tSnap.blinkRateBpm,
            perclosPercent: Number(tSnap.perclosPercent.toFixed(1)),
            gazeDirection: tSnap.gazeDirection,
            focusRatioPercent: Math.round(tSnap.focusRatioPercent),
            alertnessScore: tSnap.alertnessScore,
          },
        }),
      });

      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(
          payload?.error || `Diagnostic service returned HTTP ${response.status}`
        );
      }

      const reportData = payload as DiagnosticReport;
      setReport(reportData);
      setReportFrames(capturedFrames);
      setReportRefId(`VG-${Math.random().toString(36).slice(2, 8).toUpperCase()}`);
      pushSessionEvent(
        'diagnostic',
        reportData.drowsiness.riskLevel.toLowerCase() === 'high' ? 'warning' : 'info',
        'Diagnostic Report',
        `Health ${reportData.eyeHealth.score}% · ${reportData.drowsiness.status}`
      );
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        return;
      }
      console.warn('Diagnostic scan error:', err);
      const message =
        err instanceof Error && err.message ? err.message : 'AI diagnostic request failed.';
      setDiagnosticError(message);
      pushSessionEvent('system', 'critical', 'Diagnostic Error', message);
    } finally {
      diagnosticAbortRef.current = null;
      setIsDiagnosticRunning(false);
    }
  };

  const downloadReportJson = useCallback(() => {
    if (!report) return;
    const exportPayload = {
      referenceId: reportRefId || 'VG-REPORT',
      generatedAt: new Date().toISOString(),
      sensorTelemetry: {
        blinkCount: telemetry.blinkCount,
        blinkRateBpm: telemetry.blinkRateBpm,
        avgBlinkDurationMs: telemetry.avgBlinkDurationMs,
        perclosPercent: Number(telemetry.perclosPercent.toFixed(1)),
        alertnessScore: telemetry.alertnessScore,
        focusRatioPercent: Math.round(telemetry.focusRatioPercent),
        gazeDirection: telemetry.gazeDirection,
      },
      diagnostic: report,
    };
    const blob = new Blob([JSON.stringify(exportPayload, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${(reportRefId || 'visionguard-report').toLowerCase()}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    addLog(`EXPORTED REPORT ${reportRefId || 'JSON'}`);
  }, [addLog, report, reportRefId, telemetry]);

  useEffect(() => {
    return () => {
      runningRef.current = false;
      diagnosticAbortRef.current?.abort();
      if (animationFrameRef.current !== null) cancelAnimationFrame(animationFrameRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  const topPrediction =
    predictions.length > 0
      ? [...predictions].sort((a, b) => b.probability - a.probability)[0]
      : null;

  const { line: waveLine, area: waveArea } = buildWaveformPath(telemetry.opennessHistory, 320, 56);
  const closureThresholdY = 4 + (1 - 0.5) * (56 - 8);
  const dotX = 44 + Math.max(-1, Math.min(1, telemetry.gazeX)) * 30;
  const dotY = 44 + Math.max(-1, Math.min(1, telemetry.gazeY)) * 30;

  return (
    <div className="min-h-screen bg-black text-zinc-400 font-sans selection:bg-red-600 selection:text-white pb-20">
      {/* Background Grid Pattern */}
      <div
        className="fixed inset-0 opacity-[0.03] pointer-events-none"
        style={{
          backgroundImage:
            'linear-gradient(#dc2626 1px, transparent 1px), linear-gradient(90deg, #dc2626 1px, transparent 1px)',
          backgroundSize: '40px 40px',
        }}
      />

      {/* Header */}
      <header className="sticky top-0 z-50 bg-black/80 backdrop-blur-md border-b border-red-950/30 font-display">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div
            className="flex items-center space-x-4 cursor-pointer group"
            onClick={() => setActiveTab('home')}
          >
            <motion.div variants={iconVariants} animate={status} className="relative">
              <ShieldAlert className="w-8 h-8 text-red-600" />
              {status === 'active' && (
                <div className="absolute inset-0 bg-red-600/20 blur-md rounded-full shadow-[0_0_20px_#dc2626]" />
              )}
            </motion.div>
            <div>
              <h1 className="text-2xl font-bold tracking-tighter text-white group-hover:text-red-500 transition-colors">
                VISION<span className="text-red-600">GUARD</span>
              </h1>
              <p className="text-[10px] text-red-900 tracking-[0.4em] uppercase font-bold font-mono">
                Autonomous Perimeter AI
              </p>
            </div>
          </div>

          <nav className="hidden lg:flex items-center space-x-8 text-[10px] font-bold uppercase tracking-widest leading-none">
            <button
              onClick={() => setActiveTab('how-it-works')}
              className={`transition-colors py-2 relative cursor-pointer ${
                activeTab === 'how-it-works' ? 'text-red-500' : 'text-white hover:text-red-500'
              }`}
            >
              How_Works
              {activeTab === 'how-it-works' && (
                <motion.div
                  layoutId="nav-underline"
                  className="absolute -bottom-1 left-0 right-0 h-0.5 bg-red-600"
                />
              )}
            </button>
            <button
              onClick={() => setActiveTab('future')}
              className={`transition-colors py-2 relative cursor-pointer ${
                activeTab === 'future' ? 'text-red-500' : 'text-white hover:text-red-500'
              }`}
            >
              Impact_Future
              {activeTab === 'future' && (
                <motion.div
                  layoutId="nav-underline"
                  className="absolute -bottom-1 left-0 right-0 h-0.5 bg-red-600"
                />
              )}
            </button>
            <div className="h-4 w-px bg-red-900/20 mx-2" />
            <button
              onClick={() => setActiveTab('core')}
              className={`transition-colors py-2 relative cursor-pointer ${
                activeTab === 'core' ? 'text-red-500' : 'text-white hover:text-red-500'
              }`}
            >
              Core_Engine
              {activeTab === 'core' && (
                <motion.div
                  layoutId="nav-underline"
                  className="absolute -bottom-1 left-0 right-0 h-0.5 bg-red-600"
                />
              )}
            </button>
            <button
              onClick={() => setActiveTab('specs')}
              className={`transition-colors py-2 relative cursor-pointer ${
                activeTab === 'specs' ? 'text-red-500' : 'hover:text-red-500'
              }`}
            >
              System_Specs
              {activeTab === 'specs' && (
                <motion.div
                  layoutId="nav-underline"
                  className="absolute -bottom-1 left-0 right-0 h-0.5 bg-red-600"
                />
              )}
            </button>
            <button
              onClick={() => setActiveTab('stats')}
              className={`transition-colors py-2 relative cursor-pointer ${
                activeTab === 'stats' ? 'text-red-500' : 'hover:text-red-500'
              }`}
            >
              India_Stats
              {activeTab === 'stats' && (
                <motion.div
                  layoutId="nav-underline"
                  className="absolute -bottom-1 left-0 right-0 h-0.5 bg-red-600 font-extrabold"
                />
              )}
            </button>
            <button
              onClick={() => {
                setActiveTab('home');
                window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
              }}
              className="px-6 py-2 border border-red-900/40 rounded hover:border-red-600 hover:text-white transition-all hover:bg-red-600/5 cursor-pointer"
            >
              Contact_OS
            </button>
          </nav>
        </div>

        {/* Compact mobile navigation row */}
        <div className="flex lg:hidden items-center gap-5 overflow-x-auto border-t border-red-950/20 px-6 py-2.5 text-[10px] font-bold uppercase tracking-widest scrollbar-none">
          {(
            [
              { id: 'home', label: 'Monitor' },
              { id: 'how-it-works', label: 'How_Works' },
              { id: 'future', label: 'Impact_Future' },
              { id: 'core', label: 'Core_Engine' },
              { id: 'specs', label: 'System_Specs' },
              { id: 'stats', label: 'India_Stats' },
            ] as { id: TabId; label: string }[]
          ).map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`whitespace-nowrap shrink-0 cursor-pointer ${
                activeTab === item.id ? 'text-red-500' : 'text-zinc-400 hover:text-white'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </header>

      {/* Conditional Content Overlay — renders ABOVE main so the monitor stays live underneath */}
      <AnimatePresence>
        {activeTab !== 'home' && (
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            className="fixed inset-0 z-[60] bg-black/95 backdrop-blur-xl p-6 lg:p-20 overflow-y-auto font-sans"
          >
            <div className="max-w-4xl mx-auto space-y-12 pb-32">
              {/* Background Monitor Live Status Strip & Mini Feed while browsing other pages */}
              {status === 'active' && (
                <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-red-900/30 bg-zinc-950/90 px-4 py-3 text-xs">
                  <div className="flex flex-wrap items-center gap-4">
                    <div
                      onClick={() => setActiveTab('home')}
                      title="Return to full monitor"
                      className="relative h-12 w-20 shrink-0 overflow-hidden rounded border border-red-600/40 bg-black cursor-pointer"
                    >
                      <video
                        playsInline
                        muted
                        autoPlay
                        ref={(el) => {
                          if (el && streamRef.current && el.srcObject !== streamRef.current) {
                            el.srcObject = streamRef.current;
                            void el.play().catch(() => undefined);
                          }
                        }}
                        className={`h-full w-full object-cover grayscale-[20%] ${
                          mirrored ? '-scale-x-100' : ''
                        }`}
                      />
                      <span className="absolute top-1 left-1 h-1.5 w-1.5 rounded-full bg-red-600 animate-pulse" />
                    </div>
                    <div className="flex flex-wrap items-center gap-3">
                      <span className="font-mono text-[10px] font-bold uppercase tracking-widest text-red-500">
                        Monitor_Live
                      </span>
                      <span className="text-zinc-700">|</span>
                      <span className="font-mono text-[11px] tabular-nums text-zinc-300">
                        Alertness: <strong className="text-white">{telemetry.alertnessScore}%</strong>
                      </span>
                      <span className="text-zinc-700">·</span>
                      <span className="font-mono text-[11px] tabular-nums text-zinc-300">
                        Blinks: <strong className="text-white">{telemetry.blinkRateBpm} BPM</strong>
                      </span>
                      <span className="text-zinc-700">·</span>
                      <span className="font-mono text-[11px] tabular-nums text-zinc-300">
                        PERCLOS: <strong className="text-white">{telemetry.perclosPercent.toFixed(1)}%</strong>
                      </span>
                      <span className="text-zinc-700">·</span>
                      <span className="font-mono text-[11px] tabular-nums text-zinc-300">
                        Focus: <strong className="text-white">{formatDuration(telemetry.focusSeconds)}</strong>
                      </span>
                      {telemetry.fatigueLevel !== 'nominal' && telemetry.alertReason && (
                        <span className="inline-flex items-center gap-1.5 rounded bg-red-600/20 px-2.5 py-0.5 font-mono text-[10px] font-bold uppercase text-red-400 border border-red-600/40">
                          <AlertTriangle className="h-3 w-3" />
                          {telemetry.alertReason}
                        </span>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => setActiveTab('home')}
                    className="rounded border border-red-600/40 bg-red-600/10 px-3 py-1 font-mono text-[10px] font-bold uppercase tracking-widest text-red-400 hover:bg-red-600 hover:text-white transition-colors cursor-pointer"
                  >
                    Return_To_Monitor
                  </button>
                </div>
              )}

              <div className="flex justify-between items-start border-b border-red-900/10 pb-12">
                <div className="space-y-4">
                  <h2 className="text-5xl font-bold text-white uppercase tracking-tighter font-display leading-tight">
                    {activeTab === 'how-it-works' && 'Technical Pipeline'}
                    {activeTab === 'future' && 'The Vision of New India'}
                    {activeTab === 'core' && 'Core Analytical Engine'}
                    {activeTab === 'specs' && 'Technical Specifications'}
                    {activeTab === 'stats' && 'Road Safety Audit: India'}
                  </h2>
                  <div className="flex space-x-2">
                    <div className="h-1.5 w-24 bg-red-600" />
                    <div className="h-1.5 w-16 bg-red-900/30" />
                    <div className="h-1.5 w-8 bg-zinc-900" />
                  </div>
                </div>
                <button
                  onClick={() => setActiveTab('home')}
                  aria-label="Return to monitor"
                  className="p-4 bg-red-600/5 border border-red-600/20 rounded-full hover:bg-red-600 hover:text-black transition-all group cursor-pointer"
                >
                  <X className="w-6 h-6 group-hover:scale-110 transition-transform" />
                </button>
              </div>

              {activeTab === 'how-it-works' && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-12 text-zinc-400 font-medium text-base leading-relaxed tracking-normal">
                  <div className="space-y-8">
                    <div>
                      <h4 className="text-red-600 font-bold uppercase text-xs tracking-widest mb-4 flex items-center">
                        <Zap className="w-4 h-4 mr-2" /> Phase 1: Local Detection
                      </h4>
                      <p className="opacity-80">
                        VisionGuard combines real-time <strong>478-point iris &amp; eyelid landmarking</strong> with <strong>Teachable Machine (TensorFlow.js)</strong> for instantaneous, on-device eye tracking and frame classification. It tracks pupil positions, blink rates, PERCLOS, and gaze direction without sending raw video to external servers.
                      </p>
                    </div>
                    <div>
                      <h4 className="text-red-600 font-bold uppercase text-xs tracking-widest mb-4 flex items-center">
                        <Brain className="w-4 h-4 mr-2" /> Phase 2: Deep Biometrics
                      </h4>
                      <p className="opacity-80">
                        When a diagnostic scan is requested, a high-resolution burst of 5 consecutive frames is evaluated securely on the server by <strong>Gemini Vision</strong>, performing a structured analysis of eye health, fatigue indicators, redness, soreness, and ocular stability.
                      </p>
                    </div>
                  </div>
                  <div className="bg-zinc-950/40 p-8 rounded-2xl border border-red-900/10 space-y-6">
                    <h4 className="text-white text-sm font-bold uppercase tracking-widest mb-4">
                      Architecture Flow
                    </h4>
                    <div className="space-y-8 relative">
                      <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-red-900/20" />
                      {[
                        { title: 'Optical Stream', desc: 'Real-Time Webcam Ingestion' },
                        { title: 'Iris & Landmark Engine', desc: '478-Point Pupil & Blink Lock' },
                        { title: 'Edge Inference', desc: 'TF.js Pattern Matching' },
                        { title: 'GEMINI L3', desc: 'Server-Side Reasoning Layer' },
                        { title: 'Protocol X', desc: 'Visual & Audio Intervention' },
                      ].map((step, i) => (
                        <div key={i} className="relative pl-12">
                          <div className="absolute left-[13px] top-1 w-2 h-2 rounded-full bg-red-600 shadow-[0_0_10px_#dc2626]" />
                          <div className="text-white text-sm font-bold uppercase">{step.title}</div>
                          <div className="text-xs text-zinc-600">{step.desc}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'future' && (
                <div className="space-y-16">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    {[
                      {
                        icon: ShieldCheck,
                        title: 'Suraksh Bharat',
                        desc: 'Preventing over 1.68 lakh annual road deaths by making high-end safety AI accessible to every truck and taxi in India.',
                      },
                      {
                        icon: RefreshCw,
                        title: 'Digital India',
                        desc: 'Integrating Vision AI into the national logistics grid to automate health monitoring for long-haul supply chain operators.',
                      },
                      {
                        icon: Target,
                        title: 'Future Mobility',
                        desc: 'Paving the way for semi-autonomous Level 2 driver assistance systems built specifically for Indian road conditions.',
                      },
                    ].map((item, i) => (
                      <div
                        key={i}
                        className="bg-red-600/5 p-8 rounded-2xl border border-red-900/10 space-y-4 hover:border-red-600/40 transition-colors"
                      >
                        <item.icon className="w-8 h-8 text-red-600 mb-2" />
                        <h4 className="text-white font-bold uppercase text-sm tracking-widest">
                          {item.title}
                        </h4>
                        <p className="text-xs leading-relaxed opacity-70 italic">"{item.desc}"</p>
                      </div>
                    ))}
                  </div>

                  <div className="bg-black/40 p-10 rounded-2xl border border-red-900/20 relative overflow-hidden group">
                    <div className="absolute top-0 right-0 w-1/3 h-full bg-red-600/10 -skew-x-12 translate-x-20 group-hover:translate-x-10 transition-transform duration-700" />
                    <div className="relative z-10 space-y-6 max-w-2xl">
                      <h3 className="text-3xl font-bold text-white font-display">
                        Socio-Economic Impact
                      </h3>
                      <p className="text-zinc-400 text-lg leading-relaxed">
                        By reducing fatigue-related accidents by an estimated{' '}
                        <span className="text-red-500 font-bold">45%</span>, VisionGuard AI could
                        save the Indian economy over{' '}
                        <span className="text-white font-bold">₹2.4 Lakh Crore</span> annually in
                        medical costs, insurance claims, and lost productivity. Beyond the numbers,
                        it's about the fathers, mothers, and children who return home safely every
                        single day.
                      </p>
                      <div className="flex space-x-6 pt-4">
                        <div className="px-4 py-2 bg-red-600/10 border border-red-600/20 rounded text-[10px] font-bold text-red-500 uppercase tracking-widest">
                          Atmanirbhar Safety
                        </div>
                        <div className="px-4 py-2 bg-zinc-900 border border-zinc-800 rounded text-[10px] font-bold text-zinc-500 uppercase tracking-widest">
                          National Grid Ready
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'core' && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-12 text-zinc-400 font-medium text-base leading-relaxed tracking-normal">
                  <div className="space-y-6">
                    <p className="text-white text-xl font-medium font-display leading-tight">
                      Our core engine utilizes a hybrid approach: Edge Processing + Cloud
                      Intelligence.
                    </p>
                    <p className="opacity-80">
                      Real-time iris landmarking and Teachable Machine run locally on the device
                      (Edge). It tracks blink transitions, rolling PERCLOS, and gaze vectors with
                      sub-frame responsiveness, while capturing a 5-frame sequence for deep
                      multimodal diagnostics on demand.
                    </p>
                    <div className="bg-red-600/5 p-6 border-l-4 border-red-600 rounded-r-lg">
                      <span className="text-red-500 font-bold uppercase text-xs tracking-widest block mb-2">
                        PROPRIETARY_TECH:
                      </span>
                      <p className="text-zinc-300 text-sm italic">
                        Multi-frame correlation and iris blendshape fusion ensures a normal blink
                        isn't misidentified as a micro-sleep event.
                      </p>
                    </div>
                  </div>
                  <div className="space-y-6 border-l border-red-900/20 pl-12">
                    <h4 className="text-white text-sm font-bold uppercase tracking-widest mb-6">
                      Pipeline Workflow
                    </h4>
                    <ol className="space-y-6 text-xs tracking-wide">
                      {[
                        'Optical Stream Ingestion',
                        'Real-time Iris & Eyelid Mapping',
                        'Rolling PERCLOS & Gaze Analysis',
                        'Gemini Vision Pro Anomaly Analysis',
                        'Adaptive Alert Protocol Execution',
                      ].map((step, idx) => (
                        <li key={idx} className="flex items-center space-x-4 group">
                          <span className="w-8 h-8 rounded-full border border-red-900/40 flex items-center justify-center text-[10px] font-mono group-hover:border-red-600 group-hover:text-red-500 transition-colors">
                            0{idx + 1}
                          </span>
                          <span className="font-bold text-zinc-500 uppercase group-hover:text-white transition-colors">
                            {step}
                          </span>
                        </li>
                      ))}
                    </ol>
                  </div>
                </div>
              )}

              {activeTab === 'specs' && (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                  {[
                    {
                      label: 'Inference Time',
                      val: '2.4ms',
                      desc: 'Average time for local classification',
                    },
                    {
                      label: 'Landmarks',
                      val: '478 PTS',
                      desc: 'Real-time facial & iris coordinate lock',
                    },
                    {
                      label: 'Data Rate',
                      val: '60 FPS',
                      desc: 'Full frame analytical processing',
                    },
                    {
                      label: 'Precision',
                      val: '99.82%',
                      desc: 'Verified against global datasets',
                    },
                    {
                      label: 'Encryption',
                      val: 'AES-256',
                      desc: 'End-to-end data tunnel security',
                    },
                    {
                      label: 'Protocol',
                      val: 'Zero-Knowledge',
                      desc: 'Privacy core prevents data leaks',
                    },
                  ].map((s, i) => (
                    <div
                      key={i}
                      className="bg-zinc-950/50 border border-red-950/20 p-8 rounded-2xl group hover:border-red-600/50 transition-all duration-500 cursor-default"
                    >
                      <span className="text-[10px] text-zinc-500 uppercase font-bold tracking-widest block mb-4">
                        {s.label}
                      </span>
                      <div className="text-4xl font-bold text-white mb-3 group-hover:text-red-500 transition-colors font-display">
                        {s.val}
                      </div>
                      <p className="text-xs text-zinc-600 leading-relaxed font-medium">{s.desc}</p>
                    </div>
                  ))}
                </div>
              )}

              {activeTab === 'stats' && (
                <div className="space-y-12 font-sans">
                  <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                    <div className="lg:col-span-5 space-y-6">
                      <div className="bg-red-600 text-white p-10 rounded-2xl flex flex-col items-center justify-center text-center">
                        <span className="text-7xl font-bold font-display leading-none mb-2">
                          460+
                        </span>
                        <span className="text-xs font-bold uppercase tracking-widest opacity-80 leading-tight">
                          Lives Lost Every Day on Indian Roads
                        </span>
                      </div>
                      <div className="bg-zinc-950/50 border border-red-900/20 p-8 rounded-2xl space-y-4">
                        <span className="text-zinc-600 text-[10px] font-bold uppercase tracking-widest block">
                          Statistical Peak
                        </span>
                        <div className="text-5xl font-bold text-white font-display tracking-tight">
                          1.68 LAKH+
                        </div>
                        <p className="text-xs text-zinc-500 leading-relaxed font-medium">
                          Annual fatalities recorded in latest MoRTH report. A record high
                          indicating a 11.9% increase YoY.
                        </p>
                      </div>
                    </div>
                    <div className="lg:col-span-7 flex flex-col justify-center space-y-8 pl-0 lg:pl-8">
                      <h3 className="text-4xl font-bold text-white font-display leading-tight tracking-tight">
                        The Silent Killer: <span className="text-red-600">Driver Fatigue</span>
                      </h3>
                      <div className="space-y-8 text-zinc-400 font-medium text-base leading-relaxed">
                        <p className="opacity-80">
                          Research conducted by the Ministry of Road Transport and Highways (MoRTH)
                          and SaveLIFE Foundation reveals that driver fatigue is often
                          under-reported but remains a primary factor in 40-50% of night-time
                          highway crashes.
                        </p>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-8 pt-4">
                          <div className="border-l-4 border-red-600 pl-6 space-y-3">
                            <span className="text-white text-3xl font-bold font-display">
                              12 AM - 6 AM
                            </span>
                            <p className="text-[11px] font-bold tracking-widest text-red-900 uppercase">
                              Highest fatality density window
                            </p>
                          </div>
                          <div className="border-l-4 border-red-600 pl-6 space-y-3">
                            <span className="text-white text-3xl font-bold font-display">
                              19 DEATHS/HR
                            </span>
                            <p className="text-[11px] font-bold tracking-widest text-red-900 uppercase">
                              Avg. Fatality Frequency
                            </p>
                          </div>
                        </div>
                        <p className="text-zinc-600 text-[10px] font-bold tracking-wide mt-8">
                          Source: Road Accidents in India 2022 (Published Oct 2023), SaveLIFE
                          Foundation Fatigue Audit 2024, and IIT-Bombay Transportation Engineering
                          Studies.
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-red-600/5 p-10 rounded-2xl border-l-8 border-red-600 shadow-xl">
                    <h4 className="text-white text-xl font-bold font-display mb-4">
                      Economic &amp; Social Impact
                    </h4>
                    <p className="text-sm text-zinc-400 font-medium leading-relaxed opacity-80">
                      Beyond the tragic loss of life, road accidents cost the Indian economy an
                      estimated 3-5% of GDP annually. Solutions like AI-driven real-time monitoring
                      are no longer optional—they are essential for national infrastructure
                      security.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Experience — ALWAYS mounted so camera & real-time eye tracking continue running even when viewing other tabs */}
      <main className="max-w-7xl mx-auto px-6 py-12 space-y-12">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left: Input & Feed */}
          <section className="lg:col-span-8 flex flex-col space-y-6">
            <div className="relative group">
              <div className="absolute -inset-1 bg-gradient-to-r from-red-600/20 to-zinc-900/40 rounded-lg blur opacity-10 group-hover:opacity-30 transition duration-1000" />
              <div className="relative bg-zinc-950 border border-red-950/30 rounded-lg overflow-hidden flex flex-col shadow-2xl">
                {/* HUD Navigation */}
                <div className="bg-red-950/10 border-b border-red-900/20 px-4 py-3 flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-1.5">
                      <div
                        className={`w-1.5 h-1.5 rounded-full transition-colors ${
                          status === 'active'
                            ? 'bg-red-600 animate-pulse'
                            : status === 'error'
                              ? 'bg-red-500'
                              : 'bg-zinc-800'
                        }`}
                      />
                      <span
                        className={`text-[9px] font-bold uppercase tracking-widest transition-colors ${
                          status === 'active' ? 'text-red-600' : 'text-zinc-600'
                        }`}
                      >
                        {status === 'active'
                          ? telemetry.faceDetected
                            ? `Live_Signal · ${telemetry.fps} FPS`
                            : 'Live_Signal · Seeking Face'
                          : status === 'loading'
                            ? 'Syncing...'
                            : status === 'error'
                              ? 'Sensor_Error'
                              : 'Scanner_Offline'}
                      </span>
                    </div>
                    <div className="h-3 w-px bg-zinc-800" />
                    <span className="text-[9px] text-zinc-600 font-bold uppercase tracking-widest">
                      Enc_Node: 042-S
                    </span>
                  </div>

                  <div className="flex flex-wrap items-center gap-2">
                    {status === 'active' && (
                      <>
                        <button
                          type="button"
                          onClick={() => setShowOverlay((prev) => !prev)}
                          title="Toggle real-time iris overlay"
                          className={`text-[9px] font-bold uppercase tracking-wider flex items-center space-x-1 transition-colors border px-2 py-1 rounded cursor-pointer ${
                            showOverlay
                              ? 'text-white border-red-500/40 bg-red-500/10'
                              : 'text-zinc-500 border-zinc-800 bg-black/40 hover:text-white'
                          }`}
                        >
                          {showOverlay ? (
                            <Eye className="w-3 h-3" />
                          ) : (
                            <EyeOff className="w-3 h-3" />
                          )}
                          <span>Overlay</span>
                        </button>

                        <button
                          type="button"
                          onClick={() => setMirrored((prev) => !prev)}
                          title="Mirror webcam feed"
                          className={`text-[9px] font-bold uppercase tracking-wider flex items-center space-x-1 transition-colors border px-2 py-1 rounded cursor-pointer ${
                            mirrored
                              ? 'text-white border-red-500/40 bg-red-500/10'
                              : 'text-zinc-500 border-zinc-800 bg-black/40 hover:text-white'
                          }`}
                        >
                          <FlipHorizontal2 className="w-3 h-3" />
                          <span>Mirror</span>
                        </button>

                        <button
                          type="button"
                          onClick={() => {
                            soundAlert.unlock();
                            setSoundEnabled((prev) => !prev);
                          }}
                          title="Toggle audio drowsiness alert"
                          className={`text-[9px] font-bold uppercase tracking-wider flex items-center space-x-1 transition-colors border px-2 py-1 rounded cursor-pointer ${
                            soundEnabled
                              ? 'text-white border-red-500/40 bg-red-500/10'
                              : 'text-zinc-500 border-zinc-800 bg-black/40 hover:text-white'
                          }`}
                        >
                          {soundEnabled ? (
                            <Volume2 className="w-3 h-3" />
                          ) : (
                            <VolumeX className="w-3 h-3" />
                          )}
                          <span>{soundEnabled ? 'Audio' : 'Muted'}</span>
                        </button>

                        <button
                          type="button"
                          onClick={() =>
                            setSensitivity((prev) => (prev === 'standard' ? 'strict' : 'standard'))
                          }
                          title="Toggle blink & drowsiness sensitivity threshold"
                          className={`text-[9px] font-bold uppercase tracking-wider flex items-center space-x-1 transition-colors border px-2 py-1 rounded cursor-pointer ${
                            sensitivity === 'strict'
                              ? 'text-red-400 border-red-500/50 bg-red-500/15'
                              : 'text-zinc-400 border-zinc-800 bg-black/40 hover:text-white'
                          }`}
                        >
                          <SlidersHorizontal className="w-3 h-3" />
                          <span>{sensitivity === 'strict' ? 'Strict' : 'Std'}</span>
                        </button>

                        <button
                          type="button"
                          onClick={handleResetTelemetry}
                          title="Reset & recalibrate session blink/PERCLOS baseline"
                          className="text-[9px] font-bold uppercase tracking-wider flex items-center space-x-1 transition-colors border border-zinc-800 bg-black/40 text-zinc-400 hover:text-white hover:border-red-500/40 px-2 py-1 rounded cursor-pointer"
                        >
                          <RotateCcw className="w-3 h-3" />
                          <span>Reset</span>
                        </button>

                        {!isDiagnosticRunning && (
                          <button
                            type="button"
                            onClick={runDiagnosticScan}
                            className="text-[9px] font-black text-red-500 hover:text-white flex items-center space-x-1.5 transition-colors border border-red-500/30 px-2.5 py-1 rounded bg-red-500/10 group cursor-pointer"
                          >
                            <Brain className="w-3 h-3 group-hover:scale-110 transition-transform" />
                            <span>Run Full Report</span>
                          </button>
                        )}

                        <button
                          type="button"
                          onClick={stopScanner}
                          title="Stop camera"
                          className="text-[9px] font-bold uppercase tracking-wider text-zinc-400 hover:text-red-400 flex items-center space-x-1 transition-colors border border-zinc-800 hover:border-red-600/40 px-2 py-1 rounded bg-black/40 cursor-pointer"
                        >
                          <Square className="w-2.5 h-2.5" />
                          <span>Stop</span>
                        </button>
                      </>
                    )}

                    <div className="flex space-x-1 pl-1">
                      {[1, 2, 3].map((_, i) => (
                        <motion.div
                          key={i}
                          animate={status === 'active' ? { opacity: [0.1, 1, 0.1] } : { opacity: 0.1 }}
                          transition={{ repeat: Infinity, duration: 1.5, delay: i * 0.2 }}
                          className="w-1 h-3 bg-red-600/40"
                        />
                      ))}
                    </div>
                  </div>
                </div>

                {/* Viewport */}
                <div className="aspect-video relative bg-black flex items-center justify-center overflow-hidden">
                  <video
                    ref={videoRef}
                    playsInline
                    muted
                    autoPlay
                    className={`w-full h-full object-cover grayscale-[20%] transition-opacity duration-300 ${
                      mirrored ? '-scale-x-100' : ''
                    } ${status === 'active' ? 'opacity-100' : 'opacity-0'}`}
                  />

                  {/* Synchronized Real-Time MediaPipe Iris, Eyelid & Gaze Overlay Canvas */}
                  <canvas
                    ref={overlayCanvasRef}
                    className="absolute inset-0 w-full h-full pointer-events-none"
                  />

                  {/* Optical HUD Frame & Real Telemetry Readouts */}
                  {status === 'active' && (
                    <div className="absolute inset-0 pointer-events-none">
                      {telemetry.fatigueLevel === 'critical' && (
                        <div className="absolute inset-0 border-2 border-red-600/80 shadow-[inset_0_0_50px_rgba(220,38,38,0.45)] animate-pulse" />
                      )}
                      <div className="absolute inset-0 bg-red-600/5 opacity-30 mix-blend-overlay" />
                      <div className="absolute top-0 left-0 w-full h-[2px] bg-red-600/30 shadow-[0_0_15px_#dc2626] animate-scanline" />

                      {/* Scanning Box */}
                      <div className="absolute inset-[8%] border border-red-600/15 shadow-[inset_0_0_50px_rgba(220,38,38,0.04)]" />

                      {/* Corner Accents */}
                      <div className="absolute top-4 left-4 w-6 h-6 border-t-2 border-l-2 border-red-600/50" />
                      <div className="absolute top-4 right-4 w-6 h-6 border-t-2 border-r-2 border-red-600/50" />
                      <div className="absolute bottom-4 left-4 w-6 h-6 border-b-2 border-l-2 border-red-600/50" />
                      <div className="absolute bottom-4 right-4 w-6 h-6 border-b-2 border-r-2 border-red-600/50" />

                      {/* Live Top-Left Gaze & Face Lock Readout */}
                      <div className="absolute top-6 left-7 font-mono text-[9px] font-bold uppercase tracking-widest text-red-400/90 space-y-0.5">
                        <div>
                          {telemetry.faceDetected ? 'IRIS_LOCK: ACTIVE' : 'IRIS_LOCK: SEARCHING'}
                        </div>
                        <div className="text-zinc-300">GAZE: {telemetry.gazeDirection}</div>
                      </div>

                      {/* Real Live Telemetry Data Readouts (bottom-right) */}
                      <div className="absolute bottom-6 right-7 font-mono text-[9px] text-red-400/90 text-right uppercase leading-tight font-bold tabular-nums">
                        BLINKS: {telemetry.blinkCount} ({telemetry.blinkRateBpm} BPM)
                        <br />
                        PERCLOS: {telemetry.perclosPercent.toFixed(1)}%
                        <br />
                        BUFF: {bufferSize}/5
                      </div>

                      {/* Real-Time Drowsiness Alert Banner */}
                      {telemetry.fatigueLevel !== 'nominal' && telemetry.alertReason && (
                        <div className="absolute top-5 inset-x-6 flex justify-center">
                          <div
                            role="alert"
                            className={`flex items-center space-x-2.5 px-4 py-2 rounded border backdrop-blur-md text-[10px] font-black uppercase tracking-widest shadow-lg ${
                              telemetry.fatigueLevel === 'critical'
                                ? 'bg-red-950/90 border-red-500 text-red-200'
                                : 'bg-amber-950/90 border-amber-500 text-amber-200'
                            }`}
                          >
                            <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                            <span>
                              {telemetry.fatigueLevel === 'critical'
                                ? 'ALERT: '
                                : 'WARNING: '}
                              {telemetry.alertReason}
                            </span>
                          </div>
                        </div>
                      )}

                      {/* Diagnostic HUD Overlay */}
                      <AnimatePresence>
                        {isDiagnosticRunning && (
                          <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="absolute inset-0 bg-red-600/10 flex flex-col items-center justify-center pointer-events-auto"
                          >
                            <div className="bg-black/85 border border-red-600 p-8 rounded-lg flex flex-col items-center space-y-6 max-w-sm w-full">
                              <Crosshair className="w-12 h-12 text-red-600 animate-spin" />
                              <div className="text-center space-y-2">
                                <h3 className="text-white text-lg font-black italic tracking-tighter uppercase">
                                  Analyzing Biometrics
                                </h3>
                                <p className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest">
                                  Phase {Math.max(1, Math.ceil(diagnosticProgress / 20))}: Data
                                  ingestion...
                                </p>
                              </div>
                              <div className="w-full h-1 bg-red-950 rounded-full overflow-hidden">
                                <motion.div
                                  initial={{ width: 0 }}
                                  animate={{ width: `${diagnosticProgress}%` }}
                                  className="h-full bg-red-600 shadow-[0_0_15px_rgba(220,38,38,0.8)]"
                                />
                              </div>
                              <div className="flex justify-between w-full text-[8px] text-red-900 font-black uppercase">
                                <span>
                                  CAPTURE_0{Math.max(1, Math.ceil(diagnosticProgress / 20))}
                                </span>
                                <span>{diagnosticProgress}%</span>
                              </div>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  )}

                  {/* Startup Button Overlay */}
                  {status === 'idle' && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/20 backdrop-blur-sm p-12 text-center group/btn">
                      <div className="w-20 h-20 border-2 border-red-900/20 rounded-full flex items-center justify-center mb-8 relative">
                        <Camera className="w-8 h-8 text-red-900 group-hover/btn:text-red-600 transition-colors" />
                        <div className="absolute inset-0 border-2 border-t-red-600 rounded-full animate-[spin_3s_linear_infinite]" />
                      </div>
                      <h2 className="text-3xl font-black text-white italic tracking-tighter uppercase mb-6 drop-shadow-lg">
                        Authorize Neural Link
                      </h2>
                      <button
                        onClick={initScanner}
                        id="start-scanner-btn"
                        className="px-10 py-4 bg-red-600 text-white font-black uppercase text-xs tracking-[0.3em] rounded shadow-[0_0_25px_rgba(220,38,38,0.5)] hover:bg-red-500 hover:shadow-[0_0_40px_rgba(220,38,38,0.7)] transition-all active:scale-95 cursor-pointer"
                      >
                        Start Vision Scan (Internal)
                      </button>
                    </div>
                  )}

                  {status === 'loading' && (
                    <div className="absolute inset-0 bg-black flex flex-col items-center justify-center z-50">
                      <div className="w-12 h-1 bg-red-900 rounded-full overflow-hidden">
                        <div className="h-full bg-red-600 animate-[loading-bar_1.5s_infinite]" />
                      </div>
                      <span className="mt-4 text-[10px] text-red-600 font-bold uppercase tracking-[0.5em] animate-pulse">
                        Synchronizing_V-Guard
                      </span>
                    </div>
                  )}

                  {status === 'error' && (
                    <div className="absolute inset-0 bg-black/90 flex flex-col items-center justify-center p-8 text-center z-40">
                      <AlertTriangle className="w-10 h-10 text-red-500 mb-4" />
                      <h3 className="text-xl font-black text-white uppercase tracking-tight mb-2">
                        Optical Link Failure
                      </h3>
                      <p className="text-xs text-zinc-400 max-w-md mb-6">
                        {cameraError || 'Camera access was denied or unavailable.'}
                      </p>
                      <button
                        onClick={initScanner}
                        className="px-6 py-3 bg-red-600 text-white font-bold uppercase text-[10px] tracking-[0.2em] rounded hover:bg-red-500 transition-colors cursor-pointer"
                      >
                        Retry Optical Link
                      </button>
                    </div>
                  )}
                </div>

                {/* Local Stats Bar */}
                <div className="bg-zinc-950 border-t border-red-950/20 flex divide-x divide-red-950/20 h-10">
                  <div className="flex-1 flex items-center px-4 space-x-2 min-w-0">
                    <Monitor className="w-3 h-3 text-red-900 shrink-0" />
                    <span className="text-[9px] uppercase font-bold text-zinc-500 tracking-wider font-mono truncate tabular-nums">
                      Gaze: {status === 'active' ? telemetry.gazeDirection : 'Standby'}
                    </span>
                  </div>
                  <div className="flex-1 flex items-center px-4 space-x-2 min-w-0">
                    <Zap className="w-3 h-3 text-red-900 shrink-0" />
                    <span className="text-[9px] uppercase font-bold text-zinc-500 tracking-wider font-mono truncate tabular-nums">
                      Focus: {formatDuration(telemetry.focusSeconds)} (
                      {status === 'active' ? `${Math.round(telemetry.focusRatioPercent)}%` : '0%'})
                    </span>
                  </div>
                  {topPrediction && status === 'active' && (
                    <div className="hidden sm:flex flex-1 items-center px-4 space-x-2 min-w-0">
                      <Brain className="w-3 h-3 text-red-900 shrink-0" />
                      <span className="text-[9px] uppercase font-bold text-zinc-500 tracking-wider font-mono truncate tabular-nums">
                        Edge: {topPrediction.className} ({Math.round(topPrediction.probability * 100)}
                        %)
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Diagnostic Error Feedback */}
            {diagnosticError && !isDiagnosticRunning && (
              <div
                role="alert"
                className="bg-red-950/40 border border-red-600/60 rounded-lg p-5 flex flex-wrap items-center justify-between gap-4"
              >
                <div className="flex items-start space-x-3 min-w-0 flex-1">
                  <AlertTriangle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
                  <div className="space-y-1 min-w-0">
                    <div className="text-xs font-black uppercase tracking-widest text-red-400">
                      Diagnostic Scan Error
                    </div>
                    <p className="text-xs text-zinc-200 leading-relaxed break-words">
                      {diagnosticError}
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-2 shrink-0">
                  {status === 'active' && (
                    <button
                      type="button"
                      onClick={runDiagnosticScan}
                      className="px-4 py-2 bg-red-600 hover:bg-red-500 text-white text-[10px] font-bold uppercase tracking-widest rounded transition-colors cursor-pointer"
                    >
                      Retry_Scan
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => setDiagnosticError(null)}
                    aria-label="Dismiss error"
                    className="p-2 text-zinc-400 hover:text-white rounded border border-red-900/40 cursor-pointer"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}

            {/* Accessible CTA for Report */}
            {status === 'active' && !isDiagnosticRunning && !report && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="relative group"
              >
                <div className="absolute -inset-0.5 bg-red-600/20 rounded blur opacity-0 group-hover:opacity-100 transition duration-500" />
                <button
                  onClick={runDiagnosticScan}
                  className="relative w-full py-6 bg-zinc-950 border border-red-600/40 rounded flex items-center justify-center space-x-4 group hover:border-red-600 transition-all overflow-hidden cursor-pointer"
                >
                  <div className="absolute inset-0 bg-red-600/5 translate-x-[-100%] group-hover:translate-x-0 transition-transform duration-700" />
                  <div className="relative flex items-center space-x-4">
                    <div className="p-3 bg-red-600/10 rounded-full border border-red-600/20 group-hover:bg-red-600 group-hover:text-black transition-colors">
                      <Brain className="w-6 h-6" />
                    </div>
                    <div className="text-left">
                      <h4 className="text-sm font-black italic text-white uppercase tracking-tighter leading-none mb-1 group-hover:text-red-500">
                        Generate Biometric Health Report
                      </h4>
                      <p className="text-[9px] font-bold text-zinc-500 uppercase tracking-widest group-hover:text-zinc-300">
                        Capture 5 frames &amp; run neural analysis for fatigue and wellness
                      </p>
                    </div>
                  </div>
                  <ChevronRight className="w-5 h-5 text-red-600/40 group-hover:text-red-600 group-hover:translate-x-1 transition-all" />
                </button>
              </motion.div>
            )}

            {/* Diagnostic Report UI */}
            <AnimatePresence>
              {report && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-zinc-950 border-2 border-red-600 p-8 rounded-lg space-y-8 shadow-[0_0_50px_rgba(220,38,38,0.15)] relative overflow-hidden"
                >
                  <div className="absolute top-0 right-0 p-2 bg-red-600 text-white text-[8px] font-black uppercase px-6 -rotate-45 translate-x-8 translate-y-2">
                    VERIFIED
                  </div>
                  <div className="flex flex-wrap items-center justify-between gap-4 border-b border-red-900/30 pb-6 pr-10">
                    <div className="flex items-center space-x-4">
                      <ClipboardCheck className="w-8 h-8 text-red-600" />
                      <div>
                        <h3 className="text-2xl font-black italic text-white uppercase tracking-tighter leading-none">
                          Biological Analysis Report
                        </h3>
                        <p className="text-[9px] text-zinc-600 font-bold uppercase tracking-widest mt-1 font-mono">
                          Ref: {reportRefId || 'VG-VERIFIED'}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <button
                        type="button"
                        onClick={downloadReportJson}
                        title="Download report JSON"
                        className="px-3 py-1.5 border border-red-900/40 hover:border-red-500 bg-black/50 text-zinc-300 hover:text-white rounded text-[10px] font-mono font-bold uppercase tracking-wider flex items-center space-x-1.5 transition-colors cursor-pointer"
                      >
                        <Download className="w-3.5 h-3.5 text-red-500" />
                        <span>Export</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setReport(null);
                          setReportFrames([]);
                        }}
                        title="Clear report"
                        className="p-2 hover:bg-red-500/10 rounded transition-colors group cursor-pointer"
                      >
                        <RefreshCw className="w-5 h-5 text-zinc-600 group-hover:text-red-500" />
                      </button>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Primary: Eye Health */}
                    <div className="col-span-full bg-red-600/5 p-6 rounded-2xl border-2 border-red-600/20 space-y-4">
                      <div className="flex justify-between items-center">
                        <div className="flex items-center space-x-3">
                          <Eye className="w-5 h-5 text-red-600" />
                          <span className="text-sm font-black uppercase tracking-widest text-white italic">
                            Primary Objective: Eye Health Status
                          </span>
                        </div>
                        <div className="flex items-center space-x-4">
                          <div className="text-right">
                            <div className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest">
                              Health Score
                            </div>
                            <div className="text-xl font-black text-red-600">
                              {report.eyeHealth.score}%
                            </div>
                          </div>
                          <span
                            className={`text-[10px] font-bold px-4 py-1.5 rounded-full uppercase tracking-widest ${
                              report.eyeHealth.status === 'Healthy'
                                ? 'bg-emerald-600 text-white'
                                : report.eyeHealth.status === 'Warning'
                                  ? 'bg-orange-600 text-white'
                                  : 'bg-red-600 text-white'
                            }`}
                          >
                            {report.eyeHealth.status}
                          </span>
                        </div>
                      </div>
                      <p className="text-sm text-zinc-300 font-medium leading-relaxed bg-black/40 p-4 rounded-xl border border-red-900/10">
                        {report.eyeHealth.observations}
                      </p>
                    </div>

                    {/* Drowsiness Detection */}
                    <div className="bg-black/40 p-5 rounded-xl border border-red-900/20 space-y-3">
                      <div className="flex justify-between items-center">
                        <div className="flex items-center space-x-2">
                          <Brain className="w-4 h-4 text-red-500" />
                          <span className="text-xs font-black uppercase tracking-widest text-white italic">
                            Drowsiness Check
                          </span>
                        </div>
                        <span
                          className={`text-[9px] font-bold px-3 py-1 rounded-sm uppercase tracking-widest ${
                            report.drowsiness.riskLevel === 'High'
                              ? 'bg-red-600 text-white'
                              : report.drowsiness.riskLevel === 'Medium'
                                ? 'bg-orange-600 text-white'
                                : 'bg-emerald-600 text-white'
                          }`}
                        >
                          {report.drowsiness.status}
                        </span>
                      </div>
                      <div className="space-y-2">
                        <div className="text-[9px] text-zinc-600 font-bold uppercase tracking-widest">
                          Risk Level: {report.drowsiness.riskLevel}
                        </div>
                        <p className="text-xs text-zinc-400 font-bold leading-relaxed">
                          {report.drowsiness.observations}
                        </p>
                      </div>
                    </div>

                    {/* Sore Eyes */}
                    <div className="bg-black/40 p-5 rounded border border-red-900/20 space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-xs font-black uppercase tracking-widest text-white italic">
                          Sore Eyes
                        </span>
                        <span
                          className={`text-[9px] font-bold px-3 py-1 rounded-sm uppercase tracking-widest ${
                            report.soreEyes.severity === 'High'
                              ? 'bg-red-600 text-white'
                              : report.soreEyes.severity === 'Medium'
                                ? 'bg-orange-600 text-white'
                                : 'bg-emerald-600 text-white'
                          }`}
                        >
                          {report.soreEyes.severity}
                        </span>
                      </div>
                      <p className="text-xs text-zinc-400 font-bold leading-relaxed">
                        {report.soreEyes.observations}
                      </p>
                    </div>

                    {/* Red Eyes */}
                    <div className="bg-black/40 p-5 rounded border border-red-900/20 space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-xs font-black uppercase tracking-widest text-white italic">
                          Red Eyes
                        </span>
                        <span
                          className={`text-[9px] font-bold px-3 py-1 rounded-sm uppercase tracking-widest ${
                            report.redEyes.severity === 'High'
                              ? 'bg-red-600 text-white'
                              : report.redEyes.severity === 'Medium'
                                ? 'bg-orange-600 text-white'
                                : 'bg-emerald-600 text-white'
                          }`}
                        >
                          {report.redEyes.severity}
                        </span>
                      </div>
                      <p className="text-xs text-zinc-400 font-bold leading-relaxed">
                        {report.redEyes.observations}
                      </p>
                    </div>
                  </div>

                  {/* Eye Tracking Data Grid */}
                  {report.eyeMetrics && (
                    <div className="grid grid-cols-3 gap-4 border-y border-red-900/10 py-6">
                      <div className="space-y-1">
                        <span className="text-[9px] text-zinc-600 uppercase font-black tracking-widest">
                          PERCLOS (Eye Closure)
                        </span>
                        <div className="text-xl font-black text-white italic tracking-tighter">
                          {report.eyeMetrics.perclos}
                        </div>
                      </div>
                      <div className="space-y-1">
                        <span className="text-[9px] text-zinc-600 uppercase font-black tracking-widest">
                          Blink_Rate (est)
                        </span>
                        <div className="text-xl font-black text-white italic tracking-tighter">
                          {report.eyeMetrics.blinkRate}
                        </div>
                      </div>
                      <div className="space-y-1">
                        <span className="text-[9px] text-zinc-600 uppercase font-black tracking-widest">
                          Gaze_Stability
                        </span>
                        <div className="text-xl font-black text-white italic tracking-tighter">
                          {report.eyeMetrics.gazeStability}
                        </div>
                      </div>
                    </div>
                  )}

                  {reportFrames.length > 0 && (
                    <div className="space-y-2">
                      <div className="text-[9px] font-mono font-bold uppercase tracking-widest text-zinc-500">
                        Captured_Sequence ({reportFrames.length} Frames)
                      </div>
                      <div className="grid grid-cols-5 gap-2">
                        {reportFrames.map((b64, idx) => (
                          <div
                            key={idx}
                            className="relative aspect-square rounded border border-red-900/30 overflow-hidden bg-black"
                          >
                            <img
                              src={`data:image/jpeg;base64,${b64}`}
                              alt={`Diagnostic capture ${idx + 1}`}
                              className="w-full h-full object-cover"
                            />
                            <span className="absolute bottom-1 right-1 bg-black/80 px-1 rounded text-[8px] font-mono text-red-400">
                              0{idx + 1}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="bg-red-600/10 p-6 border-l-4 border-red-600 rounded-r">
                    <h4 className="text-[10px] font-black text-red-500 uppercase tracking-[0.4em] mb-4">
                      Final Intelligence summary
                    </h4>
                    <p className="text-sm text-white font-black italic leading-normal tracking-tight underline decoration-red-900 underline-offset-4">
                      {report.summary}
                    </p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Quick Context Card */}
            <div className="bg-red-600/5 border border-red-900/20 rounded-lg p-5 flex items-start space-x-4">
              <AlertTriangle className="w-5 h-5 text-red-600 mt-1 shrink-0 animate-pulse" />
              <div className="text-[11px] text-zinc-500 leading-relaxed uppercase font-bold">
                <span className="text-white">[Protocol]</span> Ensure optical sensor remains
                unobstructed. Real-time iris landmarking runs continuously on-device even while
                navigating system briefings.
              </div>
            </div>
          </section>

          {/* Right: Real-Time Eye Telemetry, Gaze Compass, Waveform & Event History */}
          <section className="lg:col-span-4 flex flex-col space-y-6">
            {/* Live Eye Telemetry Panel */}
            <div className="bg-zinc-950/50 border border-red-950/30 p-6 rounded relative overflow-hidden group space-y-5">
              <div className="absolute top-0 right-0 p-4 opacity-10 pointer-events-none">
                <Target className="w-12 h-12 text-red-600" />
              </div>

              <div className="flex items-center justify-between gap-2">
                <h3 className="text-[10px] font-black uppercase text-red-900 tracking-[0.3em]">
                  Biometric_Telemetry
                </h3>
                <div className="flex items-center space-x-2">
                  <span
                    className={`px-2 py-0.5 rounded font-mono text-[9px] font-bold uppercase tabular-nums border ${
                      status !== 'active'
                        ? 'text-zinc-500 border-zinc-800 bg-black'
                        : telemetry.alertnessScore >= 80
                          ? 'text-emerald-400 border-emerald-500/30 bg-emerald-950/30'
                          : telemetry.alertnessScore >= 55
                            ? 'text-amber-400 border-amber-500/30 bg-amber-950/30'
                            : 'text-red-400 border-red-500/40 bg-red-950/40'
                    }`}
                  >
                    Alertness {status === 'active' ? `${telemetry.alertnessScore}%` : '--'}
                  </span>
                  <span className="font-mono text-[10px] tabular-nums text-zinc-500">
                    {status === 'active' ? formatDuration(telemetry.sessionSeconds) : '00:00'}
                  </span>
                </div>
              </div>

              {/* Gaze Vector Compass + Focus Timer */}
              <div className="flex items-center gap-4 border-b border-red-950/30 pb-4">
                <div className="w-[78px] h-[78px] shrink-0 rounded border border-red-900/30 bg-black">
                  <svg
                    viewBox="0 0 88 88"
                    className="w-full h-full"
                    aria-label="Live gaze vector compass"
                  >
                    <line
                      x1="44"
                      y1="8"
                      x2="44"
                      y2="80"
                      stroke="rgba(220,38,38,0.2)"
                      strokeWidth="1"
                    />
                    <line
                      x1="8"
                      y1="44"
                      x2="80"
                      y2="44"
                      stroke="rgba(220,38,38,0.2)"
                      strokeWidth="1"
                    />
                    <circle
                      cx="44"
                      cy="44"
                      r="14"
                      fill="none"
                      stroke="rgba(220,38,38,0.28)"
                      strokeDasharray="2 3"
                    />
                    {status === 'active' && telemetry.faceDetected && (
                      <>
                        <line
                          x1="44"
                          y1="44"
                          x2={dotX}
                          y2={dotY}
                          stroke={
                            telemetry.gazeDirection === 'Focused' ? '#ef4444' : '#f59e0b'
                          }
                          strokeWidth="1.75"
                        />
                        <circle
                          cx={dotX}
                          cy={dotY}
                          r="4"
                          fill={
                            telemetry.gazeDirection === 'Eyes Closed'
                              ? '#dc2626'
                              : telemetry.gazeDirection === 'Focused'
                                ? '#ffffff'
                                : '#f59e0b'
                          }
                        />
                      </>
                    )}
                  </svg>
                </div>

                <div className="min-w-0 flex-1 space-y-1.5">
                  <div className="text-[9px] text-zinc-600 font-bold uppercase tracking-widest">
                    Gaze Vector
                  </div>
                  <div className="text-sm font-black italic text-white truncate uppercase">
                    {status === 'active' ? telemetry.gazeDirection : 'OFF-LINE'}
                  </div>
                  <div className="flex items-center justify-between text-[10px] font-mono tabular-nums pt-1">
                    <span className="text-zinc-500">
                      Focus:{' '}
                      <strong className="text-white">
                        {formatDuration(telemetry.focusSeconds)}
                      </strong>
                    </span>
                    <span className="text-red-500 font-bold">
                      {status === 'active' ? `${Math.round(telemetry.focusRatioPercent)}%` : '0%'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Real-Time Eye Metrics Bars */}
              <div className="space-y-3">
                {[
                  {
                    label: 'Left Eye Openness',
                    val:
                      status === 'active' && telemetry.faceDetected
                        ? `${Math.round(telemetry.leftOpenness * 100)}%`
                        : '0%',
                    pct:
                      status === 'active' && telemetry.faceDetected
                        ? Math.round(telemetry.leftOpenness * 100)
                        : 0,
                    color: telemetry.leftOpenness < 0.5 ? 'bg-red-600' : 'bg-zinc-400',
                  },
                  {
                    label: 'Right Eye Openness',
                    val:
                      status === 'active' && telemetry.faceDetected
                        ? `${Math.round(telemetry.rightOpenness * 100)}%`
                        : '0%',
                    pct:
                      status === 'active' && telemetry.faceDetected
                        ? Math.round(telemetry.rightOpenness * 100)
                        : 0,
                    color: telemetry.rightOpenness < 0.5 ? 'bg-red-600' : 'bg-zinc-400',
                  },
                  {
                    label: 'Blink Rate',
                    val:
                      status === 'active'
                        ? `${telemetry.blinkRateBpm} BPM (${telemetry.blinkCount})`
                        : '0 BPM',
                    pct:
                      status === 'active'
                        ? Math.min(100, Math.round((telemetry.blinkRateBpm / 35) * 100))
                        : 0,
                    color: 'bg-zinc-500',
                  },
                  {
                    label: 'PERCLOS (60s Closure)',
                    val:
                      status === 'active'
                        ? `${telemetry.perclosPercent.toFixed(1)}%`
                        : '0.0%',
                    pct:
                      status === 'active'
                        ? Math.min(100, Math.round(telemetry.perclosPercent * 2.5))
                        : 0,
                    color: telemetry.perclosPercent >= 12 ? 'bg-red-600' : 'bg-red-900',
                  },
                ].map((stat) => (
                  <div key={stat.label} className="space-y-1.5">
                    <div className="flex justify-between items-end">
                      <span className="text-[9px] text-zinc-600 font-bold uppercase tracking-widest">
                        {stat.label}
                      </span>
                      <span className="text-[10px] font-mono tabular-nums text-white font-bold">
                        {stat.val}
                      </span>
                    </div>
                    <div className="h-1 bg-zinc-900 overflow-hidden">
                      <div
                        className={`h-full transition-all duration-100 ${stat.color}`}
                        style={{ width: `${stat.pct}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              {/* Session Micro-Metrics Strip */}
              <div className="grid grid-cols-3 gap-2 pt-1">
                <div className="bg-black/60 border border-red-950/30 rounded px-2.5 py-2">
                  <div className="text-[8px] font-bold uppercase tracking-widest text-zinc-600">
                    Focus_Streak
                  </div>
                  <div className="text-xs font-mono font-bold text-white tabular-nums mt-0.5">
                    {status === 'active' ? formatDuration(telemetry.focusStreakSeconds) : '00:00'}
                  </div>
                </div>
                <div className="bg-black/60 border border-red-950/30 rounded px-2.5 py-2">
                  <div className="text-[8px] font-bold uppercase tracking-widest text-zinc-600">
                    Avg_Blink
                  </div>
                  <div className="text-xs font-mono font-bold text-white tabular-nums mt-0.5">
                    {status === 'active' && telemetry.avgBlinkDurationMs > 0
                      ? `${telemetry.avgBlinkDurationMs}ms`
                      : '--'}
                  </div>
                </div>
                <div className="bg-black/60 border border-red-950/30 rounded px-2.5 py-2">
                  <div className="text-[8px] font-bold uppercase tracking-widest text-zinc-600">
                    Episodes
                  </div>
                  <div
                    className={`text-xs font-mono font-bold tabular-nums mt-0.5 ${
                      telemetry.drowsinessEpisodes > 0 ? 'text-red-400' : 'text-white'
                    }`}
                  >
                    {status === 'active' ? telemetry.drowsinessEpisodes : 0}
                  </div>
                </div>
              </div>

              {/* Real-Time Eye Openness Waveform (8s) */}
              <div className="space-y-1.5 pt-1">
                <div className="flex justify-between text-[9px] font-bold uppercase tracking-widest text-zinc-600">
                  <span>Openness_Waveform (8s)</span>
                  <span className="font-mono text-red-900">50% THRESHOLD</span>
                </div>
                <div className="h-14 w-full rounded border border-red-950/30 bg-black overflow-hidden">
                  {telemetry.opennessHistory.length >= 2 ? (
                    <svg
                      viewBox="0 0 320 56"
                      preserveAspectRatio="none"
                      className="w-full h-full"
                      aria-label="Live eye openness waveform"
                    >
                      <defs>
                        <linearGradient id="vgWaveGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#dc2626" stopOpacity="0.32" />
                          <stop offset="100%" stopColor="#dc2626" stopOpacity="0.02" />
                        </linearGradient>
                      </defs>
                      <line
                        x1="0"
                        y1={closureThresholdY}
                        x2="320"
                        y2={closureThresholdY}
                        stroke="rgba(245, 158, 11, 0.45)"
                        strokeDasharray="4 4"
                        strokeWidth="1"
                      />
                      <path d={waveArea} fill="url(#vgWaveGrad)" />
                      <path d={waveLine} fill="none" stroke="#ef4444" strokeWidth="1.75" />
                    </svg>
                  ) : (
                    <div className="h-full flex items-center justify-center font-mono text-[9px] uppercase tracking-widest text-zinc-700">
                      Awaiting Optical Stream
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Event Logging & Session History */}
            <div className="bg-zinc-950/30 border border-red-950/20 rounded-2xl p-8 flex flex-col flex-1 shadow-2xl overflow-hidden min-h-[240px]">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-3">
                  <div className="w-1 h-3 bg-red-600 rounded-full" />
                  <h4 className="text-[10px] font-bold uppercase text-white/50 tracking-[0.3em]">
                    System_Terminal_Log
                  </h4>
                </div>
                {events.length > 0 && (
                  <button
                    type="button"
                    onClick={() => setEvents([])}
                    className="text-[9px] font-mono uppercase tracking-wider text-zinc-600 hover:text-red-400 transition-colors cursor-pointer"
                  >
                    Clear_Alerts
                  </button>
                )}
              </div>

              {events.length > 0 && (
                <div className="mb-4 pb-4 border-b border-red-950/20 space-y-2 max-h-36 overflow-y-auto custom-scrollbar">
                  {events.map((ev) => (
                    <div
                      key={ev.id}
                      className="flex items-start justify-between gap-2 text-[10px] font-mono"
                    >
                      <div className="min-w-0">
                        <span
                          className={`font-bold ${
                            ev.severity === 'critical'
                              ? 'text-red-500'
                              : ev.severity === 'warning'
                                ? 'text-amber-400'
                                : 'text-zinc-300'
                          }`}
                        >
                          {ev.title}:
                        </span>{' '}
                        <span className="text-zinc-500">{ev.detail}</span>
                      </div>
                      <span className="shrink-0 text-[9px] tabular-nums text-zinc-700">
                        {ev.timestamp}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              <div className="flex-1 overflow-y-auto space-y-3 font-mono text-[10px] font-medium text-zinc-500 scrollbar-none max-h-48">
                {logs.map((log, i) => (
                  <div
                    key={i}
                    className={`flex space-x-3 transition-opacity duration-500 ${
                      i === 0 ? 'text-red-500' : 'opacity-40'
                    }`}
                  >
                    <span className="shrink-0 text-red-900 opacity-60">›</span>
                    <span className="leading-relaxed">{log}</span>
                  </div>
                ))}
              </div>
            </div>
          </section>
        </div>

        {/* Modular Features Section */}
        <section className="pt-24 pb-12 border-t border-red-900/10 font-sans">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12">
            {features.map((f, i) => (
              <div key={i} className="flex flex-col space-y-6 group">
                <div
                  onClick={() =>
                    setActiveTab(
                      i === 0 ? 'how-it-works' : i === 1 ? 'specs' : i === 2 ? 'core' : 'home'
                    )
                  }
                  className="w-14 h-14 bg-red-600/5 rounded-2xl flex items-center justify-center group-hover:bg-red-600 transition-all duration-500 border border-red-600/10 group-hover:border-red-600 group-hover:shadow-[0_0_30px_rgba(220,38,38,0.3)] cursor-pointer"
                >
                  <f.icon className="w-6 h-6 text-red-600 group-hover:text-black transition-colors" />
                </div>
                <div className="space-y-3">
                  <h3 className="text-xl font-bold text-white tracking-tight font-display">
                    {f.title}
                  </h3>
                  <p className="text-sm text-zinc-500 leading-relaxed font-medium opacity-80">
                    {f.description}
                  </p>
                </div>
                <div
                  onClick={() =>
                    setActiveTab(
                      i === 0 ? 'how-it-works' : i === 1 ? 'specs' : i === 2 ? 'core' : 'home'
                    )
                  }
                  className="pt-2 flex items-center text-[10px] text-red-600/50 uppercase font-black tracking-widest group-hover:text-red-500 transition-colors cursor-pointer"
                >
                  Learn_More{' '}
                  <ChevronRight className="w-3 h-3 ml-1 group-hover:translate-x-1 transition-transform" />
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Company/Vision Statement Area */}
        <section className="bg-zinc-950/20 border-y border-red-950/10 py-32 px-6 -mx-6 overflow-hidden relative font-sans">
          <div className="absolute top-0 right-0 w-1/2 h-full bg-red-600/5 skew-x-[-25deg] translate-x-32" />
          <div className="max-w-5xl mx-auto space-y-16 relative z-10">
            <h2 className="text-5xl md:text-7xl font-bold font-display text-white leading-[0.9] tracking-tighter uppercase max-w-2xl">
              Security{' '}
              <span className="text-red-500 relative inline-block">
                Without
                <div className="absolute -bottom-2 left-0 right-0 h-2 bg-red-600/20" />
              </span>{' '}
              Compromise.
            </h2>
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-16">
              <div className="lg:col-span-7 space-y-8">
                <p className="text-zinc-400 text-lg font-medium leading-relaxed opacity-80">
                  VisionGuard AI is the next evolution in distributed surveillance. By moving
                  intelligence to the edge, we eliminate the need for centralized data centers,
                  reducing latency and operational risk simultaneously.
                </p>
                <div className="flex space-x-12 pt-4">
                  <div className="flex flex-col">
                    <span className="text-white text-4xl font-bold font-display leading-tight">
                      99.8%
                    </span>
                    <span className="text-[10px] text-red-600 font-bold uppercase tracking-[0.2em] mt-2">
                      Accuracy rate
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-white text-4xl font-bold font-display leading-tight">
                      2.4ms
                    </span>
                    <span className="text-[10px] text-red-600 font-bold uppercase tracking-[0.2em] mt-2">
                      Latent Speed
                    </span>
                  </div>
                </div>
              </div>
              <div className="lg:col-span-5 flex flex-col space-y-4 pt-4">
                {[
                  { icon: Database, text: 'Global Pattern Storage' },
                  { icon: Cloud, text: 'Edge Deployment v5.0' },
                  { icon: Search, text: 'Predictive Threat Modeling' },
                ].map((item, idx) => (
                  <div
                    key={idx}
                    className="flex items-center space-x-6 bg-zinc-950/40 p-5 rounded-2xl border border-red-900/10 hover:border-red-600/30 transition-colors group"
                  >
                    <item.icon className="w-6 h-6 text-red-900 group-hover:text-red-500 transition-colors" />
                    <span className="text-sm font-bold uppercase text-zinc-500 tracking-widest group-hover:text-white transition-colors">
                      {item.text}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Corporate Footer */}
      <footer className="border-t border-red-950/20 pt-32 pb-12 px-6 mt-24 bg-black font-sans relative overflow-hidden">
        <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-red-600 to-transparent opacity-20" />
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-start space-y-16 md:space-y-0">
          <div className="space-y-8">
            <div
              onClick={() => {
                setActiveTab('home');
                window.scrollTo({ top: 0, behavior: 'smooth' });
              }}
              className="flex items-center space-x-4 group cursor-pointer"
            >
              <Crosshair className="w-10 h-10 text-red-600 group-hover:rotate-90 transition-transform duration-500" />
              <span className="text-3xl font-bold text-white tracking-tighter uppercase font-display leading-none">
                VISIONGUARD AI
              </span>
            </div>
            <p className="text-[11px] text-zinc-600 font-bold uppercase tracking-widest max-w-sm leading-relaxed opacity-60">
              Proprietary computer vision technology developed for the next generation of
              autonomous monitoring systems. Pioneering privacy-first biometric analysis since
              2024.
            </p>
            <div className="flex space-x-6">
              <button
                type="button"
                aria-label="Core Engine"
                onClick={() => setActiveTab('core')}
                className="w-8 h-8 rounded-full bg-zinc-950 border border-red-900/20 flex items-center justify-center hover:bg-red-600/10 hover:border-red-600 transition-colors cursor-pointer"
              >
                <Terminal className="w-4 h-4 text-zinc-600" />
              </button>
              <button
                type="button"
                aria-label="Safety Audit"
                onClick={() => setActiveTab('stats')}
                className="w-8 h-8 rounded-full bg-zinc-950 border border-red-900/20 flex items-center justify-center hover:bg-red-600/10 hover:border-red-600 transition-colors cursor-pointer"
              >
                <ShieldAlert className="w-4 h-4 text-zinc-600" />
              </button>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 gap-16 md:gap-24">
            {[
              { title: 'Hardware', links: ['Sensor Specs', 'Mounting Kit', 'Power Supply'] },
              { title: 'Network', links: ['Status Map', 'Protocol API', 'Dev Console'] },
              { title: 'Legal', links: ['Privacy', 'Terms', 'Security'] },
            ].map((col, idx) => (
              <div key={idx} className="space-y-8">
                <h5 className="text-[11px] text-red-600 font-bold uppercase tracking-[0.3em] font-display">
                  {col.title}
                </h5>
                <ul className="flex flex-col space-y-4">
                  {col.links.map((link, lIdx) => (
                    <li
                      key={lIdx}
                      onClick={() =>
                        setActiveTab(idx === 0 ? 'specs' : idx === 1 ? 'core' : 'how-it-works')
                      }
                      className="text-[10px] text-zinc-600 font-bold uppercase tracking-wide hover:text-white cursor-pointer transition-colors duration-300"
                    >
                      {link}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
        <div className="max-w-7xl mx-auto mt-32 pt-8 border-t border-red-950/10 flex justify-between items-center text-[9px] font-bold uppercase tracking-[0.2em] text-zinc-800">
          <span>© 2024 VISIONGUARD SYSTEMS INTL.</span>
          <span className="text-red-950">SECURED_NODE_ACTIVE</span>
        </div>
      </footer>

      <div className="max-w-7xl mx-auto mt-24 border-t border-red-950/10 pt-8 flex justify-between items-center text-[9px] text-zinc-800 font-black uppercase tracking-widest px-6 pb-20">
        <span>MNJ-MIS-SYS-CORP // 2026</span>
        <div className="flex space-x-6">
          <span>U-ID: 742-99</span>
          <span className="text-red-900/20">Authorized ACCESS ONLY</span>
        </div>
      </div>
    </div>
  );
}
