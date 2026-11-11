export type CameraStatus = 'idle' | 'loading' | 'active' | 'error';

export type ActiveView = 'monitor' | 'how-it-works' | 'impact' | 'safety-data';

export type SensitivityMode = 'standard' | 'strict';

export type GazeDirection =
  | 'Focused'
  | 'Looking Left'
  | 'Looking Right'
  | 'Looking Up'
  | 'Looking Down'
  | 'Eyes Closed'
  | 'No Face';

export type FatigueLevel = 'nominal' | 'warning' | 'critical';

export interface EyeTelemetry {
  faceDetected: boolean;
  leftOpenness: number; // 0..1
  rightOpenness: number; // 0..1
  avgOpenness: number; // 0..1
  isBlinking: boolean;
  closedDurationMs: number;
  blinkCount: number;
  blinkRateBpm: number;
  avgBlinkDurationMs: number;
  perclosPercent: number; // 0..100
  alertnessScore: number; // 0..100
  drowsinessEpisodes: number;
  gazeX: number; // -1..1 (screen coordinates)
  gazeY: number; // -1..1 (screen coordinates)
  gazeDirection: GazeDirection;
  focusSeconds: number;
  focusStreakSeconds: number;
  bestFocusStreakSeconds: number;
  sessionSeconds: number;
  focusRatioPercent: number;
  fatigueLevel: FatigueLevel;
  alertReason: string | null;
  opennessHistory: number[]; // recent samples 0..1
  fps: number;
}

export interface PredictionClass {
  className: string;
  probability: number;
}

export interface SessionEvent {
  id: string;
  timestamp: string;
  type: 'drowsiness' | 'closure' | 'gaze' | 'diagnostic' | 'system';
  severity: 'info' | 'warning' | 'critical';
  title: string;
  detail: string;
}

export interface DiagnosticReport {
  eyeHealth: { status: string; observations: string; score: number };
  drowsiness: { status: string; observations: string; riskLevel: string };
  soreEyes: { severity: string; observations: string };
  redEyes: { severity: string; observations: string };
  eyeMetrics?: {
    perclos: string;
    blinkRate: string;
    gazeStability: string;
  };
  summary: string;
}
