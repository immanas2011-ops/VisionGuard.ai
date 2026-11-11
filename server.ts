import "dotenv/config";
import path from "path";
import { fileURLToPath } from "url";
import express from "express";
import type { Request, Response } from "express";
import { GoogleGenAI, ThinkingLevel, Type } from "@google/genai";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;
const PRIMARY_MODEL = "gemini-3.8-flash";
const SECONDARY_MODEL = "gemini-3.1-flash-lite";
const FALLBACK_MODEL = "gemini-flash-latest";

let cachedClient: GoogleGenAI | null = null;
let cachedApiKey: string | undefined;

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY || process.env.API_KEY;
  if (!apiKey) {
    throw Object.assign(
      new Error("GEMINI_API_KEY is not configured on the server. Add it in Settings > Secrets."),
      { status: 500 }
    );
  }
  if (!cachedClient || cachedApiKey !== apiKey) {
    cachedClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        apiVersion: "v1beta",
        retryOptions: {
          attempts: 1,
        },
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
    cachedApiKey = apiKey;
  }
  return cachedClient;
}

function getErrorStatus(err: unknown): number {
  if (err && typeof err === "object") {
    const rec = err as Record<string, unknown>;
    const direct = Number(rec.status ?? rec.statusCode);
    if (Number.isInteger(direct) && direct >= 400 && direct <= 599) {
      return direct;
    }
    const cause = rec.cause as Record<string, unknown> | undefined;
    const nested = Number(cause?.status);
    if (Number.isInteger(nested) && nested >= 400 && nested <= 599) {
      return nested;
    }
  }
  const msg = err instanceof Error ? err.message : String(err ?? "");
  if (msg.includes("404") || msg.includes("NOT_FOUND")) return 404;
  if (msg.includes("403") || msg.includes("PERMISSION_DENIED")) return 403;
  if (msg.includes("401") || msg.includes("UNAUTHENTICATED")) return 401;
  if (msg.includes("400") || msg.includes("API_KEY_INVALID")) return 400;
  if (msg.includes("429") || msg.includes("RESOURCE_EXHAUSTED")) return 429;
  return 500;
}

function formatErrorMessage(err: unknown): string {
  const raw =
    err instanceof Error && err.message
      ? err.message
      : String(err || "Failed to run biometric analysis.");
  try {
    const match = raw.match(/\{[\s\S]*\}/);
    if (match) {
      const parsed = JSON.parse(match[0]) as { error?: { message?: string }; message?: string };
      const nestedMsg = parsed?.error?.message || parsed?.message;
      if (typeof nestedMsg === "string" && nestedMsg.trim()) {
        return nestedMsg.trim();
      }
    }
  } catch {
    // ignore JSON parse errors when formatting error strings
  }
  return raw;
}

function isTimeoutError(err: unknown): boolean {
  return (
    err instanceof Error &&
    (err.name === "TimeoutError" ||
      err.name === "AbortError" ||
      err.message.toLowerCase().includes("aborted") ||
      err.message.toLowerCase().includes("timeout"))
  );
}

function parseJsonResponse(rawText: string) {
  const cleaned = rawText
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();
  const match = cleaned.match(/\{[\s\S]*\}/);
  const jsonText = match ? match[0] : cleaned;
  const parsed = JSON.parse(jsonText);
  return {
    eyeHealth: {
      status: String(parsed?.eyeHealth?.status || "Healthy"),
      observations: String(
        parsed?.eyeHealth?.observations || "Ocular landmarks and clarity evaluated across captured frames."
      ),
      score:
        typeof parsed?.eyeHealth?.score === "number" && Number.isFinite(parsed.eyeHealth.score)
          ? Math.max(0, Math.min(100, Math.round(parsed.eyeHealth.score)))
          : 85,
    },
    drowsiness: {
      status: String(parsed?.drowsiness?.status || "Alert"),
      observations: String(
        parsed?.drowsiness?.observations || "No prolonged micro-sleep episodes observed."
      ),
      riskLevel: String(parsed?.drowsiness?.riskLevel || "Low"),
    },
    soreEyes: {
      severity: String(parsed?.soreEyes?.severity || "Low"),
      observations: String(
        parsed?.soreEyes?.observations || "No significant periorbital strain or squinting detected."
      ),
    },
    redEyes: {
      severity: String(parsed?.redEyes?.severity || "Low"),
      observations: String(
        parsed?.redEyes?.observations || "Sclera appears clear within normal visual variance."
      ),
    },
    eyeMetrics: {
      perclos: String(parsed?.eyeMetrics?.perclos || "0.0%"),
      blinkRate: String(parsed?.eyeMetrics?.blinkRate || "15 BPM"),
      gazeStability: String(parsed?.eyeMetrics?.gazeStability || "Stable"),
    },
    summary: String(
      parsed?.summary || "Biometric vision scan completed successfully."
    ),
  };
}

const DIAGNOSTIC_PROMPT_BASE =
  "Analyze these consecutive frames from a vision sensor. Your PRIMARY OBJECTIVE is to assess Eye Health and detect Drowsiness with HIGH PRECISION. Focus on mapping facial landmarks to verify anatomical alignment. Provide a health diagnostic report focusing on: 1. Eye Health (status of eyes, clarity, iris/pupil health, micro-vessels), 2. Drowsiness Detection (detect micro-sleeps, heavy lids, ptosis, decreased blink latency, eye closure frequency, yawning), 3. External Eye Conditions (soreness, puffiness, redness). Also estimate 4. Eye Metrics: PERCLOS (Percentage of Eye Closure), Blink Rate, and Gaze Stability. Return the results in a structured JSON format.";

const DIAGNOSTIC_RESPONSE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    eyeHealth: {
      type: Type.OBJECT,
      properties: {
        status: { type: Type.STRING, description: "Healthy, Warning, or Critical" },
        observations: { type: Type.STRING },
        score: { type: Type.NUMBER, description: "0-100 score of eye health" },
      },
      required: ["status", "observations", "score"],
    },
    drowsiness: {
      type: Type.OBJECT,
      properties: {
        status: { type: Type.STRING, description: "Alert, Tired, or Drowsy" },
        observations: { type: Type.STRING },
        riskLevel: { type: Type.STRING, description: "Low, Medium, or High" },
      },
      required: ["status", "observations", "riskLevel"],
    },
    soreEyes: {
      type: Type.OBJECT,
      properties: {
        severity: { type: Type.STRING, description: "Low, Medium, or High" },
        observations: { type: Type.STRING },
      },
      required: ["severity", "observations"],
    },
    redEyes: {
      type: Type.OBJECT,
      properties: {
        severity: { type: Type.STRING, description: "Low, Medium, or High" },
        observations: { type: Type.STRING },
      },
      required: ["severity", "observations"],
    },
    eyeMetrics: {
      type: Type.OBJECT,
      properties: {
        perclos: { type: Type.STRING },
        blinkRate: { type: Type.STRING },
        gazeStability: { type: Type.STRING },
      },
      required: ["perclos", "blinkRate", "gazeStability"],
    },
    summary: { type: Type.STRING },
  },
  required: ["eyeHealth", "drowsiness", "soreEyes", "redEyes", "eyeMetrics", "summary"],
};

async function startServer() {
  const app = express();

  app.use(express.json({ limit: "25mb" }));

  app.get("/api/health", (_req: Request, res: Response) => {
    res.json({ ok: true, model: PRIMARY_MODEL });
  });

  app.post("/api/diagnostic", async (req: Request, res: Response) => {
    try {
      const rawFrames = req.body?.frames;
      if (!Array.isArray(rawFrames) || rawFrames.length === 0 || rawFrames.length > 10) {
        res.status(400).json({ error: "Expected 1 to 10 captured image frames." });
        return;
      }

      const frames: { mimeType: string; data: string }[] = [];
      for (const item of rawFrames) {
        if (typeof item !== "string" || item.trim().length === 0) {
          res.status(400).json({ error: "Each frame must be a non-empty base64 string." });
          return;
        }
        const trimmed = item.trim();
        let mimeType = "image/jpeg";
        let data = trimmed;
        const match = /^data:(image\/[a-zA-Z0-9+.-]+);base64,(.+)$/.exec(trimmed);
        if (match) {
          mimeType = match[1];
          data = match[2];
        } else if (trimmed.includes(",")) {
          data = trimmed.split(",")[1] || "";
        }
        if (!data) {
          res.status(400).json({ error: "Invalid base64 image frame data." });
          return;
        }
        frames.push({ mimeType, data });
      }

      const telemetry = req.body?.telemetry;
      const telemetryNote =
        telemetry && typeof telemetry === "object"
          ? `\n\nMeasured on-device optical sensor context during capture window: Blink Rate=${Number(telemetry.blinkRateBpm ?? 0)} BPM, PERCLOS=${Number(telemetry.perclosPercent ?? 0).toFixed(1)}%, Gaze=${String(telemetry.gazeDirection ?? "Focused")}, Attention=${Math.round(Number(telemetry.focusRatioPercent ?? 100))}%, Alertness=${Math.round(Number(telemetry.alertnessScore ?? 100))}/100.`
          : "";

      const ai = getGeminiClient();

      const invokeModel = (model: string, useLowThinking: boolean, timeoutMs: number) =>
        ai.models.generateContent({
          model,
          contents: {
            parts: [
              ...frames.map((frame) => ({
                inlineData: {
                  mimeType: frame.mimeType,
                  data: frame.data,
                },
              })),
              {
                text: `${DIAGNOSTIC_PROMPT_BASE}${telemetryNote}`,
              },
            ],
          },
          config: {
            abortSignal: AbortSignal.timeout(timeoutMs),
            responseMimeType: "application/json",
            responseSchema: DIAGNOSTIC_RESPONSE_SCHEMA,
            ...(useLowThinking
              ? { thinkingConfig: { thinkingLevel: ThinkingLevel.LOW } }
              : {}),
          },
        });

      const candidates = [
        { model: PRIMARY_MODEL, useLowThinking: true, timeoutMs: 18000 },
        { model: SECONDARY_MODEL, useLowThinking: false, timeoutMs: 15000 },
        { model: FALLBACK_MODEL, useLowThinking: false, timeoutMs: 18000 },
      ];

      let response;
      let lastErr: unknown;
      const attemptNotes: string[] = [];
      for (let i = 0; i < candidates.length; i += 1) {
        const candidate = candidates[i];
        const startedAt = Date.now();
        try {
          response = await invokeModel(
            candidate.model,
            candidate.useLowThinking,
            candidate.timeoutMs
          );
          break;
        } catch (err) {
          lastErr = err;
          const elapsedMs = Date.now() - startedAt;
          const status = getErrorStatus(err);
          const isTimeout = isTimeoutError(err);
          const reasonTag = isTimeout ? "timeout" : String(status);
          attemptNotes.push(`${candidate.model}=${reasonTag}(${elapsedMs}ms)`);
          const shouldFallback =
            status === 404 || status === 429 || status >= 500 || isTimeout;
          if (!shouldFallback || i === candidates.length - 1) {
            throw Object.assign(err instanceof Error ? err : new Error(String(err)), {
              attemptNotes,
            });
          }
          console.warn(
            `Model ${candidate.model} failed (${reasonTag}, ${elapsedMs}ms), falling back to ${candidates[i + 1].model}`
          );
        }
      }

      if (!response) {
        throw lastErr || new Error("No diagnostic response received from Gemini API.");
      }

      const text = response.text?.trim();
      if (!text) {
        res.status(502).json({ error: "Empty diagnostic response from Gemini API." });
        return;
      }

      const reportData = parseJsonResponse(text);
      res.json(reportData);
    } catch (err) {
      console.error("Diagnostic API error:", err);
      const isTimeout = isTimeoutError(err);
      const status = isTimeout ? 504 : getErrorStatus(err);
      const baseMessage = isTimeout
        ? "Biometric AI analysis timed out. Please try again."
        : formatErrorMessage(err);
      const notes =
        err && typeof err === "object" && "attemptNotes" in err && Array.isArray((err as { attemptNotes?: unknown }).attemptNotes)
          ? ((err as { attemptNotes: string[] }).attemptNotes).join(", ")
          : "";
      const message = notes ? `${baseMessage} [${notes}]` : baseMessage;
      res.status(status).json({ error: message });
    }
  });

  if (process.env.NODE_ENV !== "production") {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, "dist");
    app.use(express.static(distPath));
    app.use((req, res, next) => {
      if (req.method !== "GET" || req.path.startsWith("/api/")) {
        next();
        return;
      }
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
