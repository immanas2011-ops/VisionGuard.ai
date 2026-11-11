class SoundAlertEngine {
  private ctx: AudioContext | null = null;
  private lastPlayedAt = 0;

  private ensureContext(): AudioContext | null {
    if (typeof window === 'undefined') return null;
    const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioCtx) return null;
    if (!this.ctx) {
      this.ctx = new AudioCtx();
    }
    if (this.ctx.state === 'suspended') {
      void this.ctx.resume().catch(() => undefined);
    }
    return this.ctx;
  }

  public unlock() {
    this.ensureContext();
  }

  public playAlert(severity: 'warning' | 'critical' = 'critical', cooldownMs = 4500) {
    const nowMs = Date.now();
    if (nowMs - this.lastPlayedAt < cooldownMs) return;

    const ctx = this.ensureContext();
    if (!ctx) return;

    this.lastPlayedAt = nowMs;
    const start = ctx.currentTime;

    const freqs = severity === 'critical' ? [660, 880] : [520];
    freqs.forEach((freq, idx) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const t0 = start + idx * 0.14;
      const duration = 0.11;

      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, t0);

      gain.gain.setValueAtTime(0.0001, t0);
      gain.gain.exponentialRampToValueAtTime(0.14, t0 + 0.015);
      gain.gain.exponentialRampToValueAtTime(0.0001, t0 + duration);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(t0);
      osc.stop(t0 + duration + 0.02);
    });
  }
}

export const soundAlert = new SoundAlertEngine();
