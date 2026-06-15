/* =================================================================
   KENSHICHORD — SYNC ENGINE
   Sinkronisasi currentTime player → highlight baris/chord aktif.
   Mock mode: pakai timer dummy yang jalan saat tombol Play ditekan.
   ================================================================= */

(function () {
  "use strict";

  /**
   * SyncEngine binds a song to a virtual playhead.
   * Real mode: connect to YouTube IFrame player.getCurrentTime()
   * Mock mode: internal setInterval increments currentTime.
   */
  class SyncEngine {
    constructor(opts) {
      this.song = opts.song;
      this.onTick = opts.onTick || function () {};
      this.onEnd  = opts.onEnd  || function () {};
      this.currentTime = 0;
      this.playing = false;
      this.speed = 1.0;
      this.intervalId = null;
      this.lastTick = null;
    }

    setSong(song) { this.song = song; this.currentTime = 0; this._emit(); }
    setSpeed(s)   { this.speed = Math.max(0.1, s); }
    seek(t)       { this.currentTime = Math.max(0, Math.min(this.song.meta.duration_sec, t)); this._emit(); }

    play() {
      if (this.playing) return;
      this.playing = true;
      this.lastTick = performance.now();
      this.intervalId = setInterval(() => this._tick(), 33); // ~30 fps
    }

    pause() {
      this.playing = false;
      if (this.intervalId) { clearInterval(this.intervalId); this.intervalId = null; }
    }

    toggle() { this.playing ? this.pause() : this.play(); }

    restart() { this.pause(); this.seek(0); }

    _tick() {
      const now = performance.now();
      const dt = (now - this.lastTick) / 1000;
      this.lastTick = now;
      this.currentTime += dt * this.speed;
      if (this.currentTime >= this.song.meta.duration_sec) {
        this.currentTime = this.song.meta.duration_sec;
        this.pause();
        this._emit();
        this.onEnd();
        return;
      }
      this._emit();
    }

    _emit() {
      const t = this.currentTime;
      const activeLine = this._activeLine(t);
      const activeChord = this._activeChord(t);
      this.onTick({ time: t, activeLine, activeChord });
    }

    _activeLine(t) {
      const lines = this.song.lines || [];
      for (let i = 0; i < lines.length; i++) {
        if (t >= lines[i].start && t < lines[i].end) return lines[i];
      }
      return null;
    }

    _activeChord(t) {
      // Prefer chord from active line; else check bars (instrumental)
      const line = this._activeLine(t);
      if (line && line.chords && line.chords.length) {
        // Return chord PALING BARU yang sudah dimulai (largest start <= t).
        // Loop seluruh chord, update `result` setiap ketemu yang match,
        // JANGAN return di tengah loop (biar dapet yang terakhir, bukan pertama).
        let result = line.chords[0].chord;
        for (const c of line.chords) {
          if (t >= c.start) result = c.chord;
        }
        return result;
      }
      const bars = this.song.bars || [];
      for (const b of bars) {
        if (t >= b.start && t < b.end && b.chords && b.chords[0]) {
          return b.chords[0].chord;
        }
      }
      return null;
    }
  }

  window.KC = window.KC || {};
  Object.assign(window.KC, { SyncEngine });
})();
