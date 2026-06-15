"""WS-2: Real beat / downbeat / BPM / time-signature detection.

Uses librosa (madmom failed to build on Py3.11 + Windows).

Strategy:
  1. `librosa.beat.beat_track(plp=True, start_bpm=120, trim=False)` on the full
     track to get raw beat positions.
  2. **Tempo-octave guard**: compute global tempo via onset autocorrelation,
     snap the detected BPM to the octave (x0.5 / x1 / x2) closest to it,
     clamped to 60-200 BPM.
  3. Estimate beats-per-bar from PLP accent peaks (every N beats there's a
     stronger peak); infer time signature (default 4/4).
  4. Extract `downbeats[]` by picking every beats_per_bar-th beat.

Public API:
  analyze_rhythm(wav_path, duration=None) -> dict
  snap_to_grid(t, grid) -> int/float
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


def analyze_rhythm(
    wav_path,
    duration: Optional[float] = None,
    start_bpm: int = 120,
    min_bpm: int = 60,
    max_bpm: int = 200,
) -> dict:
    """Estimate BPM, beats, downbeats, time signature from audio.

    Returns:
      {
        "bpm": int,              # octave-corrected
        "beats": [float],        # beat times in seconds
        "downbeats": [float],    # every N-th beat
        "beats_per_bar": int,    # inferred (4 is default)
        "time_sig": str,         # e.g. "4/4"
        "confidence": float,     # 0..1
      }

    On failure (librosa import, very short file), returns a sane default.
    """
    fallback = {
        "bpm": 120,
        "beats": [],
        "downbeats": [],
        "beats_per_bar": 4,
        "time_sig": "4/4",
        "confidence": 0.0,
    }
    try:
        import librosa
    except ImportError as e:
        log.warning("librosa unavailable: %s — returning default rhythm grid", e)
        return fallback

    try:
        y, sr = librosa.load(str(wav_path), sr=22050, mono=True, duration=duration)
    except Exception as e:
        log.warning("librosa.load failed: %s", e)
        return fallback

    if len(y) < sr:  # less than 1 second — not much we can do
        return fallback

    # 1) Tempo via onset autocorrelation (reliable octave hint)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    dtempo = librosa.feature.tempo(
        onset_envelope=onset_env, sr=sr, aggregate=None, prior=None
    ).mean()

    # 2) Beat track (raw)
    tempo, beats = librosa.beat.beat_track(
        y=y, sr=sr, start_bpm=start_bpm, trim=False
    )
    # In librosa >=0.10 tempo is an ndarray; unwrap
    if hasattr(tempo, "item"):
        tempo = float(tempo.item())
    beat_times = librosa.frames_to_time(beats, sr=sr)

    # 3) Tempo-octave correction
    octave = _pick_closest_octave(tempo, dtempo, lo=min_bpm, hi=max_bpm)
    corrected_bpm = tempo * octave
    # If octave != 1, thin or double the beat array accordingly:
    if octave > 1.5:
        # halve the beat count (every other frame)
        beat_times = beat_times[::2]
    elif octave < 0.75:
        # double the beats — insert midpoints
        dups = np.zeros(len(beat_times) * 2 - 1)
        dups[::2] = beat_times
        dups[1::2] = (beat_times[:-1] + beat_times[1:]) / 2
        beat_times = dups
    bpm = int(round(corrected_bpm))
    bpm = max(min_bpm, min(max_bpm, bpm))

    # 4) Downbeats / beats-per-bar via accent peaks (PLP envelope)
    plp = librosa.beat.plp(y=y, sr=sr, onset_envelope=onset_env)
    beats_per_bar = _infer_beats_per_bar(plp, beats, sr, hop_length=512)
    if beats_per_bar not in (2, 3, 4, 5, 6):
        beats_per_bar = 4

    # Downbeats: every N-th beat starting at beat 0
    downbeats = [float(b) for i, b in enumerate(beat_times) if i % beats_per_bar == 0]
    time_sig = f"{beats_per_bar}/4"

    # 5) Confidence: 0..1 (onset strength energy normalized)
    onset_energy = float(np.mean(onset_env) / (np.max(onset_env) + 1e-9))
    confidence = float(np.clip(onset_energy * 1.5, 0.0, 1.0))

    return {
        "bpm": bpm,
        "beats": beat_times.tolist(),
        "downbeats": downbeats,
        "beats_per_bar": beats_per_bar,
        "time_sig": time_sig,
        "confidence": round(confidence, 3),
    }


def _pick_closest_octave(detected: float, target: float, lo: int = 60, hi: int = 200) -> float:
    """Pick the octave multiplier (0.5 / 1.0 / 2.0) that brings `detected` closest to `target`,
    subject to the final value being in [lo, hi]."""
    candidates = []
    for m in (0.5, 1.0, 2.0):
        v = detected * m
        if lo <= v <= hi:
            candidates.append((abs(v - target), m, v))
    if not candidates:
        # fallback: pick whichever octave puts us in range
        for m in (0.5, 1.0, 2.0):
            v = detected * m
            if lo <= v <= hi:
                return m
        return 1.0
    return min(candidates)[1]


def _infer_beats_per_bar(plp: np.ndarray, beats: np.ndarray, sr: int, hop_length: int = 512) -> int:
    """Pick beats-per-bar ∈ {2,3,4,5,6} with the highest sum of PLP accent at downbeats."""
    if len(beats) < 3:
        return 4
    best_n = 4
    best_score = -1.0
    for n in (2, 3, 4, 5, 6):
        score = 0.0
        count = 0
        for idx in range(0, len(beats), n):
            frame = int(beats[idx])
            if 0 <= frame < len(plp):
                score += plp[frame]
                count += 1
        if count:
            score /= count
            if score > best_score:
                best_score = score
                best_n = n
    return best_n


def snap_to_grid(t: float, grid: List[float]) -> Optional[float]:
    """Return the nearest grid time to `t` (e.g., snap onset to nearest beat)."""
    if not grid:
        return None
    arr = np.asarray(grid, dtype=float)
    return float(arr[np.argmin(np.abs(arr - t))])


__all__ = ["analyze_rhythm", "snap_to_grid"]
