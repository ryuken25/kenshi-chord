"""WS-3: Key detection, chord spelling, and capo suggestion.

- Key: chroma CQT + Krumhansl-Kessler major/minor templates, all 12 rotations.
- Spelling: sharp vs flat key signatures, applied in simplify_chord.
- Capo: score each capo 0-7 by how many chords become open/easy shapes.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Krumhansl-Kessler key profiles
_KK_MAJOR = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88
])
_KK_MINOR = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17
])

# Note-name tables (for spelling)
_SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_NAMES  = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_KEY_PREFER_FLATS = {
    "F", "Bb", "Eb", "Ab", "Db", "Gb",
    "Dm", "Gm", "Cm", "Fm", "Bbm", "Ebm",
}

# Open/easy shapes on guitar
_OPEN_SHAPES = {"C", "A", "G", "E", "D", "Am", "Em", "Dm"}

# Pitch classes (semitones from C): C=0, C#=1/Bb=1, D=2, ...
def _pitch_of_root(name: str) -> Optional[int]:
    sharp_idx = {"C":0, "C#":1, "D":2, "D#":3, "E":4, "F":5,
                 "F#":6, "G":7, "G#":8, "A":9, "A#":10, "B":11}
    flat_idx  = {"C":0, "Db":1, "D":2, "Eb":3, "E":4, "F":5,
                 "Gb":6, "G":7, "Ab":8, "A":9, "Bb":10, "B":11}
    return sharp_idx.get(name, flat_idx.get(name, None))


def detect_key(wav_path, duration: Optional[float] = None) -> dict:
    """Detect musical key via Krumhansl-Kessler.

    Returns: {"key": "B minor", "mode": "minor", "confidence": 0.81, "pitch_class": 11}
    """
    empty = {"key": "C major", "mode": "major", "confidence": 0.0, "pitch_class": 0}
    try:
        import librosa
    except ImportError:
        return empty

    try:
        y, sr = librosa.load(str(wav_path), sr=22050, mono=True, duration=duration)
    except Exception as e:
        log.warning("librosa.load failed for key detection: %s", e)
        return empty

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)  # (12, T)
    mean_chroma = chroma.mean(axis=1)
    mean_chroma = mean_chroma / (mean_chroma.sum() + 1e-9)

    best = (-float("inf"), "C", "major", 0)
    for mode_name, profile in (("major", _KK_MAJOR), ("minor", _KK_MINOR)):
        for shift in range(12):
            rotated = np.roll(mean_chroma, -shift)
            score = float(np.corrcoef(rotated, profile)[0, 1])
            if np.isnan(score):
                continue
            if score > best[0]:
                best = (score, _SHARP_NAMES[shift], mode_name, shift)

    score, root, mode, pc = best
    # Spell for flats if the sharp-key is not the convention
    spelled_root = root
    if (root + "m" if mode == "minor" else root) in _KEY_PREFER_FLATS:
        spelled_root = _FLAT_NAMES[pc]
    key_str = f"{spelled_root} {mode}"

    return {
        "key": key_str,
        "mode": mode,
        "confidence": round(float(np.clip(score + 0.3, 0, 1)), 3),
        "pitch_class": pc,
    }


def _parse_btc_chord(c: str) -> Tuple[Optional[int], str, str]:
    """Parse BTC/ChordPro string like 'A:min' into (root_pc, quality, original)."""
    if c == "N":
        return None, "", c
    # BTC format: "A:min7", "F#:maj", "Bb:maj7"
    m = re.match(r"^([A-G][#b]?)(?::(.+))?$", c)
    if not m:
        return None, "", c
    root = m.group(1)
    qual = m.group(2) or ""
    pc = _pitch_of_root(root)
    return pc, qual, c


_MINOR_MODS = {"min", "m", "min7", "m7", "min9", "m9", "maj9", "mM7", "minM7", "hdim7", "m7b5"}
_FLAT_QUALITY_SET = set(_MINOR_MODS)
_MAJOR_MODS = {"maj", "", "maj7", "7", "maj6", "6", "sus2", "sus4", "aug", "9", "11", "13"}


def _spell_root(pc: int, key: str, prefer_flats: bool) -> str:
    return _FLAT_NAMES[pc] if prefer_flats else _SHARP_NAMES[pc]


def spell_chords(chords: List[str], key: str) -> List[str]:
    """Re-spell BTC chord roots for the detected key signature.

    Key is a string like 'B minor'. We pick flats for keys in _KEY_PREFER_FLATS,
    sharps for others.
    """
    prefer_flat = key in _KEY_PREFER_FLATS
    out = []
    for c in chords:
        pc, qual, orig = _parse_btc_chord(c)
        if pc is None:
            out.append(c)
            continue
        root = _spell_root(pc, key, prefer_flat)
        # BTC quality → ChordPro-ish quality
        spelled = root
        if qual:
            mapping = {
                "maj": "", "min": "m", "maj7": "maj7", "min7": "m7",
                "7": "7", "maj6": "6", "min6": "m6", "hdim7": "m7b5",
                "maj9": "maj9", "min9": "m9", "9": "9", "sus2": "sus2",
                "sus4": "sus4", "aug": "aug", "dim": "dim",
            }
            spelled += mapping.get(qual, "")
        out.append(spelled)
    return out


def suggest_capo(
    spelled_chords: List[str], key: str, max_capo: int = 7, transpose_down_allowed: bool = False
) -> dict:
    """Score each capo 0..max_capo by how many chords become open/easy shapes.

    Returns {"capo": int, "shape_chords": list[str]}.
    """
    def transpose_root(pc: int, capo: int) -> int:
        return (pc - capo) % 12

    def spell_pc_for_capo(pc: int, capo: int, prefer_flat: bool) -> str:
        return _spell_root(transpose_root(pc, capo), key, prefer_flat)

    prefer_flat = key in _KEY_PREFER_FLATS
    best = {"capo": 0, "shape_chords": list(spelled_chords), "score": 0}

    for capo in range(0, max_capo + 1):
        shapes = []
        score = 0
        qual_map: dict = {"maj":"", "":"", "min":"m", "m":"m",
                          "maj7":"maj7", "min7":"m7", "7":"7",
                          "9":"9", "aug":"aug", "dim":"dim",
                          "sus2":"sus2", "sus4":"sus4"}
        for c in spelled_chords:
            pc, qual, _ = _parse_btc_chord(c)
            if pc is None:
                shapes.append(c)
                continue
            # Use already-spelled root for score only at capo=0 (identity).
            # For capo>0 we re-spell from pitch (to keep consistent key spelling).
            new_root = _spell_root(transpose_root(pc, capo), key, prefer_flat)
            mapped_qual = qual_map.get(qual, qual)
            shape = new_root + mapped_qual
            shapes.append(shape)
            if shape in _OPEN_SHAPES:
                score += 2
            elif shape and shape[0] in _SHARP_NAMES + _FLAT_NAMES and len(shape) <= 3:
                score += 1  # any short-named chord is somewhat easy
        if score > best["score"]:
            best = {"capo": capo, "shape_chords": shapes, "score": score}

    return {"capo": best["capo"], "shape_chords": best["shape_chords"]}


__all__ = ["detect_key", "spell_chords", "suggest_capo"]
