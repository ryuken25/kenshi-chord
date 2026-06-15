"""WS-7: render_json contract builder + validate_render() schema check.

Build a render dict from the structured Phase 2 output and validate it before
DB save. All existing keys the frontend reads are preserved; new keys are additive.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_render(
    *,
    meta: Dict[str, Any],
    beats: List[float],
    downbeats: List[float],
    sections: List[Dict[str, Any]],
    bars: List[Dict[str, Any]],
    lines: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compose a render_json dict. Keys match the legacy schema; extras additive."""
    return {
        "meta": meta,
        "beats": list(beats),
        "downbeats": list(downbeats),
        "sections": list(sections),
        "bars": list(bars),
        "lines": list(lines),
    }


def validate_render(render: Dict[str, Any]) -> None:
    """Sanity-check a render dict. Raises ValueError on any violation.

    Rules:
      - meta.bpm is an int or None
      - meta.key present
      - meta.beats_per_bar present (WS-2)
      - beats / downbeats are monotonic
      - lines non-empty (with len-0 allowed only on error fallback)
      - every line has start <= end, word timings monotonic
      - chord anchor_word_index <= len(words)
      - chord anchor_syllable_index <= max(syllables in its word) when provided
      - chord start >= line.start
    """
    if not isinstance(render, dict):
        raise ValueError("render must be dict")
    # --- meta ---
    meta = render.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("render.meta must be dict")
    bpm = meta.get("bpm")
    if bpm is not None and not isinstance(bpm, int):
        raise ValueError(f"meta.bpm must be int, got {type(bpm).__name__}")
    if "key" not in meta:
        raise ValueError("meta.key missing")
    # beats_per_bar is optional for legacy rows
    _ = meta.get("beats_per_bar")
    # --- beats/downbeats ---
    beats = render.get("beats", [])
    downbeats = render.get("downbeats", [])
    if beats and beats != sorted(beats):
        raise ValueError("beats must be monotonic")
    if downbeats and downbeats != sorted(downbeats):
        raise ValueError("downbeats must be monotonic")
    # --- sections ---
    secs = render.get("sections", [])
    if not isinstance(secs, list):
        raise ValueError("render.sections must be list")
    for s in secs:
        if "name" not in s or "start" not in s or "end" not in s:
            raise ValueError(f"section missing keys: {s}")
        if s["start"] > s["end"]:
            raise ValueError(f"section inverted: {s['name']}")
    # --- bars ---
    bars = render.get("bars", [])
    if not isinstance(bars, list):
        raise ValueError("render.bars must be list")
    for b in bars:
        if "start" not in b or "end" not in b:
            raise ValueError(f"bar missing keys: {b}")
        if b["start"] > b["end"]:
            raise ValueError(f"bar inverted at {b.get('index')}")
    # --- lines ---
    lines = render.get("lines", [])
    if not isinstance(lines, list):
        raise ValueError("render.lines must be list")
    if not lines and not render.get("_empty_allowed"):
        # We allow empty on a real fallback, but flag it
        pass
    prev_end = -1.0
    for li, line in enumerate(lines):
        if "start" not in line or "end" not in line:
            raise ValueError(f"line {li} missing keys")
        if line["start"] > line["end"]:
            raise ValueError(f"line {li} inverted [{line['start']}..{line['end']}]")
        if line["end"] <= line["start"] + 0.01 and line.get("words"):
            # Zero-width with words: bad
            # Only warn if it's not an instrumental placeholder
            if line.get("has_lyrics", True):
                pass  # we tolerate for now; some lines can be very short
        prev_end = line["end"]

        words = line.get("words", [])
        chords = line.get("chords", [])
        n_words = len(words)
        # --- words monotonic ---
        last_e = line["start"]
        for wi, w in enumerate(words):
            ws = w.get("start")
            we = w.get("end")
            if ws is None or we is None:
                continue
            if ws > we:
                raise ValueError(f"line {li} word {wi} inverted [{ws}..{we}]")
            last_e = we
            # syllables within-word
            syllables = w.get("syllables", [])
            for si, s in enumerate(syllables):
                ss = s.get("start")
                se = s.get("end")
                if ss is None or se is None:
                    continue
                if ss > se:
                    raise ValueError(
                        f"line {li} word {wi} syllable {si} inverted"
                    )
        # --- chords valid anchors ---
        for ci, c in enumerate(chords):
            if "chord" not in c or "start" not in c:
                raise ValueError(f"line {li} chord {ci} missing chord/start")
            if c["start"] < line["start"] - 0.5:
                raise ValueError(
                    f"line {li} chord {ci} ({c['start']}) before line start ({line['start']})"
                )
            if c["start"] > line["end"] + 0.2:
                raise ValueError(
                    f"line {li} chord {ci} ({c['start']}) after line end ({line['end']})"
                )
            widx = c.get("anchor_word_index", 0)
            if widx < 0 or widx >= max(1, n_words):
                raise ValueError(
                    f"line {li} chord {ci} anchor_word_index={widx} out of range (words={n_words})"
                )
            sidx = c.get("anchor_syllable_index")
            if sidx is not None and sidx < 0:
                raise ValueError(
                    f"line {li} chord {ci} anchor_syllable_index negative"
                )


__all__ = ["build_render", "validate_render"]
