"""WS-5: Chord → syllable anchoring, with v2 override format (zero text corruption).

Auto path (no override):
  anchor_chords_auto(aligned_line, btc_segs, beat_grid)
    - For each BTC chord onset within the line's [start, end]:
      1) snap onset to the nearest beat in beat_grid
      2) find the syllable whose [start, end] contains (or is closest to) that beat
      3) emit a ChordMark with anchor_syllable_index + anchor_word_index
    - Dedupe consecutive identical chords (keep first).

Override format v2 (replaces the old '[G]す' duplicate-char hack):
  [Am]肩を[G]濡らす[Fmaj7]雨粒で
    A bracket chord attaches to the **syllable that immediately follows it**,
    with ZERO character insertion.  parse_override() returns (clean_text, marks)
    where marks = [(char_offset_in_clean, chord_str)...].

Public API:
    anchor_chords_auto(aligned_line, btc_segs, beat_grid) -> list[ChordDict]
    parse_override(raw_line, clean_text) -> (str, list[tuple[int,str]])
    marks_to_anchors(marks, clean_text, words) -> list[ChordDict]
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np

from .beats import snap_to_grid

# Chord-bracket regex — permissive: root + optional quality + optional /bass
_CHORD_RE = re.compile(
    r"\[([A-G][#b]?"
    r"(?:maj|min|m|dim|aug|sus|add|7|9|11|13|6|hdim|b5|#5|m7b5|mM7|MM7)*"
    r"(?:/[A-G][#b]?)?)\]"
)


def parse_override(raw_line: str) -> Tuple[str, List[Tuple[int, str]]]:
    """Parse v2 override: returns (clean_text, [(char_offset, chord_str)]).

    char_offset is the position in the CLEAN text where the chord anchors,
    i.e. the first character *after* the bracket.
    """
    marks = []
    clean = ""
    cursor = 0
    for m in _CHORD_RE.finditer(raw_line):
        between = raw_line[cursor:m.start()]
        clean += between
        # Anchor point = current length of clean (next char position)
        marks.append((len(clean), m.group(1)))
        cursor = m.end()
    clean += raw_line[cursor:]
    return clean, marks


def marks_to_anchors(
    marks: List[Tuple[int, str]],
    clean_text: str,
    words: List[dict],   # words from align_lines(): each has char_start/end and syllables
    beat_grid: Optional[List[float]] = None,
) -> List[dict]:
    """Map (char_offset, chord) to a ChordDict using aligned words.

    For each mark:
      1) find the word whose char_start <= offset < char_end
         (else find the next word whose char_start >= offset)
      2) within that word, find the syllable whose char_start <= offset < char_end
      3) chord.start = word.start (or snap to beat if available)
      4) chord.anchor_word_index / anchor_syllable_index
    """
    out = []
    for offset, chord in marks:
        w_idx = _find_word_by_offset(words, offset)
        if w_idx is None:
            # Fallback: attach to first word
            w_idx = 0
        w = words[w_idx]
        # syllable
        syl_idx_local = _find_syllable_by_offset(w["syllables"], offset - (w.get("_char_start", 0)))
        start_t = w.get("start")
        if beat_grid and start_t is not None:
            snapped = snap_to_grid(start_t, beat_grid)
            if snapped is not None:
                start_t = snapped
        if start_t is None:
            start_t = 0.0
        out.append({
            "chord": chord,
            "start": float(start_t),
            "anchor_word_index": w_idx,
            "anchor_syllable_index": syl_idx_local,
        })
    return out


def _find_word_by_offset(words: List[dict], offset: int) -> Optional[int]:
    # Words are sorted by char_start
    for i, w in enumerate(words):
        cs = w.get("_char_start", 0)
        ce = w.get("_char_end", cs)
        if cs <= offset < ce:
            return i
    # fallback: first word whose cs > offset
    for i, w in enumerate(words):
        cs = w.get("_char_start", 0)
        if cs >= offset:
            return i
    return len(words) - 1 if words else None


def _find_syllable_by_offset(syllables: List[dict], local_offset: int) -> int:
    for i, s in enumerate(syllables):
        cs = s.get("char_start", 0)
        ce = s.get("char_end", cs)
        if cs <= local_offset < ce:
            return i
    return 0


def anchor_chords_auto(
    aligned_line: dict,
    btc_segs: List[Tuple[float, float, str]],
    beat_grid: Optional[List[float]] = None,
    dedupe: bool = True,
) -> List[dict]:
    """Auto-anchor BTC chords onto aligned line. Zero text corruption.

    Each BTC onset within [line.start, line.end] becomes a ChordMark.
    Onsets snapped to nearest beat if beat_grid provided.
    Onsets anchored to the syllable whose span contains them (or closest).
    Consecutive identical chords deduped (keep first).
    """
    ls = aligned_line["start"]
    le = aligned_line["end"]
    # Flatten words → syllables with absolute indices
    syl_flat = []  # (word_idx, syl_idx, start, end)
    for wi, w in enumerate(aligned_line.get("words", [])):
        for si, s in enumerate(w.get("syllables", [])):
            syl_flat.append((wi, si, s.get("start"), s.get("end")))

    out = []
    for cs, _ce, chord in btc_segs:
        if not (ls <= cs < le):
            continue
        start_t = float(cs)
        if beat_grid:
            snapped = snap_to_grid(start_t, beat_grid)
            if snapped is not None:
                start_t = float(snapped)
        # Find syllable at (or nearest to) start_t
        best_wi, best_si = 0, 0
        best_dist = float("inf")
        for wi, si, ss, se in syl_flat:
            if ss is None or se is None:
                continue
            mid = (ss + se) / 2
            d = abs(mid - start_t)
            if d < best_dist:
                best_dist = d
                best_wi, best_si = wi, si
        out.append({
            "chord": chord,
            "start": start_t,
            "anchor_word_index": best_wi,
            "anchor_syllable_index": best_si,
        })

    # Dedupe consecutive identical chords (keep first)
    if dedupe and out:
        kept = [out[0]]
        for c in out[1:]:
            if c["chord"] != kept[-1]["chord"] or abs(c["start"] - kept[-1]["start"]) < 0.4:
                # drop micro-onsets too
                if c["chord"] != kept[-1]["chord"]:
                    kept.append(c)
            # else skip (duplicate)
        return kept
    return out


__all__ = ["parse_override", "marks_to_anchors", "anchor_chords_auto"]
