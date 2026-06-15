"""WS-6: Structure-aware section detection.

Instead of alternating Verse/Chorus by gap index, we:
  1) Build a chord-progression similarity matrix across vocal-segment windows.
  2) Find *repeated* progressions → label all of them "Chorus N" (shared label).
  3) Non-repeating vocal spans → Verse K (monotonically numbered).
  4) Instrumental gaps (vocal gap > 2.5 s with BTC chords) → "Instrumental".
     Instrumental gaps with no BTC chords → "Interlude".
  5) Leading silence (first_vocal > 3 s) → "Intro".
  6) Trailing silence (duration - last_vocal > 3 s) → "Outro".

Public API:
    detect_sections(words_lines, btc, duration, gap_threshold=2.5) -> list[dict]
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple, Optional


def _root_of(c: str) -> str:
    if c == "N":
        return "N"
    m = re.match(r"^([A-G][#b]?)", c)
    return m.group(1) if m else c


def _progression_of(chords: List[Tuple[float, float, str]]) -> Tuple[str, ...]:
    """Canonicalize a chord progression to a root-only tuple for similarity."""
    roots: List[str] = []
    seen: Optional[str] = None
    for _, _, c in chords:
        r = _root_of(c)
        if r in ("N", ""):
            continue
        if r != seen:
            roots.append(r)
            seen = r
    return tuple(roots)


def _segment_chords(
    btc: List[Tuple[float, float, str]], s_s: float, e_s: float
) -> List[Tuple[float, float, str]]:
    return [(cs, ce, c) for cs, ce, c in btc if ce >= s_s and cs <= e_s]


def _chord_similarity(a: Tuple[str, ...], b: Tuple[str, ...]) -> float:
    """Jaccard on chord-root sets; cheap but good enough for repeat detection."""
    if not a or not b:
        return 0.0
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def detect_sections(
    words_lines: List[dict],
    btc: List[Tuple[float, float, str]],
    duration: float,
    gap_threshold: float = 2.5,
) -> List[dict]:
    if not words_lines:
        return [{"name": "Intro", "start": 0.0, "end": duration, "has_lyrics": False}]

    first_vocal = words_lines[0]["start"]
    last_vocal = words_lines[-1]["end"]

    def btc_covers(ss: float, ee: float) -> bool:
        for cs, ce, c in btc:
            if _root_of(c) == "N":
                continue
            if cs < ee and ce > ss:
                return True
        return False

    # Find vocal gaps
    gaps: List[Tuple[float, float]] = []
    for i in range(len(words_lines) - 1):
        ge = words_lines[i]["end"]
        gs = words_lines[i + 1]["start"]
        if gs - ge > gap_threshold:
            gaps.append((ge, gs))

    # Safety: if >=6 lines and no gap, force-split on the largest gap so we don't
    # collapse to one giant "Verse"
    if not gaps and len(words_lines) >= 6:
        best = None
        for i in range(len(words_lines) - 1):
            ge = words_lines[i]["end"]
            gs = words_lines[i + 1]["start"]
            size = gs - ge
            if size > (best[0] if best else 0.0):
                best = (size, ge, gs)
        if best and best[0] > 0.05:
            gaps = [(best[1], best[2])]

    sections: List[dict] = []
    if first_vocal > 3.0 and btc_covers(0.0, first_vocal):
        sections.append({"name": "Intro", "start": 0.0, "end": float(first_vocal),
                         "has_lyrics": False})

    # Identify vocal segments between gaps
    segments: List[Tuple[float, float]] = []  # (start, end)  each vocal segment
    if not gaps:
        segments.append((float(first_vocal), float(last_vocal)))
    else:
        prev_e = first_vocal
        for ge, gs in gaps:
            if ge > prev_e + 0.2:
                segments.append((float(prev_e), float(ge)))
            prev_e = gs
        if prev_e < last_vocal - 0.2:
            segments.append((float(prev_e), float(last_vocal)))

    # Compute progression for each segment
    progs = [_progression_of(_segment_chords(btc, ss, ee)) for ss, ee in segments]

    # Group segments by progression similarity (threshold 0.6)
    groups: List[List[int]] = []  # each inner list = indices into `segments` sharing a progression
    for i, p_i in enumerate(progs):
        placed = False
        for g in groups:
            if _chord_similarity(p_i, progs[g[0]]) >= 0.6:
                g.append(i)
                placed = True
                break
        if not placed:
            groups.append([i])

    # Assign names: the most frequent / largest group = Chorus; others = Verse.
    # Heuristic: if the biggest group has ≥2 members AND has the longest
    # combined duration, it's the Chorus.
    group_duration = [sum(segments[j][1] - segments[j][0] for j in g) for g in groups]
    sorted_groups = sorted(
        enumerate(groups), key=lambda kv: (len(kv[1]), group_duration[kv[0]]), reverse=True
    )
    chorus_label: Optional[str] = None
    verse_counter = 1
    chorus_counter = 1
    name_map: List[Optional[str]] = [None] * len(segments)
    for g_idx, g_members in sorted_groups:
        if len(g_members) >= 2 and chorus_label is None:
            # Biggest repeating group is the Chorus
            name = f"Chorus {chorus_counter}"
            chorus_counter += 1
            for j in g_members:
                name_map[j] = name
        else:
            name = f"Verse {verse_counter}"
            verse_counter += 1
            for j in g_members:
                name_map[j] = name

    # Build vocal section entries interleaved with gaps
    seg_iter = iter(name_map)
    prev_e = first_vocal
    if not gaps:
        name = next(seg_iter)
        sections.append({"name": name, "start": float(first_vocal), "end": float(last_vocal),
                         "has_lyrics": True})
    else:
        for ge, gs in gaps:
            if ge > prev_e + 0.2:
                name = next(seg_iter, f"Verse {verse_counter}")
                verse_counter += 1 if name == f"Verse {verse_counter - 1}" else 0
                sections.append({"name": name, "start": float(prev_e), "end": float(ge),
                                 "has_lyrics": True})
            gap_name = "Instrumental" if btc_covers(ge, gs) else "Interlude"
            sections.append({"name": gap_name, "start": float(ge), "end": float(gs),
                             "has_lyrics": False})
            prev_e = gs
        if prev_e < last_vocal - 0.2:
            name = next(seg_iter, f"Verse {verse_counter}")
            sections.append({"name": name, "start": float(prev_e), "end": float(last_vocal),
                             "has_lyrics": True})

    if duration - last_vocal > 3.0:
        sections.append({"name": "Outro", "start": float(last_vocal), "end": float(duration),
                         "has_lyrics": False})
    return sections


__all__ = ["detect_sections"]
