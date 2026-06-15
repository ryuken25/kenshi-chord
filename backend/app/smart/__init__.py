"""KenshiChord smart analysis package.

Refactored home for the MIR/NLP components that used to live in
smartfix_auto.py:

    smart.romaji       — WS-1  Japanese → display + word/syllable map (cutlet/fugashi)
    smart.beats        — WS-2  real beat/downbeat/BPM grid             (librosa)
    smart.key          — WS-3  key detection + chord spelling + capo   (librosa chroma)
    smart.align        — WS-4  per-line MMS_FA alignment at syllable level
    smart.anchor       — WS-5  chord → syllable anchoring, no lyric corruption
    smart.sections     — WS-6  structure-aware section detection
    smart.contract     — WS-7  render_json builder + validate_render()

smartfix_auto.main() orchestrates these; the package owns the logic.
The old entry signatures are preserved where they're called from outside.
"""
from __future__ import annotations

__all__ = [
    "romaji", "beats", "key", "align", "anchor", "sections", "contract",
]
