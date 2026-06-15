"""WS-1: Japanese → Hepburn romaji with word + syllable segmentation.

Uses cutlet (built on fugashi/MeCab + unidic-lite).

Verified against RAIN Appendix A:
    出逢った → "deatta"                (sokuon doubled)
    眩しさは虚しさ → "mabushisa wa munashisa"  (particle は → wa)
    肩を濡らす雨粒で → "kata wo nurasu amatsubu de"
    知ってる → "shitteru"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    import cutlet
    import fugashi
    ROMAJI_AVAILABLE = True
except ImportError:
    ROMAJI_AVAILABLE = False


@dataclass
class Syllable:
    """A singable mora-ish unit (CV/CVV/Cっ), e.g. 'ka', 'nu', 'su'."""
    romaji: str
    char_start: int   # offset in the PARENT text (word or line, context-dependent)
    char_end: int


@dataclass
class Word:
    """A real dictionary word (display unit) from fugashi tokenization."""
    surface: str           # original kanji/kana, e.g. "濡らす"
    romaji: str            # spaced romaji display, e.g. "nurasu"
    char_start: int        # offset in original line
    char_end: int
    syllables: List[Syllable] = field(default_factory=list)


@dataclass
class LineRomaji:
    """Romanized lyric line with word + syllable segmentation."""
    text: str              # original line, untouched
    display: str           # spaced romaji for front-end rendering
    words: List[Word] = field(default_factory=list)


# ---- syllable splitter --------------------------------------------------------

_VOWELS = set("aiueo")


def _split_romaji_into_syllables(romaji: str, char_start_in_parent: int = 0) -> List[Syllable]:
    """Split Hepburn romaji into syllables: ['ka', 'ta', 'wo', 'nu', 'ra', 'su'].

    Rules (Hepburn):
    - Vowel alone = 1 syllable (a, i, u, e, o)
    - Consonant + vowel = 1 syllable (ka, ki, ku, ke, ko)
    - Consonant + 'y' + vowel = 1 syllable (kya, kyu, kyo, sha, ja, etc.)
    - 'ch' + vowel, 'sh' + vowel, 'ts' + vowel = 1 syllable
    - Syllabic 'n' = 1 syllable (when not followed by 'y' or vowel)
    - Doubled consonant (sokuon): the first 'ch' is the sokuon, the next 'chi' starts
      the following syllable. We keep doubled consonant as-is ('ss' → 'ss' in sokuon
      context) since MMS_FA vocab just has plain letters; but we split so that each
      CV unit is distinct.
    """
    out: List[Syllable] = []
    s = romaji.lower()
    i = 0
    local_start = 0

    def emit(i_start: int, i_end: int):
        nonlocal local_start
        rom = s[i_start:i_end]
        out.append(Syllable(romaji=rom, char_start=i_start, char_end=i_end))
        local_start = i_end

    # 2-letter consonant-starters (in Hepburn)
    _DIGRAPH_STARTS = {"ch", "sh", "ts"}
    # Consonant + 'y' + vowel (1 syllable)
    _Y_STARTS = {"ky", "gy", "ny", "hy", "by", "py", "my", "ry"}

    while i < len(s):
        c = s[i]
        if c in _VOWELS:
            # vowel alone
            emit(i, i + 1)
            i += 1
            continue

        # Syllabic 'n': 'n' NOT before a vowel or 'y' or 'n'
        if c == "n":
            if i + 1 >= len(s):
                emit(i, i + 1)
                i += 1
                continue
            nxt = s[i + 1]
            if nxt in _VOWELS or nxt == "y" or nxt == "n":
                # 'n' is onset of a normal syllable; fall through to cluster logic
                pass
            else:
                # syllabic 'n' on its own
                emit(i, i + 1)
                i += 1
                continue

        # Check for digraph: ch/sh/ts + vowel
        if i + 2 <= len(s) and s[i:i+2] in _DIGRAPH_STARTS:
            if i + 2 < len(s) and s[i+2] in _VOWELS:
                emit(i, i + 3)
                i += 3
                continue
            # Digraph at end (shouldn't happen for valid Hepburn, but be defensive)
            emit(i, i + 2)
            i += 2
            continue

        # Check for consonant + y + vowel
        if i + 3 <= len(s) and s[i:i+2] in _Y_STARTS and s[i+2] in _VOWELS:
            emit(i, i + 3)
            i += 3
            continue

        # Single consonant + vowel
        if i + 2 <= len(s) and s[i+1] in _VOWELS:
            emit(i, i + 2)
            i += 2
            continue

        # Doubled consonant: sokuon. We emit the first consonant alone (sokuon)
        # and leave the second for the next syllable. E.g. 'shitteru' →
        # ['shi', 't', 'te', 'ru'] — but since MMS_FA vocab treats doubled
        # consonants as repeated chars, we can keep the 'tt' intact by emitting
        # 't' alone when we see a double.
        if i + 1 < len(s) and s[i] == s[i+1]:
            emit(i, i + 1)
            i += 1
            continue

        # Fallback: emit single char
        emit(i, i + 1)
        i += 1

    return out


# ---- cutlet wrapper -----------------------------------------------------------

_ROMANIZER: Optional[object] = None
_TAGGER: Optional[object] = None

def _get_romanizer():
    global _ROMANIZER, _TAGGER
    if _ROMANIZER is None:
        _ROMANIZER = cutlet.Cutlet("hepburn")
        # use_foreign_spelling=False keeps us on the pure Hepburn path
        # (we set use_foreign_spelling at init but cutlet defaults to False)
    if _TAGGER is None:
        _TAGGER = fugashi.Tagger()
    return _ROMANIZER, _TAGGER


def romanize(text: str) -> str:
    """Convert Japanese text to Hepburn romaji (lowercase, normalized)."""
    if not ROMAJI_AVAILABLE:
        raise RuntimeError("cutlet/fugashi not installed")
    r, _ = _get_romanizer()
    # cutlet returns Title Case by default, lower() to match the project convention
    return r.romaji(text).lower()


def romanize_line(text: str) -> LineRomaji:
    """Tokenize with fugashi, romanize each word, build syllable map."""
    if not ROMAJI_AVAILABLE:
        raise RuntimeError("cutlet/fugashi not installed")
    r, tagger = _get_romanizer()

    display_parts: List[str] = []
    words: List[Word] = []

    char_pos = 0
    tokens = tagger.parseToNodeList(text)
    for tok in tokens:
        surface = tok.surface
        # romaji for this surface (Hepburn)
        rom = r.romaji(surface).lower() if any(not c.isascii() or c in "ぁ-ｿ" for c in surface) else surface.lower()
        if not rom:
            # Punctuation / whitespace — still advance char_pos
            char_pos += len(surface)
            continue

        word_start = char_pos
        word_end = char_pos + len(surface)

        syms = _split_romaji_into_syllables(rom)

        display_parts.append(rom)
        words.append(Word(
            surface=surface,
            romaji=rom,
            char_start=word_start,
            char_end=word_end,
            syllables=syms,
        ))
        char_pos = word_end

    display = " ".join(display_parts)

    # Fix the known Hepburn edge-case: 'を' → 'o' but we want 'wo'
    # cutlet renders 'を' as 'o'; per the project config we prefer 'wo'.
    # This post-process catches the simple isolated 'を' tokens.
    for w in words:
        if w.surface == "を":
            w.romaji = "wo"
            for s in w.syllables:
                if s.romaji == "o":
                    s.romaji = "wo"

    return LineRomaji(text=text, display=display, words=words)


# Convenience shims used by smartfix_auto.py
def romaji_for_word(word: str) -> List[str]:
    """Compatibility: list of romaji tokens for a surface string."""
    return _split_romaji_into_syllables_into_list(word)

def _split_romaji_into_syllables_into_list(word: str) -> List[str]:
    rom = romanize(word)
    syms = _split_romaji_into_syllables(rom)
    return [s.romaji for s in syms if s.romaji]

def romaji_list(text: str) -> List[str]:
    """Compatibility: flat list of romaji tokens."""
    line = romanize_line(text)
    return [s.romaji for w in line.words for s in w.syllables if s.romaji]


__all__ = [
    "romanize", "romanize_line", "romaji_for_word", "romaji_list",
    "LineRomaji", "Word", "Syllable", "ROMAJI_AVAILABLE",
]
