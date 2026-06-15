"""Cache logic: normalize artist/title, lookup by youtube_id atau (artist, title)."""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .models import Song


# === Normalization helpers ===

# Pattern yang sering nempel di judul YouTube — strip dulu
_TITLE_SUFFIXES = [
    r"\(official\s+music\s+video\)",
    r"\(official\s+video\)",
    r"\(music\s+video\)",
    r"\(official\s+audio\)",
    r"\(lyric\s+video\)",
    r"\(lyrics\)",
    r"\[official\s+music\s+video\]",
    r"\[official\s+video\]",
    r"\[music\s+video\]",
    r"\[mv\]",
    r"\[lyrics?\]",
    r"\[4k\]",
    r"\[hd\]",
]
_FEAT_PATTERN = re.compile(r"\s*(feat\.?|ft\.?|featuring)\s+[^\(\[]*", re.IGNORECASE)
_NONALNUM = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def _strip_accents(s: str) -> str:
    """Remove accents: 'Café' → 'Cafe'."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_artist(artist: str) -> str:
    """Normalize artist name for cache matching."""
    if not artist:
        return ""
    s = artist.strip().lower()
    s = _strip_accents(s)
    s = _NONALNUM.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s


def normalize_title(title: str) -> str:
    """Normalize title: strip common YouTube suffixes, feat., accents, punctuation."""
    if not title:
        return ""
    s = title.strip()
    # Strip "(Official Video)" dll — case insensitive
    for pat in _TITLE_SUFFIXES:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    # Strip "feat. X" di luar parens
    s = _FEAT_PATTERN.sub("", s)
    s = s.lower()
    s = _strip_accents(s)
    s = _NONALNUM.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s


# === Lookup ===

def find_song_by_youtube_id(db: Session, youtube_id: str) -> Optional[Song]:
    return db.scalar(select(Song).where(Song.youtube_id == youtube_id))


def find_song_by_metadata(db: Session, artist: str, title: str) -> Optional[Song]:
    """Exact match by (artist_norm, title_norm)."""
    a, t = normalize_artist(artist), normalize_title(title)
    if not a or not t:
        return None
    return db.scalar(
        select(Song)
        .where(Song.artist_norm == a)
        .where(Song.title_norm == t)
    )


def find_existing_song(
    db: Session, youtube_id: str, artist: str = "", title: str = ""
) -> Optional[Song]:
    """Cek cache: youtube_id dulu, lalu (artist, title)."""
    s = find_song_by_youtube_id(db, youtube_id)
    if s:
        return s
    if artist and title:
        return find_song_by_metadata(db, artist, title)
    return None
