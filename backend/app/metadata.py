"""yt-dlp wrapper — fetch metadata (no download)."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import yt_dlp

log = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    youtube_id: str
    title: str
    uploader: str
    artist: Optional[str] = None
    track: Optional[str] = None
    duration_sec: Optional[int] = None
    thumbnail_url: Optional[str] = None


_YT_ID_PATTERNS = [
    re.compile(r"(?:youtu\.be/)([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/(?:watch\?v=|embed/|v/))([A-Za-z0-9_-]{11})"),
    re.compile(r"^([A-Za-z0-9_-]{11})$"),  # raw ID
]


def extract_youtube_id(url: str) -> Optional[str]:
    """Extract YouTube video ID dari berbagai format URL."""
    if not url:
        return None
    s = url.strip()
    for pat in _YT_ID_PATTERNS:
        m = pat.search(s)
        if m:
            return m.group(1)
    return None


def _parse_artist_title(meta: dict) -> tuple[Optional[str], Optional[str]]:
    """Best-effort parse artist + track dari metadata YouTube.
    Priority: artist/track fields > "Artist - Title" pattern > uploader/title fallback.
    """
    artist = meta.get("artist") or meta.get("creator") or meta.get("uploader")
    track  = meta.get("track")  or meta.get("title")

    # Heuristic: kalau title berformat "Artist - Song", split
    if track and " - " in track and not meta.get("track"):
        parts = track.split(" - ", 1)
        # Heuristic: kalau uploader matches "Artist", pakai format ini
        if artist and parts[0].strip().lower() == str(artist).strip().lower():
            track = parts[1].strip()
        elif not meta.get("artist"):
            artist, track = parts[0].strip(), parts[1].strip()

    return (str(artist) if artist else None,
            str(track)  if track  else None)


def fetch_metadata(url: str) -> VideoMetadata:
    """Fetch video metadata via yt-dlp (no download).

    Tries progressively:
      1. Full extract (best info, mungkin gagal kalo format API berubah)
      2. process=False (raw info tanpa format processing)
      3. Minimal fallback (cuma YouTube ID + placeholder title)

    Raises yt_dlp.utils.DownloadError kalau semua gagal AND ID gak bisa di-parse.
    """
    youtube_id = extract_youtube_id(url) or ""
    if not youtube_id:
        raise ValueError(f"Gak bisa extract YouTube ID dari URL: {url}")

    # Attempt 1: full extract
    try:
        return _fetch_full(url, youtube_id)
    except Exception as e1:
        log.info("Full extract failed (%s), trying process=False", type(e1).__name__)

    # Attempt 2: raw info
    try:
        return _fetch_raw(url, youtube_id)
    except Exception as e2:
        log.info("Raw extract failed (%s), using minimal fallback", type(e2).__name__)

    # Attempt 3: minimal placeholder
    return VideoMetadata(
        youtube_id=youtube_id,
        title=f"YouTube {youtube_id}",
        uploader="Unknown",
        artist=None,
        track=None,
        duration_sec=None,
        thumbnail_url=f"https://i.ytimg.com/vi/{youtube_id}/hqdefault.jpg",
    )


def _fetch_full(url: str, youtube_id: str) -> VideoMetadata:
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "extract_flat": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return _build_from_info(info, youtube_id)


def _fetch_raw(url: str, youtube_id: str) -> VideoMetadata:
    """Use process=False to get raw info tanpa format processing (lebih reliable)."""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "extract_flat": False,
        "process": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return _build_from_info(info, youtube_id)


def _build_from_info(info: dict, youtube_id: str) -> VideoMetadata:
    artist, track = _parse_artist_title(info)
    return VideoMetadata(
        youtube_id=info.get("id") or youtube_id,
        title=track or info.get("title", "Untitled"),
        uploader=info.get("uploader") or info.get("channel", "Unknown"),
        artist=artist,
        track=track,
        duration_sec=int(info.get("duration")) if info.get("duration") else None,
        thumbnail_url=info.get("thumbnail"),
    )
