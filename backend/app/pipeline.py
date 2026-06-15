"""Pipeline — orchestrate audio download + ML (placeholder FASE 1, real FASE 2)."""
from __future__ import annotations

import json
import logging
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from .cache import find_existing_song, normalize_artist, normalize_title
from .config import settings
from .db import db_session
from .metadata import VideoMetadata, fetch_metadata
from .models import Job, Song

log = logging.getLogger(__name__)


# === Public entrypoint ===

def run_pipeline(youtube_url: str, job_id: str) -> None:
    """Run full pipeline untuk 1 video. Update job status di setiap step.

    Pipeline FASE 1 (placeholder):
      1. Fetch metadata (yt-dlp, no download)
      2. Cache check
      3. Download audio (best audio → WAV 44.1k via ffmpeg)
      4. Detect language (heuristic from title)
      5. Estimate BPM (librosa stub or skip)
      6. Build render_json scaffold (no real chord/lyric detection — FASE 2)
      7. Save to DB

    FASE 2 replaces step 6 with: demucs → beat → autochord → whisperx → alignment.
    """
    _update_job(job_id, status="downloading", progress=5,  message="Nge-tap metadata…")

    try:
        meta = fetch_metadata(youtube_url)
    except Exception as e:
        log.exception("Metadata fetch failed")
        _update_job(job_id, status="failed", progress=0, error=f"Metadata: {e}")
        return

    _update_job(job_id, status="downloading", progress=15, message="Nyedot audio…")

    with db_session() as db:
        # Cache check by youtube_id
        existing = find_existing_song(db, meta.youtube_id, meta.artist or "", meta.title or "")
        if existing:
            log.info("Cache hit: song id=%s", existing.id)
            _update_job(job_id, status="done", progress=100, message="Cache hit ✓",
                        song_id=existing.id)
            return

    # Download audio (skip kalau duration > max)
    if meta.duration_sec and meta.duration_sec > settings.max_duration_sec:
        _update_job(job_id, status="failed", error=f"Durasi {meta.duration_sec}s > max {settings.max_duration_sec}s")
        return

    audio_path = _download_audio(meta.youtube_id, settings.audio_dir)
    _update_job(job_id, status="separating", progress=40, message="Nge-pisah vokal…")

    # === FASE 2 PLACEHOLDER ===
    # Yang sebenarnya di sini: demucs → beat → autochord → whisperx → alignment.
    # FASE 1 cuma bikin scaffold render_json + estimate BPM dari audio (kalau librosa ada).
    time.sleep(0.5)  # simulasi kerja ML

    bpm = _estimate_bpm(audio_path)
    language = _guess_language(meta.title or "", meta.artist or "")

    _update_job(job_id, status="detecting", progress=65, message="Nebak chord…")
    time.sleep(0.3)

    _update_job(job_id, status="transcribing", progress=82, message="Nyalin lirik…")
    time.sleep(0.3)

    _update_job(job_id, status="aligning", progress=92, message="Nge-join chord ↔ kata…")
    time.sleep(0.2)

    render = _build_scaffold_render(meta, bpm, language)

    # Persist
    with db_session() as db:
        # Re-check cache (race condition: mungkin ada job lain yang baru selesai)
        existing = find_existing_song(db, meta.youtube_id, meta.artist or "", meta.title or "")
        if existing:
            _update_job(job_id, status="done", progress=100, message="Cache hit (race) ✓",
                        song_id=existing.id)
            return

        song = Song(
            youtube_id=meta.youtube_id,
            artist=meta.artist or meta.uploader,
            title=meta.title,
            artist_norm=normalize_artist(meta.artist or meta.uploader),
            title_norm=normalize_title(meta.title),
            album=None,
            duration_sec=meta.duration_sec,
            bpm=bpm,
            music_key="C major",        # FASE 2: actual detection
            capo=0,
            time_sig="4/4",
            thumbnail_url=meta.thumbnail_url,
            language=language,
            status="ready",
            source="ai",
            render_json=json.dumps(render, ensure_ascii=False),
            audio_path=str(audio_path) if audio_path else None,
        )
        db.add(song)
        db.flush()
        song_id = song.id
        db.commit()

    _update_job(job_id, status="done", progress=100, message="Disimpen ke lemari ✓",
                song_id=song_id)


# === Helpers ===

def _update_job(job_id: str, **fields) -> None:
    """Update job fields. Sets updated_at automatically."""
    with db_session() as db:
        job = db.get(Job, job_id)
        if not job:
            log.warning("Job %s not found", job_id)
            return
        for k, v in fields.items():
            setattr(job, k, v)
        db.commit()


def _download_audio(youtube_id: str, out_dir: Path) -> Optional[Path]:
    """Download best audio + convert to WAV 44.1kHz mono via ffmpeg.
    Returns the WAV path, or None kalau gagal / ffmpeg not installed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = out_dir / f"{youtube_id}.%(ext)s"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": str(out_template),
        "noplaylist": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "0",  # lossless
        }],
    }
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={youtube_id}"])
        wav_path = out_dir / f"{youtube_id}.wav"
        if wav_path.exists():
            # Resample to 44.1k mono via ffmpeg kalau belum
            _ensure_wav_44k(wav_path)
            return wav_path
    except Exception as e:
        log.warning("Audio download failed for %s: %s", youtube_id, e)
    return None


def _ensure_wav_44k(wav_path: Path) -> None:
    """Best-effort: pakai ffmpeg untuk ensure WAV PCM 44.1k mono. Skip kalau ffmpeg gak ada."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_path), "-ar", "44100", "-ac", "1",
             "-f", "wav", str(wav_path) + ".tmp"],
            check=True, capture_output=True, timeout=120,
        )
        Path(str(wav_path) + ".tmp").replace(wav_path)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        log.info("ffmpeg resample skipped: %s", e)


def _estimate_bpm(audio_path: Optional[Path]) -> Optional[float]:
    """Stub: kalau librosa ada, compute BPM. Else return None.
    FASE 2 akan pakai allin1 atau madmom."""
    if audio_path is None:
        return None
    try:
        import librosa  # noqa
        y, sr = librosa.load(str(audio_path), sr=None, mono=True, duration=60)
        bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(bpm) if bpm else None
    except Exception as e:
        log.info("BPM estimation skipped: %s", e)
        return None


def _guess_language(title: str, artist: str) -> str:
    """Heuristic detect language dari artist/title."""
    text = (title + " " + artist).lower()
    # Japanese: ada hiragana/katakana
    if any("぀" <= c <= "ゟ" or "゠" <= c <= "ヿ" for c in text):
        return "ja"
    # Indonesian: kata-kata umum
    id_words = ["yang", "aku", "kamu", "cinta", "hati", "indonesia", "jakarta", "saja", "akan"]
    if any(w in text.split() for w in id_words):
        return "id"
    # Default English
    return "en"


def _build_scaffold_render(meta: VideoMetadata, bpm: Optional[float], language: str) -> dict:
    """Build minimal render_json scaffold (FASE 1 placeholder).

    FASE 2 akan replace ini dengan output alignment engine yang sebenarnya.
    Untuk FASE 1, kita bikin struktur yang valid tapi isinya 'placeholder':
    - 1 section marker (Verse 1, durasi 0 → duration)
    - 1 bar dengan 1 chord placeholder (artist pertama letter)
    - 0 lines (no lyrics)
    """
    duration = meta.duration_sec or 60
    first_chord = "C"
    return {
        "meta": {
            "youtube_id":   meta.youtube_id,
            "artist":       meta.artist or meta.uploader,
            "title":        meta.title,
            "duration_sec": duration,
            "bpm":          bpm,
            "key":          "C major",
            "capo":         0,
            "time_sig":     "4/4",
            "language":     language,
        },
        "beats":     [],
        "downbeats": [],
        "sections":  [
            {"name": "Verse 1", "start": 0.0, "end": float(duration), "has_lyrics": False},
        ],
        "bars":      [
            {"index": 0, "start": 0.0, "end": float(duration),
             "chords": [{"chord": first_chord, "start": 0.0, "end": float(duration)}]},
        ],
        "lines":     [],
    }
