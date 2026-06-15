"""API routes — songs library."""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Song
from ..schemas import SongDetail, SongListItem

router = APIRouter(prefix="/api/songs", tags=["songs"])


@router.get("", response_model=list[SongListItem])
def list_songs(
    search: Optional[str] = Query(None, description="Cari di title/artist (case-insensitive)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[SongListItem]:
    """Library listing — search by title/artist."""
    stmt = select(Song).where(Song.status == "ready").order_by(Song.created_at.desc())
    if search:
        like = f"%{search.lower()}%"
        stmt = stmt.where(
            or_(
                Song.title.ilike(like),
                Song.artist.ilike(like),
            )
        )
    stmt = stmt.offset(offset).limit(limit)
    songs = db.scalars(stmt).all()
    return [
        SongListItem(
            id=s.id,
            youtube_id=s.youtube_id,
            artist=s.artist,
            title=s.title,
            duration_sec=s.duration_sec,
            bpm=s.bpm,
            key=s.music_key,
            language=s.language,
            created_at=s.created_at,
        )
        for s in songs
    ]


@router.get("/{song_id}", response_model=SongDetail)
def get_song(song_id: int, db: Session = Depends(get_db)) -> SongDetail:
    """Full render_json untuk song view.

    Defensive: coerce recoverable mismatches (e.g. float duration_sec,
    missing artist/title) instead of raising 500. Only truly corrupt
    JSON or missing rows should fail.
    """
    song = db.get(Song, song_id)
    if not song:
        raise HTTPException(404, f"Song {song_id} not found")
    try:
        render = json.loads(song.render_json)
    except json.JSONDecodeError:
        raise HTTPException(500, "render_json corrupt")

    # Backfill from row columns so render_json and the Song row never disagree
    # (Bug 0.2 makes them agree at write time, but old rows may still mismatch).
    render.setdefault("meta", {})
    meta = render["meta"]
    meta.setdefault("youtube_id", song.youtube_id)
    meta.setdefault("artist", song.artist or "Unknown")
    meta.setdefault("title", song.title or song.youtube_id)
    if song.duration_sec is not None:
        meta["duration_sec"] = int(song.duration_sec)
    else:
        meta.setdefault("duration_sec", None)
    if song.bpm is not None:
        # Round to int at read time so the client never sees 165.44117647058823
        meta["bpm"] = int(round(float(song.bpm)))
    else:
        meta.setdefault("bpm", None)
    meta.setdefault("key", song.music_key or "C major")
    meta.setdefault("capo", song.capo)
    meta.setdefault("time_sig", song.time_sig or "4/4")
    meta.setdefault("language", song.language)

    # Belt-and-braces: if duration_sec is still a float somewhere in the JSON,
    # coerce it to int. Prevents Pydantic from 500-ing on a recoverable row.
    if isinstance(meta.get("duration_sec"), float):
        meta["duration_sec"] = int(meta["duration_sec"])

    return SongDetail(**render)


@router.get("/by-youtube/{youtube_id}", response_model=SongDetail)
def get_song_by_youtube(youtube_id: str, db: Session = Depends(get_db)) -> SongDetail:
    """Lookup by youtube_id (untuk frontend yang punya ID aja)."""
    song = db.scalar(select(Song).where(Song.youtube_id == youtube_id))
    if not song:
        raise HTTPException(404, f"Song with youtube_id={youtube_id} not found")
    try:
        render = json.loads(song.render_json)
    except json.JSONDecodeError:
        raise HTTPException(500, "render_json corrupt")
    return SongDetail(**render)


@router.delete("/{song_id}", status_code=204, response_class=Response)
def delete_song(song_id: int, db: Session = Depends(get_db)):
    """Admin: hapus song dari cache (cascade ke chords/lyrics/jobs)."""
    song = db.get(Song, song_id)
    if not song:
        raise HTTPException(404, f"Song {song_id} not found")
    db.delete(song)
    db.commit()
    return Response(status_code=204)
