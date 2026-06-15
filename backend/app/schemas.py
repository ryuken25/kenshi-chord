"""Pydantic schemas — request/response shapes."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# === Request ===

class GenerateRequest(BaseModel):
    youtube_url: str = Field(..., min_length=1)
    # Optional: caller can paste known-correct lyrics so FASE 2 can align those
    # instead of whisper-transcribing (more accurate timing on tricky audio).
    # When omitted, the pipeline falls back to faster-whisper transcription.
    reference_lyrics: Optional[str] = None

    @field_validator("youtube_url")
    @classmethod
    def _trim(cls, v: str) -> str:
        return v.strip()


# === Response: Song ===

class SongMeta(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    youtube_id: str
    artist: str
    title: str
    duration_sec: Optional[int] = None
    bpm: Optional[float] = None
    key: Optional[str] = None
    capo: int = 0
    time_sig: str = "4/4"
    language: Optional[str] = None


class SongListItem(BaseModel):
    id: int
    youtube_id: str
    artist: str
    title: str
    duration_sec: Optional[int] = None
    bpm: Optional[float] = None
    key: Optional[str] = None
    language: Optional[str] = None
    created_at: datetime


class SongDetail(BaseModel):
    """Full render_json — sama dengan APPENDIX A di megaprompt."""
    meta: SongMeta
    beats: list[float] = []
    downbeats: list[float] = []
    sections: list[dict[str, Any]] = []
    bars: list[dict[str, Any]] = []
    lines: list[dict[str, Any]] = []


# === Response: Job ===

class JobStatus(BaseModel):
    id: str
    youtube_id: Optional[str] = None
    status: str
    progress: int
    message: Optional[str] = None
    song_id: Optional[int] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None


# === Response: Generate (either cached or queued) ===

class GenerateResponse(BaseModel):
    cached: bool
    song_id: Optional[int] = None
    job_id: Optional[str] = None
    song: Optional[SongDetail] = None
