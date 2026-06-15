"""SQLAlchemy models — sesuai APPENDIX schema di megaprompt."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, Index
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


# === Helper: timestamp default ===
def _now() -> datetime:
    return datetime.utcnow()


class Song(Base):
    """1 row per lagu unik (cache key)."""
    __tablename__ = "songs"

    id:            Mapped[int]       = mapped_column(Integer, primary_key=True, autoincrement=True)
    youtube_id:    Mapped[str]       = mapped_column(String(20), nullable=False, unique=True)
    artist:        Mapped[str]       = mapped_column(String(255), nullable=False)
    title:         Mapped[str]       = mapped_column(String(255), nullable=False)
    artist_norm:   Mapped[str]       = mapped_column(String(255), nullable=False, index=True)
    title_norm:    Mapped[str]       = mapped_column(String(255), nullable=False, index=True)
    album:         Mapped[Optional[str]] = mapped_column(String(255))
    duration_sec:  Mapped[Optional[int]] = mapped_column(Integer)
    bpm:           Mapped[Optional[float]] = mapped_column(Float)
    music_key:     Mapped[Optional[str]] = mapped_column(String(20))
    capo:          Mapped[int]       = mapped_column(Integer, default=0)
    time_sig:      Mapped[str]       = mapped_column(String(8), default="4/4")
    thumbnail_url: Mapped[Optional[str]] = mapped_column(Text)
    language:      Mapped[Optional[str]] = mapped_column(String(10))
    status:        Mapped[str]       = mapped_column(String(20), default="ready")  # queued|processing|ready|failed
    source:        Mapped[str]       = mapped_column(String(20), default="ai")      # ai|manual
    render_json:   Mapped[str]       = mapped_column(Text, nullable=False)          # JSON string (APPENDIX A)
    audio_path:    Mapped[Optional[str]] = mapped_column(String(512))
    created_at:    Mapped[datetime]  = mapped_column(DateTime, default=_now)
    updated_at:    Mapped[datetime]  = mapped_column(DateTime, default=_now, onupdate=_now)

    chords:        Mapped[list["Chord"]]      = relationship(back_populates="song", cascade="all, delete-orphan")
    lyric_lines:   Mapped[list["LyricLine"]]  = relationship(back_populates="song", cascade="all, delete-orphan", order_by="LyricLine.line_index")
    jobs:          Mapped[list["Job"]]        = relationship(back_populates="song")

    __table_args__ = (
        UniqueConstraint("artist_norm", "title_norm", name="idx_artist_title"),
    )


class Chord(Base):
    """Opsional — query granular chord. render_json tetap disimpan di Song untuk fast load."""
    __tablename__ = "chords"

    id:          Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    song_id:     Mapped[int]   = mapped_column(ForeignKey("songs.id", ondelete="CASCADE"), nullable=False, index=True)
    chord:       Mapped[str]   = mapped_column(String(20), nullable=False)
    start_sec:   Mapped[float] = mapped_column(Float, nullable=False)
    end_sec:     Mapped[float] = mapped_column(Float, nullable=False)
    bar_number:  Mapped[Optional[int]] = mapped_column(Integer)
    beat_pos:    Mapped[Optional[int]] = mapped_column(Integer)
    confidence:  Mapped[Optional[float]] = mapped_column(Float)

    song: Mapped["Song"] = relationship(back_populates="chords")


class LyricLine(Base):
    __tablename__ = "lyric_lines"

    id:          Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    song_id:     Mapped[int]   = mapped_column(ForeignKey("songs.id", ondelete="CASCADE"), nullable=False, index=True)
    line_index:  Mapped[int]   = mapped_column(Integer, nullable=False)
    text:        Mapped[str]   = mapped_column(Text, nullable=False)
    start_sec:   Mapped[Optional[float]] = mapped_column(Float)
    end_sec:     Mapped[Optional[float]] = mapped_column(Float)

    song:   Mapped["Song"]            = relationship(back_populates="lyric_lines")
    words:  Mapped[list["LyricWord"]] = relationship(back_populates="line", cascade="all, delete-orphan", order_by="LyricWord.id")


class LyricWord(Base):
    __tablename__ = "lyric_words"

    id:        Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    line_id:   Mapped[int]   = mapped_column(ForeignKey("lyric_lines.id", ondelete="CASCADE"), nullable=False, index=True)
    word:      Mapped[str]   = mapped_column(String(255), nullable=False)
    start_sec: Mapped[Optional[float]] = mapped_column(Float)
    end_sec:   Mapped[Optional[float]] = mapped_column(Float)

    line: Mapped["LyricLine"] = relationship(back_populates="words")


class Job(Base):
    """Tracking proses async. 1 row per generate request."""
    __tablename__ = "jobs"

    id:          Mapped[str]   = mapped_column(String(36), primary_key=True)  # uuid
    youtube_id:  Mapped[Optional[str]] = mapped_column(String(20), index=True)
    status:      Mapped[str]   = mapped_column(String(20), default="queued", index=True)
    # queued | downloading | separating | detecting | transcribing | aligning | done | failed
    progress:    Mapped[int]   = mapped_column(Integer, default=0)            # 0..100
    message:     Mapped[Optional[str]] = mapped_column(Text)
    song_id:     Mapped[Optional[int]] = mapped_column(ForeignKey("songs.id", ondelete="SET NULL"))
    error:       Mapped[Optional[str]] = mapped_column(Text)
    created_at:  Mapped[datetime]      = mapped_column(DateTime, default=_now)
    updated_at:  Mapped[datetime]      = mapped_column(DateTime, default=_now, onupdate=_now)

    song: Mapped[Optional["Song"]] = relationship(back_populates="jobs")
