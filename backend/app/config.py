"""Configuration — baca dari .env atau environment variables."""
from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root = parent of backend/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Backend settings."""

    # Database
    # Default: SQLite di data/kenshi.db. Untuk MySQL production:
    #   DATABASE_URL=mysql+pymysql://user:pass@host:3306/kenshi
    database_url: str = f"sqlite:///{DATA_DIR / 'kenshi.db'}"

    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = True
    cors_origins: list[str] = ["*"]   # dev only — tighten in production

    # Audio pipeline
    audio_dir: Path = DATA_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    max_duration_sec: int = 600       # 10 menit hard cap
    sample_rate: int = 44100          # WAV target

    # Cache matching
    fuzzy_match_threshold: int = 88   # 0-100, untuk FASE 2 kalau mau pake rapidfuzz

    # Pipeline (placeholder FASE 1, real ML FASE 2)
    enable_real_ml: bool = False      # kalau True, pake demucs/whisperx (butuh GPU)

    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
