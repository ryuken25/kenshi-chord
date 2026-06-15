"""API routes — generate endpoint (the main entry point)."""
from __future__ import annotations

import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from ..cache import find_existing_song
from ..db import db_session, get_db
from ..metadata import extract_youtube_id
from ..models import Job, Song
from ..schemas import GenerateRequest, GenerateResponse, SongDetail

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["generate"])


@router.post("/songs/generate", response_model=GenerateResponse)
def generate(
    req: GenerateRequest,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
) -> GenerateResponse:
    """Submit YouTube URL → cek cache → return cached OR enqueue pipeline job.

    Returns:
      - { cached: true, song_id, song }  kalau udah ada di cache
      - { cached: false, job_id }        kalau mulai proses baru
    """
    youtube_id = extract_youtube_id(req.youtube_url)
    if not youtube_id:
        raise HTTPException(400, "Bukan URL YouTube yang valid")

    # Cek cache by youtube_id aja dulu (no need artist/title)
    existing = db.query(Song).filter(Song.youtube_id == youtube_id).one_or_none()
    if existing:
        try:
            render = json.loads(existing.render_json)
        except json.JSONDecodeError:
            render = None
        return GenerateResponse(
            cached=True,
            song_id=existing.id,
            song=SongDetail(**render) if render else None,
        )

    # Buat job
    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        youtube_id=youtube_id,
        status="queued",
        progress=0,
        message="Antri…",
    )
    db.add(job)
    db.commit()

    # Enqueue pipeline (BackgroundTasks runs after response dikirim)
    background.add_task(_safe_run_pipeline, req.youtube_url, job_id, req.reference_lyrics)

    return GenerateResponse(cached=False, job_id=job_id)


def _safe_run_pipeline(url: str, job_id: str, reference_lyrics: Optional[str] = None) -> None:
    """Wrapper buat handle exceptions dari pipeline (jangan crash worker).

    Phase 1.1: route the URL through FASE 2's real-ML pipeline
    (`smartfix_auto.main`) so the user gets BTC + Whisper + MMS_FA-aligned
    chords and lyrics, not a FASE 1 scaffold. The FASE 1 path is still
    importable via `app.pipeline.run_pipeline` and selectable through the
    `PIPELINE_MODE` env var (default = "fase2").
    """
    import os
    use_fase2 = os.environ.get("PIPELINE_MODE", "fase2").lower() != "fase1"
    try:
        if use_fase2:
            # Imported lazily so the FastAPI app can still boot if
            # `smartfix_auto` is missing some heavy ML deps (FASE 1 mode).
            from smartfix_auto import main as fase2_main
            fase2_main(url=url, reference_lyrics=reference_lyrics, save=True, job_id=job_id)
        else:
            from ..pipeline import run_pipeline
            run_pipeline(url, job_id)
    except Exception as e:
        log.exception("Pipeline crashed for job %s", job_id)
        with db_session() as db:
            job = db.get(Job, job_id)
            if job:
                job.status = "failed"
                job.error = str(e)[:500]
                db.commit()
