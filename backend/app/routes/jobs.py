"""API routes — job status tracking."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Job
from ..schemas import JobStatus

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatus)
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobStatus:
    """Get job status. Frontend poll ini tiap ~2 detik pas loading."""
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return JobStatus(
        id=job.id,
        youtube_id=job.youtube_id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        song_id=job.song_id,
        error=job.error,
        created_at=job.created_at,
    )
