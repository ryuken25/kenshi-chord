"""KenshiChord backend — FASE 1 entrypoint."""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db import init_db
from .routes import generate as generate_routes
from .routes import jobs as jobs_routes
from .routes import songs as songs_routes

logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("kenshi")


# === App init ===
app = FastAPI(
    title="KenshiChord API",
    version="0.1.0-fase1",
    description="AI auto-chord ala Chordify, tema samurai. FASE 1: backend skeleton + cache + scaffold pipeline.",
)

# CORS — dev only. Production: tighten ke domain frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Startup ===
@app.on_event("startup")
def on_startup() -> None:
    init_db()
    log.info("DB initialized at %s", settings.database_url)
    log.info("Audio dir: %s", settings.audio_dir)


# === Health ===
@app.get("/api/health", tags=["health"])
def health() -> dict:
    return {"status": "ok", "version": app.version, "fase": "1"}


# === Mount routes ===
app.include_router(songs_routes.router)
app.include_router(jobs_routes.router)
app.include_router(generate_routes.router)


# === Run with: uvicorn app.main:app --reload ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=settings.debug)
