"""SQLAlchemy engine + session + dependency."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    """Base class untuk semua model."""
    pass


# SQLite butuh check_same_thread=False supaya bisa dipake sama FastAPI BackgroundTasks
_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(
    settings.database_url,
    connect_args=_connect_args,
    echo=settings.debug,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: yield a DB session, close it after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session() -> Iterator[Session]:
    """Context manager untuk BackgroundTasks / script (bukan FastAPI endpoint)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables kalau belum ada. Untuk dev — production pakai Alembic."""
    # Import model di sini supaya registry Base.metadata ke-populate
    from . import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
