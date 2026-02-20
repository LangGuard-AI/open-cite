"""
SQLAlchemy engine singleton, session factory, and database lifecycle.

Supports SQLite (local dev) and PostgreSQL (production).
"""

import logging
import os
from typing import Optional

from sqlalchemy import create_engine, event
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.pool import StaticPool, QueuePool

from .models import Base

logger = logging.getLogger(__name__)

_engine = None
_session_factory = None


def _build_database_url() -> str:
    """Resolve the database URL from environment variables.

    Priority:
    1. ``OPENCITE_DATABASE_URL`` env var (full SQLAlchemy URL)
    2. ``OPENCITE_DB_PATH`` env var → ``sqlite:///{path}``
    3. Default ``sqlite:///./opencite.db``
    """
    url = os.getenv("OPENCITE_DATABASE_URL")
    if url:
        return url

    db_path = os.getenv("OPENCITE_DB_PATH", "./opencite.db")
    return f"sqlite:///{db_path}"


def get_engine():
    """Return the singleton SQLAlchemy engine, creating it on first call."""
    global _engine
    if _engine is not None:
        return _engine

    url = _build_database_url()
    is_sqlite = url.startswith("sqlite")

    if is_sqlite:
        _engine = create_engine(
            url,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        logger.info("SQLAlchemy engine created (SQLite): %s", url)
    else:
        _engine = create_engine(
            url,
            poolclass=QueuePool,
            pool_pre_ping=True,
            pool_size=5,
        )
        logger.info("SQLAlchemy engine created (PostgreSQL): %s", url.split("@")[-1] if "@" in url else url)

    return _engine


def get_session():
    """Return a thread-scoped session bound to the singleton engine."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        factory = sessionmaker(bind=engine)
        _session_factory = scoped_session(factory)
    return _session_factory()


def init_db():
    """Create all tables that don't already exist."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database tables created/verified")


def close_db():
    """Dispose the engine and remove the scoped session."""
    global _engine, _session_factory
    if _session_factory is not None:
        _session_factory.remove()
        _session_factory = None
    if _engine is not None:
        _engine.dispose()
        _engine = None
        logger.info("Database engine disposed")
