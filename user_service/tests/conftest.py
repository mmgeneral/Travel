"""Test fixtures: SQLite in-memory DB + DEV_MODE tokens.

No request ever reaches Google or Supabase.
"""

from __future__ import annotations

import os

# Must be set before any ``user_service`` module (and thus Settings) is imported.
os.environ["USER_SERVICE_DATABASE_URL"] = "sqlite+pysqlite:///:memory:"
os.environ["USER_SERVICE_DEV_MODE"] = "1"
os.environ["USER_SERVICE_SUPABASE_URL"] = ""
os.environ["USER_SERVICE_SUPABASE_JWT_SECRET"] = "test-secret"
os.environ["USER_SERVICE_SUPABASE_JWT_AUDIENCE"] = "authenticated"
os.environ["USER_SERVICE_CORS_ORIGINS"] = ""

from collections.abc import Callable, Iterator  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from user_service.database import Base, SessionLocal, engine  # noqa: E402
from user_service.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_schema() -> Iterator[None]:
    """Give every test an empty database."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def db_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def auth() -> Callable[[str], dict[str, str]]:
    """Build a DEV_MODE Authorization header for a given subject id."""

    def _headers(subject: str) -> dict[str, str]:
        return {"Authorization": f"Bearer dev:{subject}"}

    return _headers
