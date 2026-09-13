"""FastAPI dependencies: bearer token -> principal -> local user row."""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from . import crud
from .auth import AuthError, Principal, verify_token
from .database import get_db
from .models import User

bearer_scheme = HTTPBearer(auto_error=False)

_UNAUTHORIZED_HEADERS = {"WWW-Authenticate": "Bearer"}


def get_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> Principal:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers=_UNAUTHORIZED_HEADERS,
        )
    try:
        return verify_token(credentials.credentials)
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers=_UNAUTHORIZED_HEADERS,
        ) from exc


def get_current_user(
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_db),
) -> User:
    """Resolve the local user, creating it on first login."""
    return crud.get_or_create_user(
        db,
        supabase_user_id=principal.supabase_user_id,
        email=principal.email,
        display_name=principal.display_name,
        avatar_url=principal.avatar_url,
    )
