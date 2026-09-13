"""Supabase access-token verification.

Flow: Google -> Supabase Auth -> JWT -> FastAPI (this module).

The token signature is always verified:
  * new Supabase projects -> JWKS (asymmetric, RS256/ES256/EdDSA)
  * legacy projects       -> shared JWT secret (HS256)

``USER_SERVICE_DEV_MODE=1`` additionally accepts ``Authorization: Bearer dev:<id>``
for local testing only. Never enable it in production.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jwt
from jwt import PyJWKClient

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

DEV_TOKEN_PREFIX = "dev:"
_ASYMMETRIC_ALGORITHMS = ["RS256", "ES256", "EdDSA"]
_SYMMETRIC_ALGORITHMS = ["HS256", "HS384", "HS512"]


class AuthError(Exception):
    """Raised when a bearer token cannot be trusted."""


@dataclass(frozen=True)
class Principal:
    """Identity extracted from a verified token."""

    supabase_user_id: str
    email: str | None = None
    display_name: str | None = None
    avatar_url: str | None = None


@lru_cache(maxsize=4)
def _jwks_client(jwks_url: str) -> PyJWKClient:
    return PyJWKClient(jwks_url, cache_keys=True)


def _decode(token: str, key: Any, algorithms: list[str], settings: Settings) -> dict:
    audience = settings.supabase_jwt_audience or None
    issuer = settings.issuer
    options = {
        "verify_aud": audience is not None,
        "verify_iss": issuer is not None,
    }
    return jwt.decode(
        token,
        key,
        algorithms=algorithms,
        audience=audience,
        issuer=issuer,
        options=options,
    )


def _decode_with_jwks(token: str, settings: Settings) -> dict:
    signing_key = _jwks_client(settings.jwks_url).get_signing_key_from_jwt(token)
    return _decode(token, signing_key.key, _ASYMMETRIC_ALGORITHMS, settings)


def _decode_with_secret(token: str, settings: Settings) -> dict:
    return _decode(token, settings.supabase_jwt_secret, _SYMMETRIC_ALGORITHMS, settings)


def _parse_dev_token(token: str) -> str | None:
    if not token.startswith(DEV_TOKEN_PREFIX):
        return None
    subject = token[len(DEV_TOKEN_PREFIX):].strip()
    if not subject:
        raise AuthError("dev token is missing a subject, expected 'dev:<user-id>'")
    return subject


def _claim_str(claims: dict, *names: str) -> str | None:
    for name in names:
        value = claims.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def verify_token(token: str) -> Principal:
    """Verify a bearer token and return the authenticated principal."""
    settings = get_settings()

    if settings.dev_mode:
        dev_subject = _parse_dev_token(token)
        if dev_subject is not None:
            logger.warning("DEV_MODE authentication used for subject %s", dev_subject)
            return Principal(supabase_user_id=dev_subject)

    claims: dict | None = None
    last_error: AuthError | None = None

    if settings.supabase_base_url:
        try:
            claims = _decode_with_jwks(token, settings)
        except AuthError as exc:
            last_error = exc
        except Exception as exc:  # invalid signature, expired, missing key, ...
            last_error = AuthError(f"JWKS verification failed: {exc}")
        if claims is None and not settings.supabase_jwt_secret:
            raise last_error

    if claims is None and settings.supabase_jwt_secret:
        try:
            claims = _decode_with_secret(token, settings)
        except AuthError as exc:
            last_error = exc
        except Exception as exc:
            last_error = AuthError(f"JWT secret verification failed: {exc}")

    if claims is None:
        raise last_error or AuthError(
            "no Supabase JWKS URL or JWT secret configured; token cannot be verified"
        )

    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        raise AuthError("token has no usable 'sub' claim")

    return Principal(
        supabase_user_id=subject,
        email=_claim_str(claims, "email"),
        display_name=_claim_str(claims, "name", "full_name"),
        avatar_url=_claim_str(claims, "avatar_url", "picture"),
    )
