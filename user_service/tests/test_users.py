from __future__ import annotations

from collections.abc import Callable

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from user_service.models import User

ME = "/api/v1/users/me"


def test_me_requires_authentication(client: TestClient) -> None:
    response = client.get(ME)

    assert response.status_code == 401


def test_invalid_token_is_rejected(client: TestClient) -> None:
    response = client.get(ME, headers={"Authorization": "Bearer not-a-real-jwt"})

    assert response.status_code == 401


def test_first_login_creates_user(
    client: TestClient, auth: Callable[[str], dict[str, str]]
) -> None:
    response = client.get(ME, headers=auth("user-a"))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["supabase_user_id"] == "user-a"
    assert body["id"]


def test_second_login_does_not_create_another_user(
    client: TestClient,
    auth: Callable[[str], dict[str, str]],
    db_session: Session,
) -> None:
    first = client.get(ME, headers=auth("user-a"))
    second = client.get(ME, headers=auth("user-a"))

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["id"] == second.json()["id"]

    count = db_session.scalar(select(func.count()).select_from(User))
    assert count == 1


def test_users_are_isolated(
    client: TestClient, auth: Callable[[str], dict[str, str]]
) -> None:
    user_a = client.get(ME, headers=auth("user-a")).json()
    user_b = client.get(ME, headers=auth("user-b")).json()

    assert user_a["id"] != user_b["id"]


def test_update_own_profile(
    client: TestClient, auth: Callable[[str], dict[str, str]]
) -> None:
    headers = auth("user-a")
    client.get(ME, headers=headers)

    response = client.patch(
        ME,
        json={"display_name": "Alice", "locale": "zh-TW", "avatar_url": "https://x/a.png"},
        headers=headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["display_name"] == "Alice"
    assert body["locale"] == "zh-TW"
    assert body["avatar_url"] == "https://x/a.png"


def test_update_rejects_unknown_fields(
    client: TestClient, auth: Callable[[str], dict[str, str]]
) -> None:
    headers = auth("user-a")
    client.get(ME, headers=headers)

    response = client.patch(ME, json={"is_admin": True}, headers=headers)

    assert response.status_code == 422
