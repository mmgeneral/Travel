from __future__ import annotations

from collections.abc import Callable

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from user_service.models import TravelPlan, User

PLANS = "/api/v1/travel-plans"

SAMPLE_CONTENT = {
    "schema_version": 1,
    "destination": "台南",
    "days": [
        {
            "day": 1,
            "schedule": [
                {"time": "09:00", "name": "牛肉湯", "meta": {"rating": 4.5, "tags": ["早餐"]}}
            ],
        }
    ],
    "constraints": {"budget": 5000, "dietary": ["no-pork"]},
}


def _create_plan(client: TestClient, headers: dict[str, str], **overrides) -> dict:
    payload = {
        "title": "台南三天兩夜",
        "destination": "台南",
        "content": SAMPLE_CONTENT,
    }
    payload.update(overrides)
    response = client.post(PLANS, json=payload, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()


# ----------------------------------------------------------------------
# Auth
# ----------------------------------------------------------------------
def test_plans_require_authentication(client: TestClient) -> None:
    assert client.get(PLANS).status_code == 401


# ----------------------------------------------------------------------
# Create / read
# ----------------------------------------------------------------------
def test_create_plan_stores_json_content(
    client: TestClient, auth: Callable[[str], dict[str, str]]
) -> None:
    plan = _create_plan(client, auth("user-a"))

    assert plan["title"] == "台南三天兩夜"
    assert plan["destination"] == "台南"
    assert plan["content"] == SAMPLE_CONTENT


def test_create_plan_rejects_client_supplied_user_id(
    client: TestClient, auth: Callable[[str], dict[str, str]]
) -> None:
    response = client.post(
        PLANS,
        json={"title": "hack", "content": {}, "user_id": "00000000-0000-0000-0000-000000000000"},
        headers=auth("user-a"),
    )

    assert response.status_code == 422


def test_json_content_round_trips(
    client: TestClient, auth: Callable[[str], dict[str, str]]
) -> None:
    headers = auth("user-a")
    created = _create_plan(client, headers)

    fetched = client.get(f"{PLANS}/{created['id']}", headers=headers)

    assert fetched.status_code == 200
    assert fetched.json()["content"] == SAMPLE_CONTENT


def test_list_only_returns_own_plans(
    client: TestClient, auth: Callable[[str], dict[str, str]]
) -> None:
    headers_a = auth("user-a")
    headers_b = auth("user-b")
    _create_plan(client, headers_a, title="A1")
    _create_plan(client, headers_a, title="A2")
    _create_plan(client, headers_b, title="B1")

    plans_a = client.get(PLANS, headers=headers_a).json()
    plans_b = client.get(PLANS, headers=headers_b).json()

    assert sorted(p["title"] for p in plans_a) == ["A1", "A2"]
    assert [p["title"] for p in plans_b] == ["B1"]


def test_read_own_plan(
    client: TestClient, auth: Callable[[str], dict[str, str]]
) -> None:
    headers = auth("user-a")
    created = _create_plan(client, headers)

    response = client.get(f"{PLANS}/{created['id']}", headers=headers)

    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


# ----------------------------------------------------------------------
# Update / delete
# ----------------------------------------------------------------------
def test_patch_only_changes_supplied_fields(
    client: TestClient, auth: Callable[[str], dict[str, str]]
) -> None:
    headers = auth("user-a")
    created = _create_plan(client, headers)

    response = client.patch(
        f"{PLANS}/{created['id']}", json={"title": "改名了"}, headers=headers
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["title"] == "改名了"
    assert body["destination"] == "台南"
    assert body["content"] == SAMPLE_CONTENT


def test_patch_rejects_unknown_fields(
    client: TestClient, auth: Callable[[str], dict[str, str]]
) -> None:
    headers = auth("user-a")
    created = _create_plan(client, headers)

    response = client.patch(
        f"{PLANS}/{created['id']}", json={"user_id": created["user_id"]}, headers=headers
    )

    assert response.status_code == 422


def test_delete_own_plan(
    client: TestClient, auth: Callable[[str], dict[str, str]]
) -> None:
    headers = auth("user-a")
    created = _create_plan(client, headers)

    deleted = client.delete(f"{PLANS}/{created['id']}", headers=headers)
    gone = client.get(f"{PLANS}/{created['id']}", headers=headers)

    assert deleted.status_code == 204
    assert gone.status_code == 404


# ----------------------------------------------------------------------
# Cross-user isolation (404, never 403)
# ----------------------------------------------------------------------
def test_cannot_read_another_users_plan(
    client: TestClient, auth: Callable[[str], dict[str, str]]
) -> None:
    created = _create_plan(client, auth("user-a"))

    response = client.get(f"{PLANS}/{created['id']}", headers=auth("user-b"))

    assert response.status_code == 404


def test_cannot_update_another_users_plan(
    client: TestClient, auth: Callable[[str], dict[str, str]]
) -> None:
    created = _create_plan(client, auth("user-a"))

    response = client.patch(
        f"{PLANS}/{created['id']}", json={"title": "stolen"}, headers=auth("user-b")
    )

    assert response.status_code == 404


def test_cannot_delete_another_users_plan(
    client: TestClient, auth: Callable[[str], dict[str, str]]
) -> None:
    headers_a = auth("user-a")
    created = _create_plan(client, headers_a)

    response = client.delete(f"{PLANS}/{created['id']}", headers=auth("user-b"))

    assert response.status_code == 404
    assert client.get(f"{PLANS}/{created['id']}", headers=headers_a).status_code == 200


# ----------------------------------------------------------------------
# Cascade
# ----------------------------------------------------------------------
def test_deleting_user_cascades_to_plans(
    client: TestClient,
    auth: Callable[[str], dict[str, str]],
    db_session: Session,
) -> None:
    _create_plan(client, auth("user-a"), title="A1")
    _create_plan(client, auth("user-a"), title="A2")

    user = db_session.scalar(select(User).where(User.supabase_user_id == "user-a"))
    assert user is not None
    assert (
        db_session.scalar(
            select(func.count()).select_from(TravelPlan).where(TravelPlan.user_id == user.id)
        )
        == 2
    )

    db_session.delete(user)
    db_session.commit()

    assert db_session.scalar(select(func.count()).select_from(TravelPlan)) == 0
