"""
Duffel idempotency probe:
1) Search offers
2) Hold one order
3) Cancel the same held order twice using Duffel's cancellation flow:
   - POST /air/order_cancellations
   - POST /air/order_cancellations/{id}/actions/confirm
4) Print status code + Duffel-Request-ID for each cancellation attempt
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

import requests


DUFFEL_BASE = os.getenv("DUFFEL_BASE_URL", "https://api.duffel.com")
DUFFEL_TOKEN = os.getenv("DUFFEL_ACCESS_TOKEN")
DUFFEL_VERSION = "v2"


class Outcome(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    RETRYABLE = "RETRYABLE"


@dataclass
class ACLResult:
    outcome: Outcome
    semantic_status: str
    original_status: int
    request_id: str
    note: str = ""


class ResultConverter:
    """Converts raw Duffel HTTP responses into ACL-level semantic outcomes."""

    @staticmethod
    def convert(resp: requests.Response) -> ACLResult:
        status = resp.status_code
        req_id = request_id_from_response(resp)
        code = ""
        try:
            errors = resp.json().get("errors", [])
            if errors:
                code = errors[0].get("code", "")
        except Exception:
            code = ""

        if 200 <= status < 300:
            return ACLResult(
                outcome=Outcome.SUCCESS,
                semantic_status="200 (SUCCESS)",
                original_status=status,
                request_id=req_id,
            )
        if status == 422 and code == "already_cancelled":
            return ACLResult(
                outcome=Outcome.SUCCESS,
                semantic_status="200 (SUCCESS)",
                original_status=422,
                request_id=req_id,
                note="already_cancelled translated to SUCCESS",
            )
        if status >= 500:
            return ACLResult(
                outcome=Outcome.RETRYABLE,
                semantic_status="503 (RETRYABLE)",
                original_status=status,
                request_id=req_id,
            )
        return ACLResult(
            outcome=Outcome.FAILED,
            semantic_status=f"{status} (FAILED)",
            original_status=status,
            request_id=req_id,
        )


def hr(label: str) -> None:
    print(f"\n=== {label} ===")


def headers() -> dict[str, str]:
    if not DUFFEL_TOKEN:
        raise RuntimeError("Missing DUFFEL_ACCESS_TOKEN")
    return {
        "Authorization": f"Bearer {DUFFEL_TOKEN}",
        "Duffel-Version": DUFFEL_VERSION,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def print_resp(tag: str, resp: requests.Response) -> None:
    req_id = request_id_from_response(resp)
    req_key = "<missing>"
    if getattr(resp, "request", None) is not None:
        req_key = resp.request.headers.get("Idempotency-Key", "<missing>")
    print(f"{tag} Status Code       : {resp.status_code}")
    print(f"{tag} Idempotency-Key   : {req_key}")
    print(f"{tag} Duffel-Request-ID : {req_id}")
    body_preview = resp.text[:800] if resp.text else ""
    print(f"{tag} Body              : {body_preview}")


def request_id_from_response(resp: requests.Response) -> str:
    # Duffel may vary header casing / aliases across endpoints.
    for key in ("Duffel-Request-ID", "Duffel-Request-Id", "X-Request-ID", "X-Request-Id"):
        req_id = resp.headers.get(key)
        if req_id:
            return req_id
    try:
        return ((resp.json().get("meta") or {}).get("request_id")) or "<missing>"
    except Exception:
        return "<missing>"


def print_acl(tag: str, acl: ACLResult) -> None:
    print(
        f"{tag} original_status={acl.original_status}, "
        f"semantic_status={acl.semantic_status}, outcome={acl.outcome.value}, request_id={acl.request_id}"
    )
    if acl.note:
        print(f"{tag} note={acl.note}")


def print_saga(tag: str, acl: ACLResult) -> None:
    saga_decision = "CONTINUE_AS_COMPENSATED" if acl.outcome == Outcome.SUCCESS else "MARK_COMPENSATION_ERROR"
    print(f"{tag} decision={saga_decision}, normalized_outcome={acl.outcome.value}")


def cancel_once(base: str, h: dict[str, str], order_id: str, tag: str, idempotency_key: str) -> dict[str, object]:
    print(f"\n[{tag}] [Raw API]")
    headers_create = dict(h)
    headers_create["Idempotency-Key"] = idempotency_key
    # Step 1: create cancellation request
    r_create = requests.post(
        f"{base}/air/order_cancellations",
        headers=headers_create,
        json={"data": {"order_id": order_id}},
        timeout=30,
    )
    print_resp(f"[{tag} create]", r_create)
    acl_create = ResultConverter.convert(r_create)
    print(f"[{tag}] [ACL Translation]")
    print_acl(f"[{tag} create]", acl_create)

    # Already cancelled case: Duffel commonly returns 422 here.
    if acl_create.outcome == Outcome.SUCCESS and r_create.status_code == 422:
        print(f"[{tag}] [Saga Outcome]")
        print_saga(f"[{tag}]", acl_create)
        return {
            "raw_create_status": r_create.status_code,
            "raw_confirm_status": None,
            "acl_result": acl_create,
            "confirm_acl_result": None,
            "create_status": 422,
            "confirm_status": None,
            "request_id_create": request_id_from_response(r_create),
            "request_id_confirm": None,
            "idempotent_path": True,
        }

    r_create.raise_for_status()
    cancellation_id = r_create.json()["data"]["id"]

    print(f"\n[{tag}] [Raw API]")
    headers_confirm = dict(h)
    headers_confirm["Idempotency-Key"] = f"{idempotency_key}:confirm"
    # Step 2: confirm cancellation
    r_confirm = requests.post(
        f"{base}/air/order_cancellations/{cancellation_id}/actions/confirm",
        headers=headers_confirm,
        timeout=30,
    )
    print_resp(f"[{tag} confirm]", r_confirm)
    acl_confirm = ResultConverter.convert(r_confirm)
    print(f"[{tag}] [ACL Translation]")
    print_acl(f"[{tag} confirm]", acl_confirm)
    if acl_confirm.outcome == Outcome.FAILED:
        r_confirm.raise_for_status()
    print(f"[{tag}] [Saga Outcome]")
    print_saga(f"[{tag}]", acl_confirm)

    return {
        "raw_create_status": r_create.status_code,
        "raw_confirm_status": r_confirm.status_code,
        "acl_result": acl_confirm,
        "confirm_acl_result": acl_confirm,
        "create_status": r_create.status_code,
        "confirm_status": r_confirm.status_code,
        "request_id_create": request_id_from_response(r_create),
        "request_id_confirm": request_id_from_response(r_confirm),
        "idempotent_path": False,
    }


def main() -> int:
    if not DUFFEL_TOKEN:
        print("ERROR: DUFFEL_ACCESS_TOKEN is not set.")
        return 1

    h = headers()
    target_date = (date.today() + timedelta(days=30)).isoformat()
    passenger_profile = {
        "given_name": "Test",
        "family_name": "Passenger",
        "born_on": "1990-01-01",
        "phone_number": "+886912345678",
        "email": "test.passenger@example.com",
        "gender": "m",
        "title": "mr",
    }

    hr("1) Offer Request")
    offer_req_payload = {
        "data": {
            "slices": [
                {
                    "origin": "TPE",
                    "destination": "NRT",
                    "departure_date": target_date,
                }
            ],
            "passengers": [{"type": "adult"}],
            "cabin_class": "economy",
        }
    }
    offer_req_payload["data"]["passengers"][0].update(passenger_profile)
    r_offer = requests.post(
        f"{DUFFEL_BASE}/air/offer_requests",
        headers=h,
        json=offer_req_payload,
        timeout=30,
    )
    print_resp("[offer]", r_offer)
    r_offer.raise_for_status()

    offer_data = r_offer.json()["data"]
    offers = offer_data.get("offers", [])
    if not offers:
        print("ERROR: No offers found.")
        return 2

    offer_id = offers[0]["id"]
    passenger_id = offer_data["passengers"][0]["id"]
    print(f"offer_id={offer_id}")
    print(f"passenger_id={passenger_id}")

    hr("2) Hold Order")
    hold_payload = {
        "data": {
            "type": "hold",
            "selected_offers": [offer_id],
            "passengers": [
                {
                    "id": passenger_id,
                    "given_name": passenger_profile["given_name"],
                    "family_name": passenger_profile["family_name"],
                    "born_on": passenger_profile["born_on"],
                    "phone_number": passenger_profile["phone_number"],
                    "email": passenger_profile["email"],
                    "gender": passenger_profile["gender"],
                    "title": passenger_profile["title"],
                }
            ],
        }
    }
    r_hold = requests.post(
        f"{DUFFEL_BASE}/air/orders",
        headers=h,
        json=hold_payload,
        timeout=30,
    )
    print_resp("[hold ]", r_hold)
    r_hold.raise_for_status()

    order_id = r_hold.json()["data"]["id"]
    print(f"order_id={order_id}")

    cancel_idempotency_key = f"saga_demo_cancel_{order_id}"
    print(f"cancel_idempotency_key={cancel_idempotency_key}")

    hr("3) Cancellation #1")
    cancel_1 = cancel_once(DUFFEL_BASE, h, order_id, "cancel#1", cancel_idempotency_key)

    hr("4) Cancellation #2 (same order_id)")
    cancel_2 = cancel_once(DUFFEL_BASE, h, order_id, "cancel#2", cancel_idempotency_key)

    hr("5) Summary")
    c1_acl: ACLResult = cancel_1["acl_result"]  # type: ignore[assignment]
    c2_acl: ACLResult = cancel_2["acl_result"]  # type: ignore[assignment]
    summary = {
        "order_id": order_id,
        "cancel_1": {
            "raw": {
                "create_status": cancel_1["raw_create_status"],
                "confirm_status": cancel_1["raw_confirm_status"],
            },
            "acl": {
                "original_status": c1_acl.original_status,
                "semantic_status": c1_acl.semantic_status,
                "outcome": c1_acl.outcome.value,
            },
            "request_ids": {
                "create": cancel_1["request_id_create"],
                "confirm": cancel_1["request_id_confirm"],
            },
        },
        "cancel_2": {
            "raw": {
                "create_status": cancel_2["raw_create_status"],
                "confirm_status": cancel_2["raw_confirm_status"],
            },
            "acl": {
                "original_status": c2_acl.original_status,
                "semantic_status": c2_acl.semantic_status,
                "outcome": c2_acl.outcome.value,
            },
            "request_ids": {
                "create": cancel_2["request_id_create"],
                "confirm": cancel_2["request_id_confirm"],
            },
        },
        "cancel_1_create_status": cancel_1["create_status"],
        "cancel_1_confirm_status": cancel_1["confirm_status"],
        "cancel_1_request_id_create": cancel_1["request_id_create"],
        "cancel_1_request_id_confirm": cancel_1["request_id_confirm"],
        "cancel_2_create_status": cancel_2["create_status"],
        "cancel_2_confirm_status": cancel_2["confirm_status"],
        "cancel_2_request_id_create": cancel_2["request_id_create"],
        "cancel_2_request_id_confirm": cancel_2["request_id_confirm"],
        "same_request_id": (
            cancel_1["request_id_create"] == cancel_2["request_id_create"]
            and cancel_1["request_id_confirm"] == cancel_2["request_id_confirm"]
        ),
        "cancel_2_used_idempotent_path": cancel_2["idempotent_path"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except requests.HTTPError as e:
        print(f"HTTPError: {e}")
        if e.response is not None:
            print_resp("[error]", e.response)
        raise
