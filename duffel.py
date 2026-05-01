"""
Duffel Air Orders API client
============================
Covers only the two operations needed for the Saga hold-and-cancel flow:
  hold_order()   → POST /air/orders  (type=hold)
  cancel_order() → POST /air/order_cancellations + confirm action
"""

from __future__ import annotations

import hashlib
import os
import requests
from dotenv import load_dotenv
from requests import RequestException

from acl import ActionOutcome, SagaActionResult

load_dotenv()

DUFFEL_BASE = os.getenv("DUFFEL_BASE_URL", "https://api.duffel.com")
DUFFEL_VERSION = "v2"


class DuffelService:
    def __init__(self) -> None:
        self._token = os.environ["DUFFEL_ACCESS_TOKEN"]

    def _headers(self, idempotency_key: str = "") -> dict:
        if not idempotency_key:
            raise ValueError("Idempotency-Key is required for write operations")
        h = {
            "Authorization":  f"Bearer {self._token}",
            "Duffel-Version": DUFFEL_VERSION,
            "Content-Type":   "application/json",
            "Accept":         "application/json",
        }
        if idempotency_key:
            h["Idempotency-Key"] = idempotency_key
        return h

    @staticmethod
    def build_idempotency_key(saga_id: str, step_name: str, suffix: str = "") -> str:
        seed = f"saga_{saga_id}_{step_name}"
        if suffix:
            digest = hashlib.sha256(suffix.encode("utf-8")).hexdigest()[:16]
            return f"{seed}_{digest}"
        return seed

    @staticmethod
    def _request_id(resp: requests.Response) -> str:
        rid = resp.headers.get("Duffel-Request-ID")
        if rid:
            return rid
        try:
            return (resp.json().get("meta") or {}).get("request_id", "")
        except Exception:
            return ""

    def _post(self, path: str, payload: dict, idempotency_key: str, timeout: int = 20) -> requests.Response:
        return requests.post(
            f"{DUFFEL_BASE}{path}",
            headers=self._headers(idempotency_key=idempotency_key),
            json=payload,
            timeout=timeout,
        )

    def get_order(self, order_id: str, timeout: int = 10) -> dict:
        resp = requests.get(
            f"{DUFFEL_BASE}/air/orders/{order_id}",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Duffel-Version": DUFFEL_VERSION,
                "Accept": "application/json",
            },
            timeout=timeout,
        )
        request_id = self._request_id(resp)
        if resp.status_code == 404:
            return {"exists": False, "status": "missing", "duffel_request_id": request_id}
        if resp.status_code >= 500:
            return {"exists": True, "status": "unknown_error", "http_status": resp.status_code, "duffel_request_id": request_id}
        if 200 <= resp.status_code < 300:
            body = resp.json().get("data", {})
            return {
                "exists": True,
                "status": str(body.get("status", "unknown")).lower(),
                "duffel_request_id": request_id,
                "order": body,
            }
        return {"exists": True, "status": "unknown", "http_status": resp.status_code, "duffel_request_id": request_id}

    def search_offers(
        self,
        origin: str,
        destination: str,
        date: str,          # YYYY-MM-DD
        cabin_class: str = "economy",
    ) -> dict:
        """
        POST /air/offer_requests — returns {offer_request_id, passenger_id, offers[]}.
        Caller picks an offer_id + passenger_id from here, then calls hold_order().
        """
        payload = {
            "data": {
                "slices": [{"origin": origin, "destination": destination, "departure_date": date}],
                "passengers": [{"type": "adult"}],
                "cabin_class": cabin_class,
            }
        }
        key = self.build_idempotency_key(
            saga_id="system",
            step_name="offer_request",
            suffix=f"{origin}:{destination}:{date}:{cabin_class}",
        )
        resp = self._post(
            "/air/offer_requests",
            payload=payload,
            idempotency_key=key,
            timeout=20,
        )
        resp.raise_for_status()
        body = resp.json()["data"]
        return {
            "offer_request_id": body["id"],
            "passenger_id":     body["passengers"][0]["id"],
            "offers":           body.get("offers", []),
        }

    def hold_order(self, offer_id: str, passenger_id: str, idempotency_key: str = "") -> SagaActionResult:
        """
        Creates a hold order. Returns the Duffel order object dict.
        The idempotency_key ensures a crashed Saga can replay this call
        without creating a duplicate hold.
        """
        key = idempotency_key or self.build_idempotency_key("system", "hold_order", f"{offer_id}:{passenger_id}")
        try:
            resp = self._post(
                "/air/orders",
                payload={
                    "data": {
                        "type": "hold",
                        "selected_offers": [offer_id],
                        "passengers": [{"id": passenger_id}],
                    }
                },
                idempotency_key=key,
                timeout=15,
            )
        except requests.Timeout as e:
            return SagaActionResult(
                outcome=ActionOutcome.RETRYABLE,
                semantic_status="HOLD_TIMEOUT_RETRYABLE",
                trace_id="",
                raw_metadata={"http_status": 0},
                error_message=str(e),
            )
        except RequestException as e:
            return SagaActionResult(
                outcome=ActionOutcome.RETRYABLE,
                semantic_status="HOLD_NETWORK_RETRYABLE",
                trace_id="",
                raw_metadata={"http_status": 0},
                error_message=str(e),
            )

        request_id = self._request_id(resp)
        if 200 <= resp.status_code < 300:
            return SagaActionResult(
                outcome=ActionOutcome.SUCCESS,
                semantic_status="HOLD_SUCCESS",
                trace_id=request_id,
                raw_metadata={"http_status": resp.status_code, "duffel_request_id": request_id, "order": resp.json()["data"]},
            )
        if resp.status_code >= 500:
            return SagaActionResult(
                outcome=ActionOutcome.RETRYABLE,
                semantic_status="HOLD_RETRYABLE",
                trace_id=request_id,
                raw_metadata={"http_status": resp.status_code, "duffel_request_id": request_id},
                error_message=resp.text[:500],
            )
        return SagaActionResult(
            outcome=ActionOutcome.FAILED,
            semantic_status="HOLD_FAILED",
            trace_id=request_id,
            raw_metadata={"http_status": resp.status_code, "duffel_request_id": request_id},
            error_message=resp.text[:500],
        )

    def cancel_order(self, order_id: str, step_id: str = "") -> SagaActionResult:
        """
        Cancels a hold order via the two-step Duffel flow:
          1. POST /air/order_cancellations          → creates cancellation
          2. POST /air/order_cancellations/{id}/actions/confirm → commits it

        Returns the HTTP status code of the confirm step (204 on success).
        Calling this twice is safe: if Duffel says the order is already
        cancelled (422) we treat that as idempotent success and still return 204.
        """
        saga_id, step_name = (step_id.split(":", 1) + ["cancel"])[:2] if step_id else ("system", "cancel")
        create_key = self.build_idempotency_key(saga_id, f"{step_name}_cancel_create", order_id)
        confirm_key = self.build_idempotency_key(saga_id, f"{step_name}_cancel_confirm", order_id)

        try:
            r1 = self._post(
                "/air/order_cancellations",
                payload={"data": {"order_id": order_id}},
                idempotency_key=create_key,
                timeout=10,
            )
        except requests.Timeout as e:
            return SagaActionResult(
                outcome=ActionOutcome.RETRYABLE,
                semantic_status="TIMEOUT_ON_CANCEL_CREATE",
                trace_id="",
                raw_metadata={"http_status": 0, "duffel_request_id": "", "order_id": order_id},
                error_message=str(e),
            )
        except RequestException as e:
            return SagaActionResult(
                outcome=ActionOutcome.RETRYABLE,
                semantic_status="NETWORK_ERROR_ON_CANCEL_CREATE",
                trace_id="",
                raw_metadata={"http_status": 0, "duffel_request_id": "", "order_id": order_id},
                error_message=str(e),
            )

        request_id_create = self._request_id(r1)
        if r1.status_code == 422:
            try:
                body = r1.json()
            except Exception:
                body = {}
            for err in body.get("errors", []):
                if err.get("code") == "already_cancelled":
                    return SagaActionResult(
                        outcome=ActionOutcome.SUCCESS,
                        semantic_status="ALREADY_CANCELLED_SUCCESS",
                        trace_id=request_id_create,
                        raw_metadata={
                            "http_status": 422,
                            "duffel_request_id": request_id_create,
                            "order_id": order_id,
                            "phase": "create",
                        },
                    )
            return SagaActionResult(
                outcome=ActionOutcome.FAILED,
                semantic_status="CANCEL_CREATE_VALIDATION_FAILED",
                trace_id=request_id_create,
                raw_metadata={"http_status": 422, "duffel_request_id": request_id_create, "order_id": order_id},
                error_message=r1.text[:500],
            )
        if 400 <= r1.status_code < 500:
            return SagaActionResult(
                outcome=ActionOutcome.FAILED,
                semantic_status="CANCEL_CREATE_FAILED",
                trace_id=request_id_create,
                raw_metadata={"http_status": r1.status_code, "duffel_request_id": request_id_create, "order_id": order_id},
                error_message=r1.text[:500],
            )
        if r1.status_code >= 500:
            return SagaActionResult(
                outcome=ActionOutcome.RETRYABLE,
                semantic_status="CANCEL_CREATE_RETRYABLE",
                trace_id=request_id_create,
                raw_metadata={"http_status": r1.status_code, "duffel_request_id": request_id_create, "order_id": order_id},
                error_message=r1.text[:500],
            )

        cancellation_id = r1.json()["data"]["id"]
        try:
            r2 = self._post(
                f"/air/order_cancellations/{cancellation_id}/actions/confirm",
                payload={},
                idempotency_key=confirm_key,
                timeout=10,
            )
        except requests.Timeout as e:
            return SagaActionResult(
                outcome=ActionOutcome.RETRYABLE,
                semantic_status="TIMEOUT_ON_CANCEL_CONFIRM",
                trace_id="",
                raw_metadata={
                    "http_status": 0,
                    "duffel_request_id": "",
                    "order_id": order_id,
                    "cancellation_id": cancellation_id,
                },
                error_message=str(e),
            )
        except RequestException as e:
            return SagaActionResult(
                outcome=ActionOutcome.RETRYABLE,
                semantic_status="NETWORK_ERROR_ON_CANCEL_CONFIRM",
                trace_id="",
                raw_metadata={
                    "http_status": 0,
                    "duffel_request_id": "",
                    "order_id": order_id,
                    "cancellation_id": cancellation_id,
                },
                error_message=str(e),
            )

        request_id_confirm = self._request_id(r2)
        if 200 <= r2.status_code < 300:
            return SagaActionResult(
                outcome=ActionOutcome.SUCCESS,
                semantic_status="CANCELLED_SUCCESS",
                trace_id=request_id_confirm,
                raw_metadata={
                    "http_status": r2.status_code,
                    "duffel_request_id": request_id_confirm,
                    "order_id": order_id,
                    "cancellation_id": cancellation_id,
                    "create_request_id": request_id_create,
                },
            )
        if 400 <= r2.status_code < 500:
            return SagaActionResult(
                outcome=ActionOutcome.FAILED,
                semantic_status="CANCEL_CONFIRM_FAILED",
                trace_id=request_id_confirm,
                raw_metadata={
                    "http_status": r2.status_code,
                    "duffel_request_id": request_id_confirm,
                    "order_id": order_id,
                    "cancellation_id": cancellation_id,
                },
                error_message=r2.text[:500],
            )
        return SagaActionResult(
            outcome=ActionOutcome.RETRYABLE,
            semantic_status="CANCEL_CONFIRM_RETRYABLE",
            trace_id=request_id_confirm,
            raw_metadata={
                "http_status": r2.status_code,
                "duffel_request_id": request_id_confirm,
                "order_id": order_id,
                "cancellation_id": cancellation_id,
            },
            error_message=r2.text[:500],
        )


# Backward-compatible alias for existing imports.
DuffelClient = DuffelService
