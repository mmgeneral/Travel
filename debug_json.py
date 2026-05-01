"""Structured JSON lines for machine-parseable debug / audit logs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def debug_json(event: str, **fields: Any) -> str:
    """
    Return a single-line JSON object for append-only audit streams.
    Omit keys whose value is None.
    """
    doc: dict[str, Any] = {
        "event": event,
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    for k, v in fields.items():
        if v is not None:
            doc[k] = v
    return json.dumps(doc, ensure_ascii=False, default=str)


def audit_json_line_as_text(line: str) -> str:
    """Best-effort human-readable summary for mixed legacy + JSON audit lines."""
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            ev = obj.get("event", "")
            detail = obj.get("detail") or obj.get("message")
            if detail:
                return f"{ev}: {detail}" if ev else str(detail)
            return ev or line
    except json.JSONDecodeError:
        pass
    return line
