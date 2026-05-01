from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionOutcome(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    RETRYABLE = "RETRYABLE"


@dataclass
class SagaActionResult:
    outcome: ActionOutcome
    semantic_status: str
    trace_id: str = ""
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str = ""

