from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

VALID_TYPES = {"user_turn", "slot_change", "user_confirmed"}


class CheckpointEntry(ABC):
    @abstractmethod
    def get_id(self) -> str: ...

    @abstractmethod
    def get_type(self) -> str: ...

    @abstractmethod
    def get_description(self) -> str: ...

    @abstractmethod
    def get_ts(self) -> str: ...

    @abstractmethod
    def get_parent_id(self) -> str | None: ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckpointEntry": ...


class StandardCheckpointEntry(CheckpointEntry):
    """Concrete implementation for all checkpoint types."""

    def __init__(
        self,
        id: str,
        type: str,
        description: str = "",
        ts: str | None = None,
        parent_id: str | None = None,
    ) -> None:
        if type not in VALID_TYPES:
            raise ValueError(f"Invalid checkpoint type: {type!r}")
        self._id = id
        self._type = type
        self._description = description
        self._ts = ts or datetime.now(timezone.utc).isoformat()
        self._parent_id = parent_id

    def get_id(self) -> str:
        return self._id

    def get_type(self) -> str:
        return self._type

    def get_description(self) -> str:
        return self._description

    def get_ts(self) -> str:
        return self._ts

    def get_parent_id(self) -> str | None:
        return self._parent_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self._id,
            "type": self._type,
            "description": self._description,
            "ts": self._ts,
            "parent_id": self._parent_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StandardCheckpointEntry":
        if isinstance(d, str):
            # backward compat: bare string → user_turn with no description
            return cls(id=d, type="user_turn", description="")
        return cls(
            id=str(d.get("id") or d.get("checkpoint_id") or ""),
            type=str(d.get("type") or "user_turn"),
            description=str(d.get("description") or ""),
            ts=d.get("ts"),
            parent_id=d.get("parent_id"),
        )


def entry_from_raw(raw: Any) -> StandardCheckpointEntry:
    """Convert raw state value (str or dict) to StandardCheckpointEntry."""
    if isinstance(raw, dict):
        return StandardCheckpointEntry.from_dict(raw)
    return StandardCheckpointEntry(id=str(raw), type="user_turn", description="")

def entries_from_state(state_vals: dict[str, Any]) -> list[StandardCheckpointEntry]:
    """Read turn_checkpoints from state dict and return list of entries."""
    raw = state_vals.get("turn_checkpoints") or []
    return [entry_from_raw(x) for x in raw if x is not None]

def entries_to_state(entries: list[StandardCheckpointEntry]) -> list[dict[str, Any]]:
    """Serialize entries for storage in AgentState."""
    return [e.to_dict() for e in entries]
