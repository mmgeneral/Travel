import pytest
from checkpoint_entry import (
    StandardCheckpointEntry,
    entry_from_raw,
    entries_from_state,
    entries_to_state,
    VALID_TYPES,
)


# ── StandardCheckpointEntry 基本建立 ─────────────────────

def test_basic_construction_and_getters():
    e = StandardCheckpointEntry(
        id="uuid-001",
        type="user_turn",
        description="排京都一日行程",
        parent_id="uuid-000",
    )
    assert e.get_id() == "uuid-001"
    assert e.get_type() == "user_turn"
    assert e.get_description() == "排京都一日行程"
    assert e.get_parent_id() == "uuid-000"
    assert e.get_ts() is not None  # 自動填充


def test_invalid_type_raises_value_error():
    with pytest.raises(ValueError, match="Invalid checkpoint type"):
        StandardCheckpointEntry(id="x", type="invalid_type")


def test_ts_auto_filled_when_not_provided():
    e = StandardCheckpointEntry(id="x", type="slot_change")
    assert e.get_ts() is not None
    assert "T" in e.get_ts()  # ISO format 包含 T


def test_parent_id_defaults_to_none():
    e = StandardCheckpointEntry(id="x", type="user_confirmed")
    assert e.get_parent_id() is None


def test_all_valid_types_accepted():
    for t in VALID_TYPES:
        e = StandardCheckpointEntry(id="x", type=t)
        assert e.get_type() == t


# ── to_dict / from_dict roundtrip ───────────────────────

def test_to_dict_contains_all_fields():
    e = StandardCheckpointEntry(
        id="uuid-001",
        type="slot_change",
        description="換午餐",
        ts="2026-06-01T00:00:00+00:00",
        parent_id="uuid-000",
    )
    d = e.to_dict()
    assert d["id"] == "uuid-001"
    assert d["type"] == "slot_change"
    assert d["description"] == "換午餐"
    assert d["ts"] == "2026-06-01T00:00:00+00:00"
    assert d["parent_id"] == "uuid-000"


def test_from_dict_roundtrip():
    e = StandardCheckpointEntry(
        id="uuid-001",
        type="user_confirmed",
        description="京都第一天",
        ts="2026-06-01T00:00:00+00:00",
        parent_id="uuid-000",
    )
    d = e.to_dict()
    e2 = StandardCheckpointEntry.from_dict(d)
    assert e2
