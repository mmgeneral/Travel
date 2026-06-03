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


def test_from_dict_backward_compat_bare_string():
    e = StandardCheckpointEntry.from_dict("legacy-uuid-string")
    assert e.get_id() == "legacy-uuid-string"
    assert e.get_type() == "user_turn"
    assert e.get_description() == ""
    assert e.get_parent_id() is None


def test_from_dict_checkpoint_id_fallback_key():
    d = {"checkpoint_id": "old-uuid", "type": "user_turn", "description": ""}
    e = StandardCheckpointEntry.from_dict(d)
    assert e.get_id() == "old-uuid"


def test_from_dict_parent_id_preserved():
    d = {"id": "child-uuid", "type": "slot_change", "description": "", "parent_id": "parent-uuid"}
    e = StandardCheckpointEntry.from_dict(d)
    assert e.get_parent_id() == "parent-uuid"


def test_from_dict_missing_type_defaults_to_user_turn():
    d = {"id": "uuid-001"}
    e = StandardCheckpointEntry.from_dict(d)
    assert e.get_type() == "user_turn"


def test_entry_from_raw_dict():
    d = {"id": "uuid-001", "type": "slot_change", "description": "換午餐"}
    e = entry_from_raw(d)
    assert e.get_id() == "uuid-001"
    assert e.get_type() == "slot_change"


def test_entry_from_raw_str():
    e = entry_from_raw("legacy-str")
    assert e.get_id() == "legacy-str"
    assert e.get_type() == "user_turn"


def test_entries_from_state_empty():
    result = entries_from_state({})
    assert result == []


def test_entries_from_state_mixed_list():
    state = {
        "turn_checkpoints": [
            "legacy-str",
            {"id": "new-uuid", "type": "slot_change", "description": "換午餐"},
        ]
    }
    entries = entries_from_state(state)
    assert len(entries) == 2
    assert entries[0].get_id() == "legacy-str"
    assert entries[0].get_type() == "user_turn"
    assert entries[1].get_id() == "new-uuid"
    assert entries[1].get_type() == "slot_change"


def test_entries_to_state_roundtrip():
    entries = [
        StandardCheckpointEntry(id="a", type="user_turn", description="q1"),
        StandardCheckpointEntry(id="b", type="slot_change", description="換店", parent_id="a"),
    ]
    serialized = entries_to_state(entries)
    assert len(serialized) == 2
    assert serialized[0]["id"] == "a"
    assert serialized[1]["id"] == "b"
    assert serialized[1]["parent_id"] == "a"
    restored = entries_from_state({"turn_checkpoints": serialized})
    assert [e.get_id() for e in restored] == ["a", "b"]
    assert restored[1].get_parent_id() == "a"
