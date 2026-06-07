import pytest

from dietary_utils import (
    _user_negates_food_category_in_query,
    _constraint_string_to_dietary_keys,
    _dietary_keys_from_query_and_intent_signals,
    _build_plan_excluded_shop_tags,
    _parse_dietary_clarification_reply,
)


# ── _user_negates_food_category_in_query ─────────────────


def test_negates_ramen_chinese():
    assert _user_negates_food_category_in_query("不吃拉麵", "ramen") is True


def test_negates_ramen_english():
    assert _user_negates_food_category_in_query("no ramen please", "ramen") is True


def test_negates_izakaya():
    assert _user_negates_food_category_in_query("不吃居酒屋", "izakaya") is True


def test_negates_dessert():
    assert _user_negates_food_category_in_query("不吃甜點", "dessert") is True


def test_no_negation_returns_false():
    assert _user_negates_food_category_in_query("想吃拉麵", "ramen") is False


def test_unknown_category_returns_false():
    assert _user_negates_food_category_in_query("不吃拉麵", "sushi") is False


# ── _constraint_string_to_dietary_keys ───────────────────


def test_constraint_empty_returns_empty():
    assert _constraint_string_to_dietary_keys("") == []


def test_constraint_no_beef_chinese():
    assert "no_beef" in _constraint_string_to_dietary_keys("不吃牛")


def test_constraint_no_pork_chinese():
    assert "no_pork" in _constraint_string_to_dietary_keys("不吃豬")


def test_constraint_no_ramen_chinese():
    assert "no_ramen" in _constraint_string_to_dietary_keys("不吃拉麵")


def test_constraint_vegan_english():
    result = _constraint_string_to_dietary_keys("vegan")
    assert "vegan" in result


# ── _dietary_keys_from_query_and_intent_signals ──────────


def test_dietary_keys_vegan():
    assert "vegan" in _dietary_keys_from_query_and_intent_signals("我要vegan餐廳")


def test_dietary_keys_vegetarian_chinese():
    assert "vegetarian" in _dietary_keys_from_query_and_intent_signals("素食餐廳")


def test_dietary_keys_no_beef():
    assert "no_beef" in _dietary_keys_from_query_and_intent_signals("不吃牛")


def test_dietary_keys_no_pork():
    assert "no_pork" in _dietary_keys_from_query_and_intent_signals("不吃豬")


def test_dietary_keys_no_ramen():
    assert "no_ramen" in _dietary_keys_from_query_and_intent_signals("不吃拉麵")


def test_dietary_keys_empty():
    assert _dietary_keys_from_query_and_intent_signals("") == []


def test_dietary_keys_pescatarian():
    assert "pescatarian" in _dietary_keys_from_query_and_intent_signals("pescatarian")


# ── _build_plan_excluded_shop_tags ───────────────────────


def test_build_excluded_from_dietary_hints():
    intent = {"dietary_hints": "no_beef", "explicit_constraints": []}
    result = _build_plan_excluded_shop_tags("", intent, None)
    assert isinstance(result, frozenset)
    assert len(result) > 0


def test_build_excluded_from_profile_ethics():
    result = _build_plan_excluded_shop_tags("", {}, {"ethics": "vegan"})
    assert isinstance(result, frozenset)
    assert len(result) > 0


def test_build_excluded_from_query():
    result = _build_plan_excluded_shop_tags("不吃牛", {}, None)
    assert isinstance(result, frozenset)
    assert len(result) > 0


def test_build_excluded_empty():
    result = _build_plan_excluded_shop_tags("", {}, None)
    assert isinstance(result, frozenset)


# ── _parse_dietary_clarification_reply ───────────────────


def test_parse_reply_A_strict():
    assert _parse_dietary_clarification_reply("A") == "strict"


def test_parse_reply_B_loose():
    assert _parse_dietary_clarification_reply("B") == "loose"


def test_parse_reply_unclear_returns_none():
    assert _parse_dietary_clarification_reply("隨便") is None


def test_parse_reply_full_text_strict():
    assert _parse_dietary_clarification_reply("菜單完全不能有牛肉") == "strict"


def test_parse_reply_full_text_loose():
    assert _parse_dietary_clarification_reply("我自己不點牛肉") == "loose"
