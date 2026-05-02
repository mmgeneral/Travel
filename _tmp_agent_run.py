from datetime import datetime
from agent import _build_shop_catalog, _effective_plan_meal_slots
from shop_planning import MockTrafficProvider
from decision_engine import (
    GraphBuilder,
    ItinerarySynthesizer,
    OptimizationMode,
    RankingEngine,
    UserMinefield,
    UserPreference,
)

query = "京都 7點開始 安排五餐"
intent = {}
slots = _effective_plan_meal_slots(intent, query)
shops = list(_build_shop_catalog())
pref = UserPreference()
mine = UserMinefield()
ranked, _rej = RankingEngine.rank(shops, pref, mine)
phase1 = ranked[:15]

g = GraphBuilder.build_graph(
    ranked=phase1,
    traffic=MockTrafficProvider(),
    start_time=datetime(2026, 5, 10, 7, 0, 0),
    meal_slots=slots,
    mode=OptimizationMode.BALANCED,
    requested_meal_count=5,
    slot_required_tags=None,
)
uniq_slots = sorted({n.slot_index for n in g.nodes})

path = ItinerarySynthesizer.find_optimal_path(
    graph=g,
    required_length=len(slots),
    must_have_tags=set(),
    solver_audit_log=None,
)
names = [n.shop_name for n in path]

print("SLOTS", slots)
print("slot_indices_in_graph", uniq_slots)
print("edges", len(g.edges))
print("PATH_LEN", len(path), "PATH", names)
print("UNIQUE", len(names), len(set(names)))
assert len(slots) == 5 == len(path)
assert len(set(names)) == len(names)
