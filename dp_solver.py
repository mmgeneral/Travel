from __future__ import annotations
import json
import logging
import math
import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Optional
from agents.critic import CriticAgent
from debug_json import debug_json as _dj
from decision_engine import (
    GraphBuilder,
    GraphEdge,
    GraphNode,
    ItinerarySynthesizer,
    OptimizationMode,
    RankedShop,
    ScoringEngine,
    SpatioTemporalGraph,
    WeightProfile,
)
from feasibility_utils import can_transition, calculate_cooldown, estimate_travel_minutes
from shop_planning import ShopProfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import AgentState

logger = logging.getLogger(__name__)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math

    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    d1 = math.radians(lat2 - lat1)
    d2 = math.radians(lon2 - lon1)
    a = math.sin(d1 / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d2 / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def node_critic(state: "AgentState") -> "AgentState":
    from agent import _attach_node_error, _svc_llm_router
    if state.get("error"):
        return state
    state["critic_retry_count"] = state.get("critic_retry_count", 0) + 1
    print(_dj("debug_print", node="node_critic", message="CriticAgent starting"))
    try:
        agent = CriticAgent(llm_router=_svc_llm_router(state))
        report = agent.run(state)
    except Exception as exc:
        _attach_node_error(state, "critic", exc)
        state.setdefault("transit_audit", []).append(_dj("critic_failed", reason=str(exc)[:800]))
        return state
    state["critique_history"] = list(state.get("critique_history") or []) + [report.as_dict()]
    state["auditor_rejected"] = report.verdict == "request_more"
    state["auditor_feedback"] = "; ".join(report.requests_for_retriever)
    state.setdefault("transit_audit", []).append(
        _dj(
            "critic_verdict",
            verdict=report.verdict,
            accepted=len(report.accepted),
            rejected=len(report.rejected_with_reason),
            requests=report.requests_for_retriever,
            iteration=report.iteration,
            llm_analysis=report.llm_analysis,
        )
    )
    return state


def node_auditor(state: "AgentState") -> "AgentState":
    state["transit_audit"].append(
        _dj("auditor_review_skipped", reason="node_auditor not in main graph; use node_critic")
    )
    return state


def node_audit(state: "AgentState") -> "AgentState":
    state["transit_audit"].append(
        _dj("node_audit_skipped", reason="node_audit not in main graph (Task 7 ADR)")
    )
    return state


def _combine_itinerary_clock(itinerary_start: datetime, t: datetime) -> datetime:
    if itinerary_start.tzinfo is None and t.tzinfo is not None:
        t = t.replace(tzinfo=None)
    elif itinerary_start.tzinfo is not None and t.tzinfo is None:
        t = t.replace(tzinfo=itinerary_start.tzinfo)
    return itinerary_start.replace(
        hour=t.hour, minute=t.minute, second=t.second, microsecond=t.microsecond,
    )


def _relay_layer_one_edges_inplace(
    graph: SpatioTemporalGraph,
    *,
    ranked: list[RankedShop],
    traffic,
    mode: OptimizationMode = OptimizationMode.BALANCED,
) -> None:
    ranked_index = {r.shop.name: r for r in ranked}
    shop_index = {r.shop.name: r.shop for r in ranked}
    by_slot: dict[int, list] = {}
    for n in graph.nodes or []:
        by_slot.setdefault(n.slot_index, []).append(n)
    if not by_slot:
        graph.edges = []
        graph.debug_traces = []
        return
    slot_count = max(by_slot.keys()) + 1
    edges: list[GraphEdge] = []
    debug_traces: list[str] = []
    if slot_count > 1:
        for i in range(slot_count - 1):
            from_nodes = by_slot.get(i, [])
            if not from_nodes:
                continue
            for j in range(i + 1, slot_count):
                to_nodes = by_slot.get(j, [])
                if not to_nodes:
                    continue
                for a in from_nodes:
                    shop_a = shop_index.get(a.shop_name)
                    if shop_a is None:
                        continue
                    for b in to_nodes:
                        if a.shop_name == b.shop_name:
                            continue
                        shop_b = shop_index.get(b.shop_name)
                        if shop_b is None:
                            continue
                        # unified feasibility check
                        feasible, reason = can_transition(
                            a.start_time, shop_a,
                            b.start_time, shop_b,
                            mode=mode,
                            requested_meal_count=None,
                            appetite_light_mode=False,
                        )
                        if feasible:
                            # compute edge weight with same method as before
                            travel_min = estimate_travel_minutes(shop_a, shop_b)
                            cooldown_m = calculate_cooldown(
                                shop_a, mode=mode, requested_meal_count=None, appetite_light_mode=False,
                            )
                            fastest_finish_at = a.start_time + timedelta(
                                minutes=int(shop_a.base_wait_minutes)
                                + int(shop_a.min_eat_minutes or shop_a.avg_eat_minutes)
                            )
                            ready_at = fastest_finish_at + timedelta(minutes=cooldown_m) + timedelta(minutes=travel_min)
                            status = traffic.get_route_status(a.shop_name, b.shop_name)
                            queue_risk = min(1.0, max(0.0, float(shop_b.base_wait_minutes) / 45.0))
                            slack_m = max(0.0, (b.start_time - ready_at).total_seconds() / 60.0)
                            expected_buffer_m = max(0.0, float(status.transport_buffer_minutes))
                            buffer_gap_m = max(0.0, expected_buffer_m - slack_m)
                            travel_buffer_gap = min(1.0, buffer_gap_m / 45.0)
                            base_score = float(ranked_index[b.shop_name].final_score)
                            weight = ScoringEngine.risk_adjusted_score(
                                base_score, queue_risk=queue_risk, travel_buffer_gap=travel_buffer_gap,
                            )
                            edges.append(GraphEdge(from_node_id=a.node_id, to_node_id=b.node_id, weight=weight))
                        else:
                            debug_traces.append(
                                _dj(
                                    "graph_rejected_edge",
                                    from_shop=a.shop_name,
                                    to_shop=b.shop_name,
                                    detail=reason,
                                )
                            )
    graph.edges = edges
    graph.debug_traces = debug_traces


def _pin_dp_graph_to_itinerary_day_and_relayer_edges(
    graph: SpatioTemporalGraph,
    *,
    itinerary_start: datetime,
    ranked: list[RankedShop],
    traffic,
    mode: OptimizationMode = OptimizationMode.BALANCED,
) -> None:
    if not graph.nodes:
        return
    for n in graph.nodes:
        dur = n.end_time - n.start_time
        n.start_time = _combine_itinerary_clock(itinerary_start, n.start_time)
        n.end_time = n.start_time + dur
    _relay_layer_one_edges_inplace(graph, ranked=ranked, traffic=traffic, mode=mode)


def _expand_graph_node_tags_from_shop_profiles(graph: SpatioTemporalGraph, ranked: list[RankedShop]) -> None:
    by_name = {r.shop.name: r.shop for r in ranked}
    for n in graph.nodes or []:
        sp = by_name.get(n.shop_name)
        if sp is None:
            continue
        bag = {str(x).lower() for x in sp.tags} | {str(x).lower() for x in getattr(sp, "occasion_tags", ()) or []}
        n.tags = tuple(sorted(bag))


def agent_dp_find_optimal_path_no_shop_repeat(
    graph: SpatioTemporalGraph,
    required_length: int,
    must_have_tags: set[str] | None = None,
    banned_node_ids: set[str] | None = None,
    solver_audit_log: list[str] | None = None,
    excluded_shop_tags: frozenset[str] | None = None,
    allow_global_repeat: bool = False,
) -> list[GraphNode]:
    if solver_audit_log is not None:
        solver_audit_log.append(_dj("dp_start_agent", variant="unique_shops_soft_slot_tag_affinity",
                                    required_length=required_length, nodes=len(graph.nodes or [])))
    if required_length <= 0 or not graph.nodes:
        if solver_audit_log is not None:
            solver_audit_log.append(_dj("dp_early_exit_agent", reason="invalid_required_length_or_empty_graph"))
        return []
    must_have_tags_l = {t.lower() for t in (must_have_tags or set())}
    slot_names_raw = getattr(graph, "_agent_meal_slot_names_for_dp", None)
    slot_names: list[str] | None = list(slot_names_raw) if isinstance(slot_names_raw, list) else None
    required_tags_list = sorted(must_have_tags_l)
    tag_idx = {t: i for i, t in enumerate(required_tags_list)}
    full_mask = (1 << len(required_tags_list)) - 1
    banned_node_ids_set = banned_node_ids or set()
    usable_nodes = [n for n in graph.nodes if n.node_id not in banned_node_ids_set]
    _excl = excluded_shop_tags or frozenset()
    if _excl:
        usable_nodes = [n for n in usable_nodes
                        if not ItinerarySynthesizer.node_tags_intersect_excluded(n.tags, _excl)]
    if not usable_nodes:
        if solver_audit_log is not None:
            solver_audit_log.append(_dj("dp_early_exit_agent", reason="all_nodes_filtered_by_banned_node_ids"))
        return []
    node_by_id = {n.node_id: n for n in usable_nodes}
    indeg: dict[str, int] = {nid: 0 for nid in node_by_id}
    out_edges: dict[str, list[GraphEdge]] = {nid: [] for nid in node_by_id}
    for e in graph.edges:
        if (e.from_node_id not in node_by_id or e.to_node_id not in node_by_id
                or e.from_node_id in banned_node_ids_set or e.to_node_id in banned_node_ids_set):
            continue
        indeg[e.to_node_id] += 1
        out_edges[e.from_node_id].append(e)
    queue = [nid for nid, d in indeg.items() if d == 0]
    topo: list[str] = []
    while queue:
        queue.sort(key=lambda nid: (node_by_id[nid].slot_index, node_by_id[nid].start_time))
        cur = queue.pop(0)
        topo.append(cur)
        for e in out_edges[cur]:
            indeg[e.to_node_id] -= 1
            if indeg[e.to_node_id] == 0:
                queue.append(e.to_node_id)
    if len(topo) < len(node_by_id):
        topo = sorted(node_by_id.keys(), key=lambda nid: (node_by_id[nid].slot_index, node_by_id[nid].start_time))

    def tag_mask(node: GraphNode) -> int:
        m = 0
        node_tags_low = {t.lower() for t in node.tags}
        for t, idx in tag_idx.items():
            if t in node_tags_low:
                m |= 1 << idx
        return m

    def node_objective_score(node: GraphNode) -> float:
        ideal = max(1, int(node.ideal_duration_minutes))
        actual = max(1, int(node.actual_duration_minutes))
        fidelity = max(0.0, min(1.0, float(actual) / float(ideal)))
        bonus = 1.0
        if slot_names and 0 <= node.slot_index < len(slot_names):
            slot_nm = str(slot_names[node.slot_index]).lower()
            preferred = ItinerarySynthesizer.SLOT_PREFERRED_TAGS.get(slot_nm)
            if preferred and {str(t).lower() for t in node.tags} & preferred:
                bonus = 1.3
        return float(node.final_score) * fidelity * bonus

    # (node_id, length, mask, seen_shop_names, previous_shop_name) -> score
    KeyT = tuple[str, int, int, frozenset[str], str | None]
    best: dict[KeyT, float] = {}
    prev: dict[KeyT, KeyT | None] = {}

    for nid in topo:
        node = node_by_id[nid]
        m = tag_mask(node)
        fshops = frozenset({node.shop_name})
        key: KeyT = (nid, 1, m, fshops, node.shop_name)
        best[key] = node_objective_score(node)
        prev[key] = None

    for nid in topo:
        outgoing = out_edges.get(nid, [])
        cur_states = [(k, v) for k, v in best.items() if k[0] == nid]
        if not cur_states:
            continue
        for cur_key, cur_score in cur_states:
            _, cur_len, cur_mask, fshops_cur, cur_prev_shop = cur_key
            if cur_len >= required_length:
                continue
            for e in outgoing:
                to_node = node_by_id[e.to_node_id]
                # hard rule: never allow immediately consecutive same shop
                if to_node.shop_name == cur_prev_shop:
                    continue
                if not allow_global_repeat and to_node.shop_name in fshops_cur:
                    continue
                next_mask = cur_mask | tag_mask(to_node)
                if not allow_global_repeat:
                    fshops_next = frozenset(fshops_cur | {to_node.shop_name})
                else:
                    fshops_next = fshops_cur  # keep same set, do not force global uniqueness
                nxt: KeyT = (e.to_node_id, cur_len + 1, next_mask, fshops_next, to_node.shop_name)
                cand = cur_score + node_objective_score(to_node)
                if cand > best.get(nxt, float("-inf")):
                    best[nxt] = cand
                    prev[nxt] = cur_key

    max_len = max((k[1] for k in best.keys()), default=0)
    target_len = required_length if any(k[1] == required_length for k in best.keys()) else max_len
    if target_len <= 0:
        if solver_audit_log is not None:
            solver_audit_log.append(_dj("dp_early_exit_agent", reason="no_feasible_terminal_state"))
        return []
    terminal_keys = [k for k in best.keys() if k[1] == target_len]
    if required_tags_list:
        covered = [k for k in terminal_keys if k[2] == full_mask]
        if covered:
            terminal_keys = covered
        elif target_len == required_length:
            feasible_lens = sorted({k[1] for k in best.keys()}, reverse=True)
            for ln in feasible_lens:
                covered_ln = [k for k in best.keys() if k[1] == ln and k[2] == full_mask]
                if covered_ln:
                    terminal_keys = covered_ln
                    target_len = ln
                    break
    end_key = max(terminal_keys, key=lambda k: best[k])
    path_keys: list[KeyT] = []
    cur_k: KeyT | None = end_key
    while cur_k is not None:
        path_keys.append(cur_k)
        cur_k = prev.get(cur_k)
    path_keys.reverse()
    resolved_path = [node_by_id[k[0]] for k in path_keys]
    uniq = len({p.shop_name for p in resolved_path})
    if solver_audit_log is not None:
        solver_audit_log.append(_dj("dp_path_selected", variant="agent_unique_shops", unique_shop_count=uniq,
                                    path=[{"shop": n.shop_name, "slot": n.slot_index} for n in resolved_path]))
    return resolved_path


def _dp_graph_builder_with_itinerary_day_pin(
    ranked: list[RankedShop],
    traffic,
    start_time: datetime,
    *,
    meal_slots: list[str] | None = None,
    mode: OptimizationMode = OptimizationMode.BALANCED,
    requested_meal_count: int | None = None,
    slot_required_tags: dict[str, set[str]] | None = None,
    excluded_shop_tags: frozenset[str] | None = None,
) -> SpatioTemporalGraph:
    from agent import _RAW_GRAPHBUILDER_BUILD as _build_func
    g = _build_func(ranked, traffic, start_time,
                     meal_slots=meal_slots, mode=mode,
                     requested_meal_count=requested_meal_count,
                     slot_required_tags=slot_required_tags,
                     excluded_shop_tags=excluded_shop_tags)
    _pin_dp_graph_to_itinerary_day_and_relayer_edges(g, itinerary_start=start_time, ranked=ranked, traffic=traffic, mode=mode)
    setattr(g, "_agent_meal_slot_names_for_dp",
            list(ItinerarySynthesizer._normalize_slot_sequence(meal_slots or [])))
    _expand_graph_node_tags_from_shop_profiles(g, ranked)
    return g


@contextmanager
def _pinned_travel_dp_calendar_and_solver():
    from agent import _RAW_GRAPHBUILDER_BUILD
    original = GraphBuilder.build_graph
    GraphBuilder.build_graph = _dp_graph_builder_with_itinerary_day_pin
    try:
        yield
    finally:
        GraphBuilder.build_graph = original


def _append_graph_physical_transition_audit(
    *,
    graph,
    ranked: list[RankedShop],
    traffic,
    transit_audit: list,
    mode: OptimizationMode,
) -> None:
    shop_index = {r.shop.name: r.shop for r in ranked}
    traces = list(getattr(graph, "debug_traces", []) or [])
    rejects = 0
    reject_detail_sample: list[str] = []
    for tr in traces[:80]:
        try:
            o = json.loads(tr)
        except Exception:
            continue
        if isinstance(o, dict) and str(o.get("event")) == "graph_rejected_edge":
            rejects += 1
            if len(reject_detail_sample) < 4:
                dst = str(o.get("detail") or o.get("message") or "")[:260]
                reject_detail_sample.append(f"{o.get('from_shop')}→{o.get('to_shop')}: {dst}")

    by_slot: dict[int, list] = {}
    for n in graph.nodes or []:
        by_slot.setdefault(n.slot_index, []).append(n)
    probes: list[dict] = []
    indices = sorted(by_slot.keys())
    for low_i in range(max(0, len(indices) - 1)):
        i = indices[low_i]
        j = indices[low_i + 1]
        from_nodes = sorted(by_slot[i], key=lambda n: str(n.shop_name))[:5]
        to_nodes = sorted(by_slot[j], key=lambda n: str(n.shop_name))[:8]
        for a in from_nodes:
            shop_a = shop_index.get(a.shop_name)
            if shop_a is None:
                continue
            travel_m = 18 + (8 * i)
            fastest_finish = a.start_time + timedelta(
                minutes=int(shop_a.base_wait_minutes) + int(shop_a.min_eat_minutes or shop_a.avg_eat_minutes))
            ready_physical = fastest_finish + timedelta(minutes=travel_m)
            digest_minutes = ItinerarySynthesizer._calculate_cooldown(
                shop_a, mode=mode, requested_meal_count=None, appetite_light_mode=False)
            hypothetical_cool_ready = fastest_finish + timedelta(minutes=int(digest_minutes))
            for b in to_nodes[:3]:
                shop_b = shop_index.get(b.shop_name)
                if shop_b is None:
                    continue
                open_b = ItinerarySynthesizer._shop_open_at(b.start_time, shop_b)
                passed = ready_physical <= b.start_time and b.start_time >= open_b
                probes.append({
                    "slot_from": i,
                    "slot_to": j,
                    "from_shop": a.shop_name,
                    "from_start": str(a.start_time),
                    "from_end_ideal": str(getattr(a, "end_time", "")),
                    "eat_end_min_path": str(fastest_finish),
                    "cooldown_na_in_graph_edges": digest_minutes,
                    "hypothetical_ready_if_digest_enforced": str(hypothetical_cool_ready),
                    "travel_minutes_edge": travel_m,
                    "ready_at_physical_graph": str(ready_physical),
                    "to_shop": b.shop_name,
                    "b_start": str(b.start_time),
                    "b_open": str(open_b),
                    "edge_accepts_physical_layer1": passed,
                })
        if len(probes) >= 12:
            break
    slot_timelines: list[dict] = []
    for si in indices:
        layer = sorted(by_slot[si], key=lambda n: str(n.shop_name))
        if not layer:
            continue
        pick = layer[0]
        sa = shop_index.get(pick.shop_name)
        if sa is None:
            continue
        eat_end = pick.start_time + timedelta(
            minutes=int(sa.base_wait_minutes) + int(sa.min_eat_minutes or sa.avg_eat_minutes))
        digest_min = ItinerarySynthesizer._calculate_cooldown(
            sa, mode=mode, requested_meal_count=None, appetite_light_mode=False,
        )
        travel_out = 18 + (8 * si)
        slot_timelines.append({
            "slot_index": si,
            "repr_shop": pick.shop_name,
            "start": str(pick.start_time),
            "end_ideal_layer": str(pick.end_time),
            "eat_end_min_wait_path": str(eat_end),
            "synth_cooldown_minutes_note": digest_min,
            "outgoing_travel_minutes_layer1_formula": travel_out,
            "ready_at_physical_to_next_formula": f"eat_end_min + {travel_out}m travel (cooldown omitted in GraphBuilder)",
        })
    row = _dj("graph_physical_probe",
              unique_slots=len(by_slot), nodes=len(graph.nodes or []),
              edges=len(graph.edges or []),
              rejected_edges_logged=rejects,
              reject_reason_samples=reject_detail_sample,
              slot_repr_timeline=slot_timelines,
              sample_transitions=probes,
              note="ready_at=start+wait+min_eat+travel; GraphBuilder omits digestion cooldown vs synthesize()")
    transit_audit.append(row)
    dbg = os.getenv("GRAPH_PHYSICS_DEBUG", "").strip().lower()
    if dbg not in {"", "0", "false"}:
        try:
            obj = json.loads(row)
            print(json.dumps({"GRAPH_PHYSICS_DEBUG": obj}, ensure_ascii=False, indent=2, default=str))
        except Exception:
            print(row)
