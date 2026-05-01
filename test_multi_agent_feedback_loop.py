from __future__ import annotations

from agent import build_graph, make_initial_state


def test_multi_agent_feedback_loop_max_three_iterations() -> None:
    graph = build_graph()
    state = make_initial_state("請給我一個美食行程，店不要太遠")
    out = graph.invoke(state)
    iters = int(out.get("research_iteration", 0))
    assert 1 <= iters <= 3
    assert isinstance(out.get("auditor_feedback", ""), str)


def test_auditor_feedback_written_to_audit_log() -> None:
    graph = build_graph()
    state = make_initial_state("安排今天拉麵行程")
    out = graph.invoke(state)
    logs = "\n".join(out.get("transit_audit", []))
    assert "auditor_review" in logs
    assert "researcher_iteration" in logs
