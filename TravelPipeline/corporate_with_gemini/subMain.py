import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import sys

from PlannerAgent import generate_travel_itinerary_json
from Mcp import extract_searchable_activities
from AnchorResolver import resolve_anchor_context, save_anchor_context_output
from PlacesSearchAgent import search_places_for_pending_searches
from CandidateScorer import save_scoring_outputs, score_candidate_slots
from ConstraintExtractor import extract_constraints_from_query
from Verifier import verify_planner_slots
from constraint_ui import run_constraint_confirmation

# 注意：FinalItineraryAgent 需要額外設定 OPENAI_API_KEY 環境變數
# （跟其他模組使用的 Gemini key 不同），請確認 .env 或環境變數已設定。
from FinalItineraryAgent import generate_final_itinerary, save_json_output


OUTPUT_ROOT = Path(__file__).resolve().parent / "run_outputs"


def _make_run_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, data: str):
    path.write_text(data, encoding="utf-8")


def _parse_json_if_possible(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_text": raw}


def _compact_spaces(value) -> str:
    return " ".join(str(value or "").split())


def _trip_context_value(planner_result: dict, key: str) -> str:
    context = planner_result.get("trip_context")
    if not isinstance(context, dict):
        return ""
    value = context.get(key)
    if isinstance(value, dict):
        return _compact_spaces(value.get("place_query") or value.get("name"))
    return _compact_spaces(value)


async def run_multi_agent_flow(user_query: str = "去台南兩天一夜"):
    env_destination_hint = os.getenv("TRAVEL_DESTINATION_HINT", "")
    run_dir = _make_run_dir()

    print(f"📁 [Run Output] 本次輸出資料夾：{run_dir}")

    raw_constraints_result = extract_constraints_from_query(user_query)
    _write_json(
        run_dir / "00.5_raw_constraints_extraction.json",
        raw_constraints_result,
    )
    print("📋 [ConstraintExtractor] 抽取的原始約束：")
    print(json.dumps(raw_constraints_result, ensure_ascii=False, indent=2))
    print("=" * 20)

    planner_result = generate_travel_itinerary_json(user_query)
    planner_result = verify_planner_slots(planner_result, raw_constraints_result)

    destination_hint = (
        env_destination_hint
        or _trip_context_value(planner_result, "destination")
        or ""
    )
    start_anchor = _trip_context_value(planner_result, "arrival_anchor") or None
    end_anchor = _trip_context_value(planner_result, "return_anchor") or None

    _write_json(
        run_dir / "00_run_metadata.json",
        {
            "user_query": user_query,
            "destination_hint": destination_hint,
            "start_anchor": start_anchor,
            "end_anchor": end_anchor,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    _write_json(run_dir / "01_planner_agent_output.json", planner_result)

    itinerary_str = json.dumps(planner_result, ensure_ascii=False)
    print("📝 [PlannerAgent] 生成的旅遊行程 JSON：")
    print(itinerary_str)
    print("="*20)

    searchable_activities = extract_searchable_activities(itinerary_str, raw_constraints=raw_constraints_result)
    searchable_activities_json = _parse_json_if_possible(searchable_activities)

    skip_confirmation = os.environ.get("SKIP_CONFIRMATION", "").lower() in ("1", "true", "yes")
    if not skip_confirmation and "raw_text" not in searchable_activities_json:
        searchable_activities_json = run_constraint_confirmation(searchable_activities_json)
        searchable_activities = json.dumps(searchable_activities_json, ensure_ascii=False, indent=2)

    _write_text(run_dir / "02_extract_searchable_activities_output.raw.json", searchable_activities)
    _write_json(
        run_dir / "02_extract_searchable_activities_output.json",
        searchable_activities_json,
    )

    print("🔍 [MCP Tool] 待搜尋的模糊行程：")
    print(searchable_activities)
    print("="*20)

    try:
        anchor_context = resolve_anchor_context(
            planner_result,
            searchable_activities_json,
            destination_hint=destination_hint,
            start_anchor=start_anchor,
            end_anchor=end_anchor,
        )
        anchor_paths = save_anchor_context_output(anchor_context, run_dir)

        print("⚓ [AnchorResolver] 模糊 slot 前後 anchor：")
        print(json.dumps(anchor_context, ensure_ascii=False, indent=2))
        print(json.dumps(anchor_paths, ensure_ascii=False, indent=2))
        print("="*20)

        place_candidates = search_places_for_pending_searches(
            searchable_activities,
            destination_hint=destination_hint,
            max_queries_per_slot=4,
            max_results_per_query=6,
            trip_context=planner_result.get("trip_context"),
            anchor_context=anchor_context,
        )
        _write_json(run_dir / "04_places_search_agent_output.json", place_candidates)

        print("📍 [PlacesSearchAgent] Google Places 候選池：")
        print(json.dumps(place_candidates, ensure_ascii=False, indent=2))
        print("="*20)

        scoring_result = score_candidate_slots(
            place_candidates,
            top_n=2,
            run_id=run_dir.name,
            anchor_context=anchor_context,
        )
        scoring_paths = save_scoring_outputs(scoring_result, run_dir)

        print("🏅 [CandidateScorer] 給第六步 LLM 的 Top 2 候選：")
        print(json.dumps(scoring_result["llm_input"], ensure_ascii=False, indent=2))
        print("📦 [CandidateScorer] 已保留其他候選：")
        print(json.dumps(scoring_paths, ensure_ascii=False, indent=2))
        print("="*20)

        final_result = generate_final_itinerary(scoring_result["llm_input"])
        final_paths = save_json_output(
            final_result,
            run_dir,
            "06_final_itinerary_output.json",
            "final_itinerary",
        )

        print("🎯 [FinalItineraryAgent] 最終定案行程：")
        print(json.dumps(final_result.get("decision_summary", {}), ensure_ascii=False, indent=2))
        print(json.dumps(final_paths, ensure_ascii=False, indent=2))
        print("="*20)
    except RuntimeError as exc:
        _write_json(run_dir / "04_places_search_agent_error.json", {"error": str(exc)})
        print("⚠️ [PlacesSearchAgent] 尚未完成 Google Places API 設定：")
        print(str(exc))

if __name__ == "__main__":
    args = list(sys.argv[1:])
    if "--skip-confirmation" in args:
        os.environ["SKIP_CONFIRMATION"] = "1"
        args.remove("--skip-confirmation")
    query = " ".join(args).strip() or os.getenv("TRAVEL_TEST_QUERY", "去台南兩天一夜")
    asyncio.run(run_multi_agent_flow(query))
