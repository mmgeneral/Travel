from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List


EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVALUATION_DIR.parent
TRAVELEVAL_QUERY_DIR = Path("/private/tmp/TravelEval/environment/data/queries")

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from PlannerAgent import generate_travel_itinerary_json  # noqa: E402
from Mcp import extract_searchable_activities  # noqa: E402
from PlacesSearchAgent import search_places_for_pending_searches  # noqa: E402
from LodgingSearchAgent import build_lodging_search_context, search_lodging_serpapi  # noqa: E402
from CandidateScorer import build_combined_candidate_package, save_combined_candidate_outputs  # noqa: E402
from FinalItineraryAgent import generate_final_itinerary  # noqa: E402
from traveleval_offline_adapter import run as run_evaluation  # noqa: E402


DEFAULT_CASE_IDS = ["T0001", "T0201", "T0601"]

BOUNDED_PARTIAL_METRICS = [
    "CCD", "FAR", "VROH", "PD",
    "BCS", "TCS", "HCS", "PAS-A", "PAS-T", "PAS-C", "TTCS",
    "STR", "DTU", "OTU",
    "EDI", "ADS", "AQE",
]
LOWER_IS_BETTER = {"CCD", "FAR", "VROH", "PD"}
KEY_REPORT_METRICS = ["CCD", "FAR", "VROH", "BCS", "TTCS", "BE", "EDI", "ADS", "AQE", "Profit"]
SECRET_ENV_NAMES = ["OPENAI_API_KEY", "GOOGLE_PLACES_API_KEY", "SERPAPI_API_KEY", "AMAP_API_KEY"]

PAPER_BASELINES = {
    "Claude Code": {
        "CCD": 0.0407, "FAR": 0.0, "VROH": 0.0235, "PD": 0.2148,
        "BCS": 0.5703, "TCS": 1.0, "HCS": 0.5361, "PAS-A": 0.4943,
        "PAS-T": 0.8479, "PAS-C": 0.5399, "TTCS": 0.0342,
        "STR": 0.6958, "DTU": 0.1955, "OTU": 0.0503,
        "EDI": 0.5966, "ADS": 0.5375, "AQE": 0.8033,
        "BE": 13.212, "Profit": 5.6632,
    },
    "Approach A": {
        "CCD": 0.0005, "FAR": 0.0, "VROH": 0.2504, "PD": 0.0,
        "BCS": 0.6273, "TCS": 1.0, "HCS": 1.0, "PAS-A": 0.6681,
        "PAS-T": 0.7524, "PAS-C": 0.9036, "TTCS": 0.0096,
        "STR": 0.4979, "DTU": 0.1212, "OTU": 0.0608,
        "EDI": 0.6547, "ADS": 0.7649, "AQE": 0.9882,
        "BE": 16.274, "Profit": 5.9979,
    },
    "Approach B": {
        "CCD": 0.0013, "FAR": 0.0, "VROH": 0.0046, "PD": 0.0,
        "BCS": 0.9217, "TCS": 1.0, "HCS": 1.0, "PAS-A": 0.6748,
        "PAS-T": 0.7826, "PAS-C": 0.5652, "TTCS": 0.4487,
        "STR": 0.7677, "DTU": 0.3029, "OTU": 0.0649,
        "EDI": 0.5829, "ADS": 0.5129, "AQE": 0.7479,
        "BE": 22.192, "Profit": 5.5015,
    },
}


def save_json(data: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def redact_secrets(text: str) -> str:
    redacted = str(text)
    for env_name in SECRET_ENV_NAMES:
        secret = os.getenv(env_name)
        if secret:
            redacted = redacted.replace(secret, "<REDACTED>")
    redacted = re.sub(r"(api_key=)[^&\"'\s]+", r"\1<REDACTED>", redacted)
    return redacted


def run_stage(case_dir: Path, stage_name: str, func, *args, **kwargs):
    log_dir = case_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage_name}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                return func(*args, **kwargs)
        except Exception:
            log_file.write("\n\n===== TRACEBACK =====\n")
            log_file.write(redact_secrets(traceback.format_exc()))
            raise


def shift_past_lodging_dates_for_live_api(ctx: Dict[str, Any]) -> Dict[str, Any]:
    checkin_text = ctx.get("checkin_date")
    checkout_text = ctx.get("checkout_date")
    if not checkin_text or not checkout_text:
        return ctx

    try:
        checkin = date.fromisoformat(checkin_text)
        checkout = date.fromisoformat(checkout_text)
    except ValueError:
        return ctx

    today = date.today()
    if checkin > today:
        return ctx

    nights = ctx.get("nights")
    try:
        nights = int(nights)
    except (TypeError, ValueError):
        nights = max(1, (checkout - checkin).days)

    shifted = dict(ctx)
    new_checkin = today + timedelta(days=3)
    new_checkout = new_checkin + timedelta(days=max(1, nights))
    shifted["checkin_date"] = new_checkin.isoformat()
    shifted["checkout_date"] = new_checkout.isoformat()
    shifted["live_price_date_adjustment"] = {
        "reason": "TravelEval case date is in the past, but Google Hotels live price API requires a future check-in date.",
        "original_checkin_date": checkin_text,
        "original_checkout_date": checkout_text,
        "live_api_checkin_date": shifted["checkin_date"],
        "live_api_checkout_date": shifted["checkout_date"],
    }
    return shifted


def clamp01(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def mean_or_none(values: List[float]) -> float | None:
    clean = [value for value in values if isinstance(value, (int, float))]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 4)


def normalized_metric_score(metric: str, value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if metric in LOWER_IS_BETTER:
        return clamp01(1 - numeric)
    return clamp01(numeric)


def rows_by_metric(table: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {row["metric"]: row for row in table.get("computed_metrics", [])}


def summarize_partial_table(table: Dict[str, Any]) -> Dict[str, Any]:
    rows = rows_by_metric(table)
    normalized = {
        metric: normalized_metric_score(metric, (rows.get(metric) or {}).get("value"))
        for metric in BOUNDED_PARTIAL_METRICS
    }
    key_values = {
        metric: (rows.get(metric) or {}).get("value")
        for metric in KEY_REPORT_METRICS
    }
    return {
        "bounded_partial_average": mean_or_none([value for value in normalized.values() if value is not None]),
        "normalized_metrics": normalized,
        "key_values": key_values,
    }


def summarize_paper_baselines() -> Dict[str, Dict[str, Any]]:
    output = {}
    for name, values in PAPER_BASELINES.items():
        normalized = {
            metric: normalized_metric_score(metric, values.get(metric))
            for metric in BOUNDED_PARTIAL_METRICS
        }
        output[name] = {
            "bounded_partial_average": mean_or_none([value for value in normalized.values() if value is not None]),
            "raw_values": values,
            "normalized_metrics": normalized,
        }
    return output


def build_comparison_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    case_summaries = []
    for result in results:
        table = result.get("partial_metrics_table")
        if not table:
            continue
        summary = summarize_partial_table(table)
        case_summaries.append({
            "uid": result.get("uid"),
            "difficulty": result.get("difficulty"),
            "case_dir": result.get("case_dir"),
            **summary,
        })

    system_average = mean_or_none([
        item["bounded_partial_average"]
        for item in case_summaries
        if item.get("bounded_partial_average") is not None
    ])
    baselines = summarize_paper_baselines()

    return {
        "scope": "selected TravelEval cases and optional custom prompts",
        "important_note": "This is a no-Gaode partial comparison. FRTC/SSR/CSM and official route-distance metrics are skipped.",
        "bounded_partial_metrics": BOUNDED_PARTIAL_METRICS,
        "lower_is_better": sorted(LOWER_IS_BETTER),
        "your_pipeline": {
            "average_bounded_partial_score": system_average,
            "case_count": len(case_summaries),
            "cases": case_summaries,
        },
        "paper_baselines": baselines,
    }


def render_comparison_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# TravelEval Partial Comparison Without Gaode",
        "",
        summary["important_note"],
        "",
        "## Selected Cases",
        "",
        "| Difficulty | UID | Bounded partial avg | BCS | CCD | FAR | VROH | TTCS | BE | EDI | ADS | AQE | Profit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in summary["your_pipeline"]["cases"]:
        values = case["key_values"]
        lines.append(
            "| {difficulty} | {uid} | {avg} | {BCS} | {CCD} | {FAR} | {VROH} | {TTCS} | {BE} | {EDI} | {ADS} | {AQE} | {Profit} |".format(
                difficulty=case["difficulty"],
                uid=case["uid"],
                avg=case.get("bounded_partial_average"),
                BCS=values.get("BCS"),
                CCD=values.get("CCD"),
                FAR=values.get("FAR"),
                VROH=values.get("VROH"),
                TTCS=values.get("TTCS"),
                BE=values.get("BE"),
                EDI=values.get("EDI"),
                ADS=values.get("ADS"),
                AQE=values.get("AQE"),
                Profit=values.get("Profit"),
            )
        )

    lines.extend([
        "",
        "## Paper Comparison",
        "",
        "| System | Bounded partial avg | BE | Profit |",
        "|---|---:|---:|---:|",
        f"| Your pipeline, {summary['your_pipeline']['case_count']} run cases | {summary['your_pipeline']['average_bounded_partial_score']} | - | - |",
    ])
    for name, baseline in summary["paper_baselines"].items():
        raw = baseline["raw_values"]
        lines.append(
            f"| {name} | {baseline['bounded_partial_average']} | {raw.get('BE')} | {raw.get('Profit')} |"
        )

    lines.extend([
        "",
        "Bounded partial avg converts lower-is-better metrics with `1 - value`, keeps higher-is-better metrics as-is, clips to 0..1, and excludes Gaode-required metrics plus unbounded cost decomposition fields.",
    ])
    return "\n".join(lines) + "\n"


def load_queries() -> Dict[str, Dict[str, Any]]:
    cases = {}
    for difficulty in ("easy", "medium", "hard"):
        path = TRAVELEVAL_QUERY_DIR / f"{difficulty}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for case in payload.get("queries", []):
            case = dict(case)
            case["difficulty"] = difficulty
            cases[case["uid"]] = case
    return cases


def case_prompt(case: Dict[str, Any]) -> str:
    text = case.get("nature_language") or case.get("nature_language_en") or ""
    if case.get("is_custom_prompt"):
        return text
    return (
        f"{text}\n"
        f"請保留 TravelEval 測資條件：uid={case.get('uid')}，"
        f"出發城市={case.get('start_city')}，目的地={case.get('target_city')}，"
        f"天數={case.get('days')}，人數={case.get('people_number')}，"
        f"預算={case.get('budget')} 人民幣，日期={case.get('dates')}。"
    )


def run_case(case: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    uid = case["uid"]
    difficulty = case["difficulty"]
    case_dir = run_dir / f"{difficulty}_{uid}"
    case_dir.mkdir(parents=True, exist_ok=True)

    save_json(case, case_dir / "00_traveleval_query.json")
    prompt = case_prompt(case)
    (case_dir / "00_prompt.txt").write_text(prompt, encoding="utf-8")

    planner = run_stage(case_dir, "01_planner", generate_travel_itinerary_json, prompt)
    planner_path = save_json(planner, case_dir / "01_planner_result.json")

    stage1_raw = run_stage(
        case_dir,
        "02_extract_searchable",
        extract_searchable_activities,
        json.dumps(planner, ensure_ascii=False),
    )
    try:
        stage1 = json.loads(stage1_raw)
    except json.JSONDecodeError:
        stage1 = {"raw_text": stage1_raw}
    stage1_path = save_json(stage1, case_dir / "02_stage1_searchable.json")

    places = run_stage(
        case_dir,
        "03_places_search",
        search_places_for_pending_searches,
        stage1,
        destination_hint=(planner.get("trip_context") or {}).get("destination", ""),
        max_queries_per_slot=4,
        max_results_per_query=6,
        expand_queries_with_llm=True,
        trip_context=planner.get("trip_context"),
        itinerary_json=planner,
    )
    places_path = save_json(places, case_dir / "03_places_search.json")

    lodging_ctx = shift_past_lodging_dates_for_live_api(build_lodging_search_context(planner))
    save_json(lodging_ctx, case_dir / "04_lodging_search_context.json")
    lodging = run_stage(case_dir, "05_lodging_search", search_lodging_serpapi, lodging_ctx, max_results=20)
    lodging_path = save_json(lodging, case_dir / "05_lodging_search.json")

    combined = run_stage(
        case_dir,
        "06_candidate_package",
        build_combined_candidate_package,
        planner,
        places,
        lodging,
        place_top_n=5,
        lodging_top_n=8,
        cluster_k=None,
        run_id=f"{difficulty}_{uid}",
    )
    save_json(combined, case_dir / "06_candidate_combined_raw.json")
    candidate_paths = run_stage(
        case_dir,
        "07_save_candidate_outputs",
        save_combined_candidate_outputs,
        combined,
        case_dir,
        "06_candidate_package.json",
        "07_candidate_llm_input.json",
    )

    final_result = run_stage(case_dir, "08_final_itinerary", generate_final_itinerary, combined["llm_input"])
    final_path = save_json(final_result, case_dir / "08_final_itinerary.json")

    eval_dir = case_dir / "09_evaluation"
    eval_result = run_stage(case_dir, "09_evaluation", run_evaluation, final_path, planner_path, eval_dir, f"{difficulty}_{uid}")

    return {
        "uid": uid,
        "difficulty": difficulty,
        "case_dir": str(case_dir),
        "files": {
            "query": str(case_dir / "00_traveleval_query.json"),
            "prompt": str(case_dir / "00_prompt.txt"),
            "planner": str(planner_path),
            "stage1": str(stage1_path),
            "places": str(places_path),
            "lodging": str(lodging_path),
            "candidate_package": candidate_paths.get("combined_candidate"),
            "candidate_llm_input": candidate_paths.get("llm_input"),
            "final": str(final_path),
            "evaluation_score": eval_result["score_path"],
            "partial_metrics_json": eval_result["partial_table_path"],
            "partial_metrics_md": eval_result["partial_markdown_path"],
        },
        "score": eval_result["score"],
        "partial_metrics_table": eval_result["partial_metrics_table"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run corporate_with_gemini pipeline on selected TravelEval cases.")
    parser.add_argument("--case-ids", nargs="*", default=DEFAULT_CASE_IDS)
    parser.add_argument("--custom-prompt", default=None)
    parser.add_argument("--custom-uid", default="custom_prompt")
    parser.add_argument("--custom-difficulty", default="custom")
    parser.add_argument("--run-name", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    cases_by_id = load_queries()
    run_dir = EVALUATION_DIR / "case_runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    selected = []
    for uid in args.case_ids:
        if uid not in cases_by_id:
            raise ValueError(f"Unknown TravelEval uid: {uid}")
        selected.append(cases_by_id[uid])
    if args.custom_prompt:
        selected.append({
            "uid": args.custom_uid,
            "difficulty": args.custom_difficulty,
            "nature_language": args.custom_prompt,
            "is_custom_prompt": True,
        })
    save_json({"case_ids": args.case_ids, "cases": selected}, run_dir / "00_selected_cases.json")

    results: List[Dict[str, Any]] = []
    had_error = False
    for case in selected:
        print("=" * 20 + f" RUN {case['difficulty']} {case['uid']} " + "=" * 20)
        try:
            result = run_case(case, run_dir)
            results.append(result)
            print("saved:", result["case_dir"])
        except Exception as exc:
            error_payload = {
                "uid": case.get("uid"),
                "difficulty": case.get("difficulty"),
                "error_type": type(exc).__name__,
                "error": redact_secrets(str(exc)),
                "traceback": redact_secrets(traceback.format_exc()),
            }
            save_json(error_payload, run_dir / f"{case['difficulty']}_{case['uid']}" / "ERROR.json")
            results.append(error_payload)
            print("ERROR:", {key: error_payload[key] for key in ("uid", "difficulty", "error_type", "error")})
            had_error = True
            break

    comparison = build_comparison_summary(results)
    save_json(comparison, run_dir / "99_comparison_summary.json")
    (run_dir / "99_comparison_summary.md").write_text(render_comparison_markdown(comparison), encoding="utf-8")
    summary_path = save_json({"run_dir": str(run_dir), "results": results, "comparison": comparison}, run_dir / "99_run_summary.json")
    print("=" * 20 + " SUMMARY " + "=" * 20)
    print(summary_path)
    if had_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
