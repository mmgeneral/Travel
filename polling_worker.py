from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from agent import _build_shop_catalog
from shop_planning import XTimelineSnsProvider, fetch_live_status, ProbeOutcome


def run_once(output_json: str | None = None) -> list[dict]:
    shops = _build_shop_catalog()
    sns = XTimelineSnsProvider()
    alerts: list[dict] = []
    now = datetime.now(ZoneInfo("Asia/Taipei")).isoformat()
    for shop in shops:
        probe = fetch_live_status(shop, sns)
        if probe.outcome == ProbeOutcome.FORCE_ABORT:
            alerts.append(
                {
                    "timestamp": now,
                    "shop_name": shop.name,
                    "sns_handle": shop.sns_handle,
                    "semantic_status": probe.semantic_status,
                    "matched_keyword": probe.matched_keyword,
                }
            )
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(alerts, ensure_ascii=False, indent=2), encoding="utf-8")
    return alerts


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-poll SNS live status for itinerary shops.")
    parser.add_argument("--output-json", default="", help="Optional path to write matched alerts")
    args = parser.parse_args()
    alerts = run_once(args.output_json or None)
    if not alerts:
        print("[polling_worker] no closure/sold-out signals detected")
        return
    print(f"[polling_worker] alerts={len(alerts)}")
    for a in alerts:
        print(
            f"- {a['shop_name']} @{a['sns_handle']} "
            f"{a['semantic_status']} keyword={a['matched_keyword']}"
        )


if __name__ == "__main__":
    main()
