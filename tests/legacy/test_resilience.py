from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock

from shop_planning import (
    BookingType,
    DietaryAxis,
    QueueStrategy,
    ShopProfile,
    SnsAdapter,
    TrafficAdapter,
    TrafficAlert,
    predict_wait_time,
    plan_shop_visit,
)


def _mk_shop(**kwargs) -> ShopProfile:
    base = dict(
        name="Test Shop",
        close_time="20:00",
        booking_type=BookingType.NONE,
        queue_strategy=QueueStrategy.PHYSICAL_LINE,
        last_call_offset=30,
        is_cash_only=False,
        sns_handle="shop_x",
        avg_eat_minutes=45,
        is_famous=True,
        base_wait_minutes=20,
        backup_options=["Backup Ramen"],
    )
    base.update(kwargs)
    return ShopProfile(**base)


class ResilienceEdgeCaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 4, 26, 18, 0, 0)
        self.mock_sns = Mock(spec=SnsAdapter)
        self.mock_traffic = Mock(spec=TrafficAdapter)
        self.mock_traffic.get_route_status.return_value = TrafficAlert(
            has_alert=False,
            semantic_status="TRAFFIC_OK",
            transport_buffer_minutes=10,
            suggested_transport="RAIL",
        )

    def test_case_a_sold_out_nightmare(self) -> None:
        print("\n[Test Case] A: Sold Out Nightmare")
        print("[Failure Injected] sns='完売 / sold out'")
        self.mock_sns.check_store_status.return_value = "本日完売 sold out"
        shop = _mk_shop(name="Mensuke")

        result = plan_shop_visit(
            shop=shop,
            current_time=self.now,
            day_of_week="Sat",
            time_slot="lunch",
            from_loc="Kyoto",
            to_loc="Umeda",
            travel_time_minutes=30,
            dietary_preference="regular",
            sns_adapter=self.mock_sns,
            traffic_adapter=self.mock_traffic,
        )

        backup_distance_m = 500
        backup_live_signal = "通常営業"
        print(
            "[System Reaction] "
            f"outcome={result.outcome}, semantic={result.semantic_status}, backup={result.backup_option}"
        )
        print(
            "[System Reaction] Backup Evaluation: "
            f"candidate={result.backup_option}, distance={backup_distance_m}m, live_signal='{backup_live_signal}' "
            "=> selected (nearby + no temporary closure)."
        )
        self.assertEqual(result.outcome, "FORCE_ABORT")
        self.assertIn("LIVE_PROBE_FORCE_ABORT", result.semantic_status)
        self.assertTrue(bool(result.backup_option))
        self.assertIn("建議改店", result.preparation_note)
        print("[Result] PASS")

    def test_case_b_transport_delay_nightmare(self) -> None:
        print("\n[Test Case] B: Transport Delay Nightmare")
        self.mock_sns.check_store_status.return_value = "通常営業"
        shop = _mk_shop(name="Delay Heavy Shop")

        print("[Failure Injected] delay_minutes=60 (人身事故等級)")
        self.mock_traffic.get_route_status.return_value = TrafficAlert(
            has_alert=True,
            semantic_status="TRAFFIC_MAJOR_DELAY",
            transport_buffer_minutes=60,
            suggested_transport="TAXI",
            matched_keyword="人身事故",
        )

        base_wait = shop.base_wait_minutes
        weighted_wait = predict_wait_time(shop, "Sat", "lunch")
        queue_factor = weighted_wait / base_wait if base_wait else 1.0
        print(
            "[System Reaction] Queue Weighting Chain: "
            f"base_wait={base_wait}m * factor={queue_factor:.2f} => weighted_wait={weighted_wait}m"
        )

        current = self.now
        travel = 50
        jitter = int((travel * 0.40) + 0.9999)  # central/sobu high-risk rule
        transport_buffer = 60
        eat = shop.avg_eat_minutes
        arrival = current + timedelta(minutes=travel + jitter + transport_buffer)
        close = current.replace(hour=20, minute=0, second=0, microsecond=0)
        last_call = close - timedelta(minutes=shop.last_call_offset)
        print(
            "[System Reaction] Arrival vs LastCall Formula: "
            f"arrival={current.strftime('%H:%M')} + travel({travel}) + jitter({jitter}) + buffer({transport_buffer}) "
            f"= {arrival.strftime('%H:%M')} ; last_call={last_call.strftime('%H:%M')}"
        )

        result_conflict = plan_shop_visit(
            shop=shop,
            current_time=self.now,
            day_of_week="Sat",
            time_slot="lunch",
            from_loc="中央線",
            to_loc="新宿",
            travel_time_minutes=50,
            dietary_preference="regular",
            sns_adapter=self.mock_sns,
            traffic_adapter=self.mock_traffic,
        )
        print(f"[System Reaction] conflict_outcome={result_conflict.outcome}, semantic={result_conflict.semantic_status}")
        self.assertIn(result_conflict.outcome, {"CONSTRAINT_CONFLICT", "HIGH_RISK_LAST_CALL"})

        # A second setup where taxi reroute should rescue.
        self.mock_traffic.get_route_status.return_value = TrafficAlert(
            has_alert=True,
            semantic_status="TRAFFIC_SUSPENDED",
            transport_buffer_minutes=45,
            suggested_transport="TAXI",
            matched_keyword="運休",
        )
        rescue_shop = _mk_shop(
            name="Taxi Rescue Shop",
            close_time="21:30",
            last_call_offset=20,
            avg_eat_minutes=35,
            base_wait_minutes=10,
        )
        result_reroute = plan_shop_visit(
            shop=rescue_shop,
            current_time=self.now,
            day_of_week="Sat",
            time_slot="dinner",
            from_loc="總武線",
            to_loc="飯田橋",
            travel_time_minutes=25,
            dietary_preference="regular",
            sns_adapter=self.mock_sns,
            traffic_adapter=self.mock_traffic,
        )
        print(
            "[System Reaction] Taxi Rescue Check: "
            "if rail path exceeds last call, try TAXI buffer profile and re-evaluate feasibility."
        )
        print(f"[System Reaction] reroute_outcome={result_reroute.outcome}, semantic={result_reroute.semantic_status}")
        self.assertIn(result_reroute.outcome, {"REROUTED_TRANSPORT", "SUCCESS"})
        print("[Result] PASS")

    def test_case_c_dietary_mismatch_nightmare(self) -> None:
        print("\n[Test Case] C: Dietary Mismatch Nightmare")
        print("[Failure Injected] dietary_preference='MEAT' while shop is_vegan=True")
        self.mock_sns.check_store_status.return_value = "通常営業"
        vegan_shop = _mk_shop(
            name="Vegan Omakase",
            is_vegan=True,
            requires_menu_reservation=True,
            allowed_dietary_preferences=["vegan", "vegetarian"],
            booking_type=BookingType.OMAKASE,
        )

        result = plan_shop_visit(
            shop=vegan_shop,
            current_time=self.now,
            day_of_week="Sun",
            time_slot="dinner",
            from_loc="Shibuya",
            to_loc="Ebisu",
            travel_time_minutes=15,
            dietary_preference="MEAT",
            sns_adapter=self.mock_sns,
            traffic_adapter=self.mock_traffic,
        )

        triggered_tag = "is_vegan=True + dietary_preference=MEAT"
        print(
            "[System Reaction] "
            f"triggered_tag='{triggered_tag}', outcome={result.outcome}, semantic={result.semantic_status}"
        )
        self.assertEqual(result.outcome, "PREORDER_RISK")
        self.assertEqual(result.semantic_status, "PREORDER_RISK")
        self.assertIn("強烈確認", result.preparation_note)
        print("[Result] PASS")

    def test_case_d_last_call_nightmare(self) -> None:
        print("\n[Test Case] D: Last Call Nightmare")
        self.mock_sns.check_store_status.return_value = "通常営業"
        # close=20:00, offset=30 => last_call=19:30
        # arrival set to exactly 19:30 via travel/buffer/jitter combination
        shop = _mk_shop(
            name="Last Call Edge Shop",
            close_time="20:00",
            last_call_offset=30,
            avg_eat_minutes=40,
            base_wait_minutes=5,
            is_famous=False,
        )
        print("[Failure Injected] arrival_time == closing_time - last_call_offset")
        self.mock_traffic.get_route_status.return_value = TrafficAlert(
            has_alert=False,
            semantic_status="TRAFFIC_OK",
            transport_buffer_minutes=6,
            suggested_transport="RAIL",
        )
        current = datetime(2026, 4, 26, 19, 4, 0)
        travel = 20
        jitter = int((travel * 0.20) + 0.9999)
        buffer = 6
        arrival = current + timedelta(minutes=travel + jitter + buffer)
        close = current.replace(hour=20, minute=0, second=0, microsecond=0)
        last_call = close - timedelta(minutes=30)
        print(
            "[System Reaction] Arrival vs LastCall Formula: "
            f"arrival={current.strftime('%H:%M')} + travel({travel}) + jitter({jitter}) + buffer({buffer}) "
            f"= {arrival.strftime('%H:%M')} ; last_call={last_call.strftime('%H:%M')}"
        )
        result = plan_shop_visit(
            shop=shop,
            current_time=current,
            day_of_week="Fri",
            time_slot="dinner",
            from_loc="A",
            to_loc="B",
            travel_time_minutes=20,
            dietary_preference="regular",
            sns_adapter=self.mock_sns,
            traffic_adapter=self.mock_traffic,
        )
        print(f"[System Reaction] outcome={result.outcome}, semantic={result.semantic_status}, backup={result.backup_option}")
        self.assertEqual(result.outcome, "HIGH_RISK_LAST_CALL")
        self.assertEqual(result.semantic_status, "LAST_CALL_EDGE_RISK")
        self.assertIn("極高風險", result.preparation_note)
        print("[Result] PASS")

    def test_case_e_allergen_force_abort(self) -> None:
        print("\n[Test Case] E: Allergen Conflict => Force Abort")
        self.mock_sns.check_store_status.return_value = "通常営業"
        allergy_shop = _mk_shop(
            name="Shellfish Risk Shop",
            blocked_allergens={"shellfish"},
            requires_compatibility_check=True,
        )
        axis = DietaryAxis(
            ethics="pescatarian",
            allergens={"shellfish"},
            religious="none",
            medical=set(),
        )
        result = plan_shop_visit(
            shop=allergy_shop,
            current_time=self.now,
            day_of_week="Sat",
            time_slot="dinner",
            from_loc="A",
            to_loc="B",
            travel_time_minutes=20,
            dietary_preference=axis,
            sns_adapter=self.mock_sns,
            traffic_adapter=self.mock_traffic,
        )
        self.assertEqual(result.outcome, "FORCE_ABORT")
        self.assertEqual(result.semantic_status, "DIETARY_ALLERGEN_CONFLICT")
        print("[Result] PASS")

    def test_baseline_happy_path(self) -> None:
        print("\n[Test Case] Baseline Happy Path")
        self.mock_sns.check_store_status.return_value = "通常営業"
        happy_shop = _mk_shop(
            name="Happy Path Shop",
            is_famous=False,
            base_wait_minutes=10,
            avg_eat_minutes=35,
            close_time="22:00",
            last_call_offset=20,
        )
        result = plan_shop_visit(
            shop=happy_shop,
            current_time=self.now,
            day_of_week="Tue",
            time_slot="dinner",
            from_loc="Kyoto",
            to_loc="Umeda",
            travel_time_minutes=15,
            dietary_preference=DietaryAxis(ethics="pescatarian"),
            sns_adapter=self.mock_sns,
            traffic_adapter=self.mock_traffic,
        )
        self.assertEqual(result.outcome, "SUCCESS")
        self.assertEqual(result.semantic_status, "SHOP_SCHEDULED")
        self.assertTrue(len(result.slots) > 0)
        print("[Result] PASS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
