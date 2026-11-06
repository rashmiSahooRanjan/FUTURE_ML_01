"""Inventory calculation tests with hand-checked results."""
import math

import pytest

from src.inventory_planning import InventoryInputs, calculate_inventory_plan, explain_plan


def _inputs(**overrides):
    base = dict(current_stock=500, avg_daily_demand=40, lead_time_days=7, safety_stock=60,
                min_stock_level=100, planning_horizon_days=30, overstock_days_threshold=60)
    base.update(overrides)
    return InventoryInputs(**base)


def test_reorder_point_and_lead_time_demand():
    plan = calculate_inventory_plan(_inputs())
    assert plan["expected_lead_time_demand"] == 280.0        # 40 x 7
    assert plan["reorder_point"] == 340.0                    # 280 + 60


def test_target_level_and_order_quantity():
    plan = calculate_inventory_plan(_inputs())
    assert plan["target_stock_level"] == 1260.0              # 40 x 30 + 60
    assert plan["recommended_order_quantity"] == 760.0       # 1260 - 500


def test_shortage_when_stock_below_lead_time_demand():
    plan = calculate_inventory_plan(_inputs(current_stock=100))
    assert plan["estimated_stock_shortage"] == 180.0         # 280 - 100
    assert plan["action"] == "Reorder now"


def test_no_shortage_and_no_order_when_stock_is_high():
    plan = calculate_inventory_plan(_inputs(current_stock=5000, avg_daily_demand=10,
                                            planning_horizon_days=7, safety_stock=0))
    assert plan["estimated_stock_shortage"] == 0.0
    assert plan["recommended_order_quantity"] == 0.0
    assert plan["overstock_risk"] is True
    assert plan["action"] == "Overstock: pause purchasing"


def test_zero_demand_gives_infinite_cover_reported_as_none():
    plan = calculate_inventory_plan(_inputs(avg_daily_demand=0))
    assert plan["days_of_cover"] is None
    assert plan["reorder_point"] == 60.0
    # Stock that never sells is overstock: infinite days of cover exceeds the threshold.
    assert plan["overstock_risk"] is True


def test_minimum_stock_level_drives_order_when_higher_than_target():
    plan = calculate_inventory_plan(_inputs(avg_daily_demand=1, planning_horizon_days=7,
                                            safety_stock=0, min_stock_level=800, current_stock=500))
    assert plan["recommended_order_quantity"] == 300.0       # max(7, 800) - 500


@pytest.mark.parametrize("bad", [
    {"current_stock": -1}, {"avg_daily_demand": -2}, {"lead_time_days": -1},
    {"safety_stock": -3}, {"planning_horizon_days": 0},
])
def test_invalid_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        calculate_inventory_plan(_inputs(**bad))


def test_nan_input_is_rejected():
    with pytest.raises(ValueError):
        calculate_inventory_plan(_inputs(current_stock=math.nan))


def test_explanation_mentions_reorder_point():
    lines = explain_plan(calculate_inventory_plan(_inputs()))
    assert any("Reorder when stock falls to" in line for line in lines)
