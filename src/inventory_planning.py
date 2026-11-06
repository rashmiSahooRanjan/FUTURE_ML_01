"""Inventory planning calculations (Pages 8 and Module 12-D).

The Kaggle dataset has no stock levels, so every stock value is user-supplied.
Demand figures come from a forecast (or any daily demand estimate the user enters).

Definitions used below:
* Expected lead-time demand = average daily demand x supplier lead time (days).
* Reorder point = expected lead-time demand + safety stock.
* Target stock level = forecast demand over the planning horizon + safety stock.
* Recommended order quantity = max(0, max(target level, minimum stock) - current stock).
* Estimated shortage = max(0, expected lead-time demand - current stock)
  (units of demand that would go unmet before a new order could arrive).
* Days of cover = current stock / average daily demand.
* Overstock risk is raised when days of cover exceed the configured threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class InventoryInputs:
    current_stock: float
    avg_daily_demand: float
    lead_time_days: int
    safety_stock: float
    min_stock_level: float
    planning_horizon_days: int = 30
    overstock_days_threshold: int = 60


def _validate(inputs: InventoryInputs) -> None:
    numeric = {
        "current_stock": inputs.current_stock,
        "avg_daily_demand": inputs.avg_daily_demand,
        "safety_stock": inputs.safety_stock,
        "min_stock_level": inputs.min_stock_level,
    }
    for name, value in numeric.items():
        if value is None or (isinstance(value, float) and math.isnan(value)):
            raise ValueError(f"{name} must be a number.")
        if value < 0:
            raise ValueError(f"{name} cannot be negative.")
    if inputs.lead_time_days < 0:
        raise ValueError("lead_time_days cannot be negative.")
    if inputs.planning_horizon_days <= 0:
        raise ValueError("planning_horizon_days must be positive.")
    if inputs.overstock_days_threshold <= 0:
        raise ValueError("overstock_days_threshold must be positive.")


def calculate_inventory_plan(inputs: InventoryInputs) -> dict:
    """Return every inventory figure plus a plain-language explanation of each one."""
    _validate(inputs)

    daily = inputs.avg_daily_demand
    lead_time_demand = daily * inputs.lead_time_days
    reorder_point = lead_time_demand + inputs.safety_stock
    horizon_demand = daily * inputs.planning_horizon_days
    target_level = horizon_demand + inputs.safety_stock
    order_qty = max(0.0, max(target_level, inputs.min_stock_level) - inputs.current_stock)
    shortage = max(0.0, lead_time_demand - inputs.current_stock)
    days_of_cover = math.inf if daily == 0 else inputs.current_stock / daily
    overstock = days_of_cover > inputs.overstock_days_threshold

    if inputs.current_stock <= reorder_point:
        action = "Reorder now"
    elif overstock:
        action = "Overstock: pause purchasing"
    else:
        action = "Stock is adequate"

    return {
        "expected_lead_time_demand": round(lead_time_demand, 2),
        "reorder_point": round(reorder_point, 2),
        "target_stock_level": round(target_level, 2),
        "recommended_order_quantity": round(order_qty, 2),
        "estimated_stock_shortage": round(shortage, 2),
        "days_of_cover": None if math.isinf(days_of_cover) else round(days_of_cover, 2),
        "overstock_risk": bool(overstock),
        "action": action,
        "inputs": asdict(inputs),
    }


def explain_plan(plan: dict) -> list[str]:
    """Return short sentences that describe the plan for the user."""
    lines = [
        f"Expected demand during the {plan['inputs']['lead_time_days']}-day lead time is "
        f"{plan['expected_lead_time_demand']:,.1f} units.",
        f"Reorder when stock falls to {plan['reorder_point']:,.1f} units (lead-time demand plus safety stock).",
    ]
    if plan["recommended_order_quantity"] > 0:
        lines.append(f"Order about {plan['recommended_order_quantity']:,.1f} units to reach the target level.")
    else:
        lines.append("No order is needed to reach the target level.")
    if plan["estimated_stock_shortage"] > 0:
        lines.append(f"Stock would run short by about {plan['estimated_stock_shortage']:,.1f} units before a new delivery.")
    if plan["overstock_risk"]:
        lines.append(f"Current stock covers more than {plan['inputs']['overstock_days_threshold']} days, so there is overstock risk.")
    return lines
