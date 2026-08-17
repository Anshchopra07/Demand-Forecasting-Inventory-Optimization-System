import numpy as np

Z_SCORE_MAP = {
    0.90: 1.28,
    0.95: 1.65,
    0.99: 2.33
}
Z_SCORE_95 = 1.65
ORDER_COST = 45.0
HOLDING_COST_RATE = 0.22


def compute_inventory_plan(weekly_units, lead_time_days, unit_cost, current_stock, z_score=Z_SCORE_95, order_cost=ORDER_COST, holding_cost_rate=HOLDING_COST_RATE):
    daily_demand = np.array(weekly_units) / 7
    avg_daily = float(np.mean(daily_demand))
    std_daily = float(np.std(daily_demand))

    safety_stock = z_score * std_daily * np.sqrt(lead_time_days)
    reorder_point = avg_daily * lead_time_days + safety_stock

    annual_demand = avg_daily * 365
    holding_cost_per_unit = unit_cost * holding_cost_rate
    eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost_per_unit) if holding_cost_per_unit > 0 else 0

    if current_stock <= safety_stock:
        status = "critical"
    elif current_stock <= reorder_point:
        status = "reorder"
    else:
        status = "healthy"

    days_of_supply = round(current_stock / avg_daily, 1) if avg_daily > 0 else None

    return {
        "avg_daily_demand": round(avg_daily, 1),
        "safety_stock": round(safety_stock),
        "reorder_point": round(reorder_point),
        "eoq": round(eoq),
        "current_stock": round(current_stock),
        "days_of_supply": days_of_supply,
        "status": status,
        "holding_cost_annual": round(current_stock * holding_cost_per_unit, 2),
        "z_score_used": z_score,
        "lead_time_days": lead_time_days,
        "order_cost": order_cost,
        "holding_cost_rate": holding_cost_rate
    }
