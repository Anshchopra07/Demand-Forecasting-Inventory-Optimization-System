import numpy as np
import pandas as pd
from datetime import datetime, timedelta

PRODUCTS = [
    {"product_id": "SKU-1001", "name": "Wireless Earbuds",        "category": "Electronics",  "base_demand": 220, "lead_time_days": 14, "unit_cost": 18.5},
    {"product_id": "SKU-1002", "name": "Stainless Steel Bottle",  "category": "Home",         "base_demand": 150, "lead_time_days": 10, "unit_cost": 6.2},
    {"product_id": "SKU-1003", "name": "Yoga Mat",                "category": "Fitness",      "base_demand": 95,  "lead_time_days": 12, "unit_cost": 9.0},
    {"product_id": "SKU-1004", "name": "Instant Coffee Jar",      "category": "Grocery",      "base_demand": 340, "lead_time_days": 7,  "unit_cost": 3.4},
    {"product_id": "SKU-1005", "name": "LED Desk Lamp",           "category": "Home",         "base_demand": 110, "lead_time_days": 15, "unit_cost": 12.8},
    {"product_id": "SKU-1006", "name": "Running Shoes",           "category": "Fitness",      "base_demand": 130, "lead_time_days": 18, "unit_cost": 24.0},
    {"product_id": "SKU-1007", "name": "Bluetooth Speaker",       "category": "Electronics",  "base_demand": 175, "lead_time_days": 14, "unit_cost": 21.0},
    {"product_id": "SKU-1008", "name": "Notebook Pack (3-pc)",    "category": "Stationery",   "base_demand": 260, "lead_time_days": 6,  "unit_cost": 2.1},
]

def _seasonal_multiplier(week_of_year: int, category: str) -> float:
    holiday_bump = 1.0
    if 46 <= week_of_year <= 52:
        holiday_bump = 1.55
    elif 1 <= week_of_year <= 2:
        holiday_bump = 1.15

    cat_wave = 1.0
    if category == "Fitness":
        cat_wave = 1.25 if week_of_year <= 8 else 1.0
    elif category == "Electronics":
        cat_wave = 1.2 if 35 <= week_of_year <= 45 else 1.0
    elif category == "Stationery":
        cat_wave = 1.4 if 22 <= week_of_year <= 26 else 1.0

    return holiday_bump * cat_wave


def load_sales_data(weeks_of_history: int = 104, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime.today() - timedelta(weeks=weeks_of_history)
    rows = []

    for p in PRODUCTS:
        trend_rate = rng.uniform(0.0015, 0.004)
        for w in range(weeks_of_history):
            date = start + timedelta(weeks=w)
            week_of_year = date.isocalendar().week
            seasonal = _seasonal_multiplier(week_of_year, p["category"])
            trend = 1 + trend_rate * w
            noise = rng.normal(1.0, 0.12)
            units = max(0, p["base_demand"] * seasonal * trend * noise)
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "product_id": p["product_id"],
                "name": p["name"],
                "category": p["category"],
                "units_sold": round(units),
                "is_holiday": 1 if 46 <= week_of_year <= 52 else 0,
                "week_of_year": week_of_year,
            })

    return pd.DataFrame(rows)


def get_product_meta():
    return {p["product_id"]: p for p in PRODUCTS}
