from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from functools import lru_cache
import os

from data_gen import load_sales_data, get_product_meta, _seasonal_multiplier
from forecasting import train_and_forecast
from inventory import compute_inventory_plan, Z_SCORE_MAP

app = FastAPI(title="Demand Forecasting & Inventory Optimization API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_SALES_DF = load_sales_data()
_META = get_product_meta()
_rng = np.random.default_rng(7)
_CURRENT_STOCK = {pid: float(_rng.uniform(0.3, 1.4) * meta["base_demand"] * 2) for pid, meta in _META.items()}


class ProductCreate(BaseModel):
    product_id: Optional[str] = None
    name: str
    category: str
    base_demand: float
    lead_time_days: int
    unit_cost: float
    current_stock: Optional[float] = None


@lru_cache(maxsize=64)
def _get_product_result(product_id: str):
    if product_id not in _META:
        raise HTTPException(status_code=404, detail="Unknown product_id")
    pdf = _SALES_DF[_SALES_DF["product_id"] == product_id]
    fc = train_and_forecast(pdf, horizon_weeks=12)
    meta = _META[product_id]
    plan = compute_inventory_plan(
        weekly_units=[r["units_sold"] for r in fc["forecast"]],
        lead_time_days=meta["lead_time_days"],
        unit_cost=meta["unit_cost"],
        current_stock=_CURRENT_STOCK[product_id],
    )
    return fc, plan, meta


@app.get("/api/products")
def list_products():
    return [
        {
            "product_id": pid,
            "name": m["name"],
            "category": m["category"],
            "base_demand": m["base_demand"],
            "lead_time_days": m["lead_time_days"],
            "unit_cost": m["unit_cost"],
            "current_stock": round(_CURRENT_STOCK[pid])
        } for pid, m in _META.items()
    ]


@app.post("/api/products")
def create_product(item: ProductCreate):
    global _SALES_DF
    pid = item.product_id
    if not pid or not pid.strip():
        pid = f"SKU-{1001 + len(_META)}"
    
    if pid in _META:
        raise HTTPException(status_code=400, detail="Product ID already exists")

    new_meta = {
        "product_id": pid,
        "name": item.name,
        "category": item.category,
        "base_demand": item.base_demand,
        "lead_time_days": item.lead_time_days,
        "unit_cost": item.unit_cost,
    }
    _META[pid] = new_meta

    stock = item.current_stock if item.current_stock is not None else item.base_demand * 1.5
    _CURRENT_STOCK[pid] = float(stock)

    weeks_of_history = 104
    start = datetime.today() - timedelta(weeks=weeks_of_history)
    rng = np.random.default_rng(len(_META) * 17)
    trend_rate = rng.uniform(0.0015, 0.004)
    rows = []

    for w in range(weeks_of_history):
        date = start + timedelta(weeks=w)
        week_of_year = date.isocalendar().week
        seasonal = _seasonal_multiplier(week_of_year, item.category)
        trend = 1 + trend_rate * w
        noise = rng.normal(1.0, 0.12)
        units = max(0, item.base_demand * seasonal * trend * noise)
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "product_id": pid,
            "name": item.name,
            "category": item.category,
            "units_sold": round(units),
            "is_holiday": 1 if 46 <= week_of_year <= 52 else 0,
            "week_of_year": week_of_year,
        })

    new_df = pd.DataFrame(rows)
    _SALES_DF = pd.concat([_SALES_DF, new_df], ignore_index=True)
    _get_product_result.cache_clear()

    return {
        "message": "Product created successfully",
        "product_id": pid,
        "name": item.name,
        "category": item.category,
        "current_stock": round(stock)
    }


@app.get("/api/forecast/{product_id}")
def get_forecast(product_id: str):
    fc, _plan, meta = _get_product_result(product_id)
    return {
        "product_id": product_id,
        "name": meta["name"],
        "category": meta["category"],
        "history": fc["history"][-26:],
        "forecast": fc["forecast"],
        "model_mae": fc["model_mae"],
        "baseline_mae": fc["baseline_mae"],
        "improvement_pct": fc["improvement_pct"],
    }


@app.get("/api/inventory/{product_id}")
def get_inventory(
    product_id: str,
    service_level: float = Query(0.95, ge=0.80, le=0.999),
    lead_time_days: int = Query(None, ge=1, le=90),
    order_cost: float = Query(45.0, ge=1.0),
    holding_cost_rate: float = Query(0.22, ge=0.01, le=1.0)
):
    fc, default_plan, meta = _get_product_result(product_id)
    effective_lt = lead_time_days if lead_time_days is not None else meta["lead_time_days"]
    z = Z_SCORE_MAP.get(service_level, 1.65)
    
    plan = compute_inventory_plan(
        weekly_units=[r["units_sold"] for r in fc["forecast"]],
        lead_time_days=effective_lt,
        unit_cost=meta["unit_cost"],
        current_stock=_CURRENT_STOCK[product_id],
        z_score=z,
        order_cost=order_cost,
        holding_cost_rate=holding_cost_rate
    )
    
    return {
        "product_id": product_id,
        "name": meta["name"],
        "category": meta["category"],
        "lead_time_days": effective_lt,
        "unit_cost": meta["unit_cost"],
        **plan
    }


@app.get("/api/summary")
def get_summary():
    results = [_get_product_result(pid) for pid in _META.keys()]
    statuses = [plan["status"] for _fc, plan, _meta in results]
    avg_improvement = round(float(np.mean([fc["improvement_pct"] for fc, _p, _m in results])), 1)
    
    all_products = []
    for pid in _META.keys():
        fc, plan, meta = _get_product_result(pid)
        all_products.append({
            "product_id": pid,
            "name": meta["name"],
            "category": meta["category"],
            "unit_cost": meta["unit_cost"],
            "lead_time_days": meta["lead_time_days"],
            "current_stock": plan["current_stock"],
            "reorder_point": plan["reorder_point"],
            "safety_stock": plan["safety_stock"],
            "eoq": plan["eoq"],
            "days_of_supply": plan["days_of_supply"],
            "status": plan["status"],
            "improvement_pct": fc["improvement_pct"]
        })

    return {
        "total_products": len(_META),
        "critical_count": statuses.count("critical"),
        "reorder_count": statuses.count("reorder"),
        "healthy_count": statuses.count("healthy"),
        "avg_forecast_improvement_pct": avg_improvement,
        "products": all_products
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Serve frontend static assets if available
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def read_root():
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Stockwatch API is online. Access documentation at /docs"}
