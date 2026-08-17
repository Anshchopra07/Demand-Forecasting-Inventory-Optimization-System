# Stockwatch — Demand Forecasting & Inventory Optimization

End-to-end system: XGBoost demand forecasting + reorder-point/EOQ inventory logic, served through a FastAPI backend and a live control-room style dashboard.

```
inventory-forecast/
├── backend/
│   ├── main.py          FastAPI app (all endpoints)
│   ├── data_gen.py       synthetic retail dataset (swap for real data later)
│   ├── forecasting.py    XGBoost model + recursive 12-week forecast
│   ├── inventory.py      reorder point / safety stock / EOQ formulas
│   ├── requirements.txt
│   └── Procfile          Render start command
└── frontend/
    └── index.html         single-file dashboard (vanilla JS + Chart.js)
```

## Run it locally (2 minutes)

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then, in another terminal:

```bash
cd frontend
python3 -m http.server 8080
```

Open `http://localhost:8080` — it talks to the backend at `http://localhost:8000` by default.

## Deploy for real (Render, free tier)

### Backend
1. Push this folder to a GitHub repo.
2. On [render.com](https://render.com) → New → Web Service → connect the repo.
3. Root directory: `backend`. Build command: `pip install -r requirements.txt`. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT` (already in the Procfile, Render auto-detects it).
4. Deploy. You'll get a URL like `https://stockwatch-api.onrender.com`.

### Frontend
1. On Render → New → Static Site → same repo, root directory `frontend`.
2. Before deploying, open `frontend/index.html` and change this one line near the bottom:
   ```js
   const API_BASE = window.STOCKWATCH_API_BASE || "http://localhost:8000";
   ```
   Replace `http://localhost:8000` with your backend's Render URL.
3. Deploy. You now have a live link to put on your resume.

*(GitHub Pages works too for the frontend — it's a static file with no build step.)*

## What to say about it on your resume / in interviews

- **The forecasting story**: XGBoost trained on lag features + seasonal (sin/cos week-of-year) features, evaluated against a naive last-value baseline. Talk through *why* you chose those specific features and how you validated (holdout split, MAE).
- **The ops story**: reorder point, safety stock (95% service level via z-score), and EOQ (Wilson formula) — standard supply-chain formulas, correctly implemented and driven by live forecast output, not static numbers.
- **The engineering story**: clean separation of concerns (data / model / business logic / API / UI), a REST API you designed, and a deployed, clickable product — not just a notebook.

## Swapping in the real Kaggle dataset later

`data_gen.py` generates synthetic-but-realistic data so the whole thing runs with zero setup.
To use the real "Walmart Recruiting - Store Sales Forecasting" dataset instead: download it from Kaggle, merge `train.csv` + `features.csv` + `stores.csv` on store/date, and replace `load_sales_data()` in `data_gen.py` with a loader that returns the same columns (`date`, `product_id`, `name`, `category`, `units_sold`, `is_holiday`, `week_of_year`). Nothing else in the pipeline needs to change.
