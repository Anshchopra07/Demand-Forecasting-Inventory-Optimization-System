import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def mean_absolute_error(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def _make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    df["t"] = np.arange(len(df))
    df["lag_1"] = df["units_sold"].shift(1)
    df["lag_2"] = df["units_sold"].shift(2)
    df["lag_4"] = df["units_sold"].shift(4)
    df["roll_mean_4"] = df["units_sold"].shift(1).rolling(4).mean()
    df["sin_week"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["cos_week"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    return df


FEATURE_COLS = ["t", "lag_1", "lag_2", "lag_4", "roll_mean_4", "sin_week", "cos_week", "is_holiday"]


def train_and_forecast(product_df: pd.DataFrame, horizon_weeks: int = 12) -> dict:
    df = _make_features(product_df.copy())
    train_df = df.dropna(subset=FEATURE_COLS).copy()

    split = int(len(train_df) * 0.85)
    train, test = train_df.iloc[:split], train_df.iloc[split:]

    model = HistGradientBoostingRegressor(
        max_iter=150, max_depth=4, learning_rate=0.06, random_state=42
    )
    model.fit(train[FEATURE_COLS], train["units_sold"])

    xgb_pred_test = model.predict(test[FEATURE_COLS])
    xgb_mae = mean_absolute_error(test["units_sold"], xgb_pred_test)

    naive_pred_test = test["lag_1"]
    naive_mae = mean_absolute_error(test["units_sold"], naive_pred_test)

    improvement_pct = round((1 - xgb_mae / naive_mae) * 100, 1) if naive_mae > 0 else 0.0

    model.fit(train_df[FEATURE_COLS], train_df["units_sold"])

    history = df[["date", "units_sold", "week_of_year"]].copy()
    working = df.copy()
    future_rows = []
    last_date = pd.to_datetime(working["date"].iloc[-1])

    for step in range(horizon_weeks):
        next_date = last_date + pd.Timedelta(weeks=step + 1)
        week_of_year = next_date.isocalendar().week
        recent = working["units_sold"].values
        feat = {
            "t": float(working["t"].iloc[-1] + 1),
            "lag_1": float(recent[-1]),
            "lag_2": float(recent[-2]),
            "lag_4": float(recent[-4]),
            "roll_mean_4": float(np.mean(recent[-4:])),
            "sin_week": float(np.sin(2 * np.pi * week_of_year / 52)),
            "cos_week": float(np.cos(2 * np.pi * week_of_year / 52)),
            "is_holiday": 1 if 46 <= week_of_year <= 52 else 0,
        }
        input_df = pd.DataFrame([feat])[FEATURE_COLS]
        pred = max(0, float(model.predict(input_df)[0]))
        future_rows.append({"date": next_date.strftime("%Y-%m-%d"), "units_sold": round(pred), "week_of_year": week_of_year})
        working = pd.concat([working, pd.DataFrame([{**feat, "units_sold": pred}])], ignore_index=True)

    return {
        "history": history.to_dict(orient="records"),
        "forecast": future_rows,
        "model_mae": round(xgb_mae, 1),
        "baseline_mae": round(naive_mae, 1),
        "improvement_pct": improvement_pct,
    }
