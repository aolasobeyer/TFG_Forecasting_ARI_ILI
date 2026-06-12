from __future__ import annotations
import warnings
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .metrics import mae, rmse
from .config import HORIZONS

warnings.filterwarnings("ignore")

def rolling_sarima(
    history_init: pd.Series,
    y_test: pd.Series,
    order=(1, 0, 0),
    seasonal_order=(1, 0, 0, 52),
    trend="c",
    horizons=HORIZONS
) -> pd.DataFrame:
    """
    Rolling re-estimation SARIMA:
    each origin, append truth if available, fit SARIMAX on full history, forecast 4.
    """
    history = history_init.copy()
    results = []

    for origin in y_test.index:
        if pd.notna(y_test.loc[origin]):
            history.loc[origin] = y_test.loc[origin]

        if origin not in history.index or pd.isna(history.loc[origin]):
            continue

        model = SARIMAX(
            history,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        y_hat = res.forecast(steps=max(horizons))

        for h in horizons:
            target = origin + pd.Timedelta(weeks=h)
            if target not in y_test.index or pd.isna(y_test.loc[target]):
                continue
            results.append({
                "origin": origin,
                "target": target,
                "h": h,
                "y": float(y_test.loc[target]),
                "pred_sarima": float(y_hat.iloc[h - 1]),
            })

    return pd.DataFrame(results)

def summarize_sarima(df_eval: pd.DataFrame, disease: str, location: str, model_name: str) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        sub = df_eval[df_eval["h"] == h].dropna(subset=["y", "pred_sarima"])
        if len(sub) == 0:
            continue
        rows.append({
            "disease": disease,
            "location": location,
            "h": h,
            "model": model_name,
            "MAE": mae(sub["y"], sub["pred_sarima"]),
            "RMSE": rmse(sub["y"], sub["pred_sarima"]),
            "n": int(len(sub)),
        })
    return pd.DataFrame(rows)
