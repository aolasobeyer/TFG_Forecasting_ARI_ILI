from __future__ import annotations
import numpy as np
import pandas as pd
from .metrics import mae, rmse
from .config import HORIZONS

def rolling_baselines(history_init: pd.Series, y_test: pd.Series, horizons=HORIZONS) -> pd.DataFrame:
    """
    Weekly rolling evaluation over test index.
    Each week in test, reveal truth and append to history, then predict h=1..4.
    """
    history = history_init.copy()
    results = []

    for origin in y_test.index:
        # reveal truth at origin
        if pd.notna(y_test.loc[origin]):
            history.loc[origin] = y_test.loc[origin]

        # need origin value for naive_last
        if origin not in history.index or pd.isna(history.loc[origin]):
            continue

        for h in horizons:
            target = origin + pd.Timedelta(weeks=h)
            if target not in y_test.index:
                continue
            if pd.isna(y_test.loc[target]):
                continue

            pred_last = float(history.loc[origin])

            ref = target - pd.Timedelta(weeks=52)
            pred_year = float(history.loc[ref]) if (ref in history.index and pd.notna(history.loc[ref])) else np.nan

            pred_mix = pred_last if h in (1, 2) else pred_year

            results.append({
                "origin": origin,
                "target": target,
                "h": h,
                "y": float(y_test.loc[target]),
                "pred_last": pred_last,
                "pred_year": pred_year,
                "pred_mix": pred_mix,
            })

    return pd.DataFrame(results)

def metrics_from_eval(df_eval: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        sub = df_eval[df_eval["h"] == h].dropna(subset=["y", pred_col])
        if len(sub) == 0:
            continue
        rows.append({
            "h": h,
            "MAE": mae(sub["y"], sub[pred_col]),
            "RMSE": rmse(sub["y"], sub[pred_col]),
            "n": int(len(sub)),
        })
    return pd.DataFrame(rows)

def summarize_baselines(df_eval: pd.DataFrame, disease: str, location: str) -> pd.DataFrame:
    out = []
    for model, col in [
        ("naive_last", "pred_last"),
        ("naive_year", "pred_year"),
        ("naive_mixed", "pred_mix"),
    ]:
        m = metrics_from_eval(df_eval, col)
        if len(m):
            m["model"] = model
            m["disease"] = disease
            m["location"] = location
            out.append(m)
    if not out:
        return pd.DataFrame(columns=["h","MAE","RMSE","n","model","disease","location"])
    return pd.concat(out, ignore_index=True)[["disease","location","h","model","MAE","RMSE","n"]]
