from __future__ import annotations
import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mase_scale_seasonal(y_train: pd.Series, m: int = 52) -> float:
    """
    Classic MASE scaling term for seasonal data:
      scale = mean(|y_t - y_{t-m}|) over training only.
    Returns np.nan if cannot be computed robustly.
    """
    y = y_train.astype(float).copy()

    # Need at least m+1 points
    if y.dropna().shape[0] <= m:
        return np.nan

    diff = (y - y.shift(m)).abs()
    diff = diff.dropna()
    if diff.empty:
        return np.nan

    scale = float(diff.mean())
    # Avoid division by ~0
    if scale <= 1e-12:
        return np.nan
    return scale


def mase_from_mae(mae_value: float, scale: float) -> float:
    """Compute MASE given MAE and precomputed scale."""
    if scale is None or (isinstance(scale, float) and np.isnan(scale)) or scale <= 0:
        return np.nan
    return float(mae_value / scale)


def safe_dropna(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.dropna(subset=cols).copy()
