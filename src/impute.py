from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

def find_nan_gaps(s: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """
    Returns list of gaps: (start, end, length), inclusive, for NaN blocks.
    """
    is_na = s.isna()
    if is_na.sum() == 0:
        return []

    block_id = (is_na != is_na.shift(fill_value=False)).cumsum()
    gaps = []
    for _, block in is_na.groupby(block_id):
        if bool(block.iloc[0]):
            gaps.append((block.index[0], block.index[-1], int(block.sum())))
    return gaps

def impute_singletons_neighbor_mean(s: pd.Series) -> pd.Series:
    """
    Impute gaps of length 1 with mean of neighbors (t-1 and t+1) if both exist and not NaN.
    """
    s = s.copy()
    gaps = find_nan_gaps(s)
    idx = s.index

    for start, _, length in gaps:
        if length != 1:
            continue
        t = start
        pos = idx.get_loc(t)
        if pos - 1 < 0 or pos + 1 >= len(idx):
            continue
        prev_t = idx[pos - 1]
        next_t = idx[pos + 1]
        if pd.notna(s.loc[prev_t]) and pd.notna(s.loc[next_t]):
            s.loc[t] = 0.5 * (s.loc[prev_t] + s.loc[next_t])
    return s

def build_lag_features(series: pd.Series, n_lags: int = 8) -> pd.DataFrame:
    df = pd.DataFrame({"value": series})
    for k in range(1, n_lags + 1):
        df[f"lag_{k}"] = series.shift(k)
        df[f"lead_{k}"] = series.shift(-k)
    return df

def impute_gaps_knn(
    s: pd.Series,
    max_gap_knn: int = 8,
    n_lags: int = 8,
    k_neighbors: int = 5
) -> pd.Series:
    """
    Impute gaps of length 2..max_gap_knn using local KNN within a window = 2*gap_len.
    """
    s = s.copy()
    gaps = find_nan_gaps(s)
    idx = s.index

    for start, end, length in gaps:
        if length < 2 or length > max_gap_knn:
            continue

        w = 2 * length
        start_i = idx.get_loc(start)
        end_i = idx.get_loc(end)

        left_i = max(0, start_i - w)
        right_i = min(len(idx) - 1, end_i + w)

        block_idx = idx[left_i:right_i + 1]
        block = s.loc[block_idx]

        # If too little info, skip
        if block.notna().sum() < max(5, 2 * length):
            continue

        X = build_lag_features(block, n_lags=n_lags)
        imputer = KNNImputer(n_neighbors=k_neighbors, weights="distance")
        X_imp = pd.DataFrame(imputer.fit_transform(X), index=X.index, columns=X.columns)

        s.loc[start:end] = X_imp.loc[start:end, "value"]

    return s

def impute_series_weekly(
    df_loc: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    trim_to_first_obs: bool = True,
    max_gap_knn: int = 8,
    n_lags: int = 8,
    k_neighbors: int = 5
) -> pd.Series:
    """
    df_loc columns: truth_date, value (and maybe others)
    Returns weekly series indexed by calendar (or trimmed) with training-only imputations applied.
    """
    g = df_loc.copy()
    g["truth_date"] = pd.to_datetime(g["truth_date"])
    g = (
        g.sort_values("truth_date")
         .drop_duplicates("truth_date", keep="last")
         .set_index("truth_date")
    )

    s = g["value"].reindex(calendar)

    if trim_to_first_obs:
        first = s.first_valid_index()
        if first is not None:
            s = s.loc[first:]

    s = impute_singletons_neighbor_mean(s)
    s = impute_gaps_knn(s, max_gap_knn=max_gap_knn, n_lags=n_lags, k_neighbors=k_neighbors)
    return s
