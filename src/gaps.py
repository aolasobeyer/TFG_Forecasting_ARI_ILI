from __future__ import annotations
import pandas as pd

def _gap_lengths_and_starts(missing_mask: pd.Series) -> tuple[list[int], list[pd.Timestamp]]:
    """
    missing_mask: boolean Series indexed by dates (True if missing)
    Returns:
      gap_lengths: list[int]
      gap_starts: list[pd.Timestamp]
    """
    if missing_mask.empty:
        return [], []

    block_id = (missing_mask != missing_mask.shift(fill_value=False)).cumsum()

    gap_lengths: list[int] = []
    gap_starts: list[pd.Timestamp] = []

    for _, block in missing_mask.groupby(block_id):
        if bool(block.iloc[0]):  # block of True => gap
            gap_lengths.append(int(block.sum()))
            gap_starts.append(block.index[0])

    return gap_lengths, gap_starts


def gap_summary(df: pd.DataFrame, disease: str, calendar_mode: str = "fixed_2014") -> pd.DataFrame:
    """
    df columns: location, truth_date, value
    calendar_mode:
      - "fixed_2014": calendar from 2014-01-05 to max date (coverage)
      - "data_range": calendar from min date to max date
      - "per_location": calendar min..max per location (internal gaps only)
    """
    required = {"location", "truth_date", "value"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in df: {missing_cols}")

    df = df.copy()
    df["truth_date"] = pd.to_datetime(df["truth_date"])
    df = df.sort_values(["location", "truth_date"])

    if calendar_mode == "fixed_2014":
        cal_start = pd.Timestamp("2014-01-05")
        cal_end = df["truth_date"].max()
        full_weeks_global = pd.date_range(start=cal_start, end=cal_end, freq="W-SUN")
    elif calendar_mode == "data_range":
        cal_start = df["truth_date"].min()
        cal_end = df["truth_date"].max()
        full_weeks_global = pd.date_range(start=cal_start, end=cal_end, freq="W-SUN")
    elif calendar_mode == "per_location":
        full_weeks_global = None
    else:
        raise ValueError("calendar_mode must be: 'fixed_2014', 'data_range', or 'per_location'")

    rows = []

    for loc, g in df.groupby("location", sort=True):
        g = g.drop_duplicates(subset=["truth_date"], keep="last").set_index("truth_date").sort_index()

        if calendar_mode == "per_location":
            if g.index.size == 0:
                continue
            full_weeks = pd.date_range(start=g.index.min(), end=g.index.max(), freq="W-SUN")
        else:
            full_weeks = full_weeks_global

        g_full = g.reindex(full_weeks)
        missing_mask = g_full["value"].isna()

        gap_lengths, gap_starts = _gap_lengths_and_starts(missing_mask)

        n_weeks = int(len(full_weeks))
        n_observed = int(g_full["value"].notna().sum())
        n_missing = int(missing_mask.sum())

        max_gap = int(max(gap_lengths)) if gap_lengths else 0
        n_gaps = int(len(gap_lengths))

        biggest_gap_start = pd.NaT
        if gap_lengths:
            idx = gap_lengths.index(max_gap)
            biggest_gap_start = gap_starts[idx]

        last_gap_start = gap_starts[-1] if gap_starts else pd.NaT

        rows.append({
            "disease": disease,
            "location": loc,
            "calendar_mode": calendar_mode,
            "n_weeks": n_weeks,
            "n_observed": n_observed,
            "n_missing": n_missing,
            "na_pct": (n_missing / n_weeks) if n_weeks > 0 else 0.0,
            "n_gaps": n_gaps,
            "gap_lengths": gap_lengths,
            "gap_starts": gap_starts,
            "max_gap_length": max_gap,
            "biggest_gap_start": biggest_gap_start,
            "last_gap_start": last_gap_start,
        })

    return pd.DataFrame(rows)
