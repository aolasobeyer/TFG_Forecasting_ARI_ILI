from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from .config import WEEK_FREQ, DEFAULT_FIXED_START, TRAIN_END, TEST_START, TEST_END

@dataclass(frozen=True)
class Calendars:
    full_weeks: pd.DatetimeIndex
    train_weeks: pd.DatetimeIndex
    test_weeks: pd.DatetimeIndex

def make_calendars_from_df(df: pd.DataFrame, fixed_start: str = DEFAULT_FIXED_START) -> Calendars:
    """
    Full weekly calendar from fixed_start to df max date.
    Train: <= TRAIN_END
    Test: TEST_START..TEST_END
    """
    end = pd.to_datetime(df["truth_date"]).max()
    full = pd.date_range(start=pd.Timestamp(fixed_start), end=end, freq=WEEK_FREQ)

    train_end = pd.Timestamp(TRAIN_END)
    test_start = pd.Timestamp(TEST_START)
    test_end = pd.Timestamp(TEST_END)

    train = full[full <= train_end]
    test = full[(full >= test_start) & (full <= test_end)]
    return Calendars(full_weeks=full, train_weeks=train, test_weeks=test)

def extract_raw_weekly_series(df: pd.DataFrame, location: str, calendar: pd.DatetimeIndex) -> pd.Series:
    """
    Returns raw value series reindexed on calendar (may include NaNs).
    """
    dfloc = df[df["location"] == location].copy()
    dfloc["truth_date"] = pd.to_datetime(dfloc["truth_date"])
    s = (
        dfloc.sort_values("truth_date")
        .drop_duplicates("truth_date", keep="last")
        .set_index("truth_date")["value"]
        .reindex(calendar)
    )
    return s
