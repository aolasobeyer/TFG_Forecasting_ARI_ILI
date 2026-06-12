from __future__ import annotations
import pandas as pd
from pathlib import Path

REQUIRED_COLS = {"location", "truth_date", "value"}

def read_respicast_csv(path: Path) -> pd.DataFrame:
    """
    Reads an ERVISS target-data CSV (latest-... or snapshot).
    Ensures truth_date is datetime and sorted.
    """
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    df = df.copy()
    df["truth_date"] = pd.to_datetime(df["truth_date"])
    df = df.sort_values(["location", "truth_date"]).reset_index(drop=True)
    return df

def read_ari_ili(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expects data_dir contains latest-ARI_incidence.csv and latest-ILI_incidence.csv
    """
    ari = read_respicast_csv(data_dir / "latest-ARI_incidence.csv")
    ili = read_respicast_csv(data_dir / "latest-ILI_incidence.csv")
    return ari, ili
