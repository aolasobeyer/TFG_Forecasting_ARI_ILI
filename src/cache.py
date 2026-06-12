from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import Callable

def load_or_compute_csv(path: Path, compute_fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """
    If file exists: load.
    Else: compute, save, return.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return pd.read_csv(path)

    df = compute_fn()
    df.to_csv(path, index=False)
    return df
