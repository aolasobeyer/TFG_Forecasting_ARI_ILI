from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    results: Path

def get_project_paths(project_root: Path | None = None) -> ProjectPaths:
    """
    Resolves project root. In notebooks, call with:
      paths = get_project_paths(Path.cwd().parent)
    assuming notebooks/ is one level below root.
    """
    root = project_root if project_root is not None else Path.cwd()
    data = root / "data"
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(root=root, data=data, results=results)

# Common forecasting setup
WEEK_FREQ = "W-SUN"
DEFAULT_FIXED_START = "2014-01-05"

TRAIN_END = "2022-12-25"   # last Sunday of 2022
TEST_START = "2023-01-01"
TEST_END = "2023-12-31"
HORIZONS = (1, 2, 3, 4)
