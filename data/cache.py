"""Local cache for every download. Importing this configures nflreadpy's own cache too."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"

# nflreadpy reads these at import time, so they must be set before it is imported.
os.environ.setdefault("NFLREADPY_CACHE", "filesystem")
os.environ.setdefault("NFLREADPY_CACHE_DIR", str(CACHE / "nflreadpy"))
os.environ.setdefault("NFLREADPY_CACHE_DURATION", str(12 * 3600))
os.environ.setdefault("NFLREADPY_VERBOSE", "false")

REFRESH = False  # set by main.py --refresh


def _fresh(path: Path, hours: float | None) -> bool:
    if REFRESH or not path.exists():
        return False
    return hours is None or (time.time() - path.stat().st_mtime) < hours * 3600


def frame(name: str, build: Callable[[], pd.DataFrame], hours: float | None) -> pd.DataFrame:
    """Parquet-cached DataFrame. hours=None means never expires (past seasons)."""
    path = CACHE / f"{name}.parquet"
    if _fresh(path, hours):
        return pd.read_parquet(path)
    df = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def blob(name: str, fetch: Callable[[], Any], hours: float | None) -> Any:
    """JSON-cached API response."""
    path = CACHE / f"{name}.json"
    if _fresh(path, hours):
        return json.loads(path.read_text())
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data
