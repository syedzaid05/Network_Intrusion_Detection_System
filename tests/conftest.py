"""Shared pytest fixtures and path setup."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture
def sample_frame():
    """A tiny hand-built UNSW-NB15-shaped dataframe."""
    import pandas as pd

    rows = [
        {**{c: 0 for c in config.NUMERIC_COLS}, "proto": "tcp", "service": "http",
         "state": "CON", "attack_cat": "Normal", "label": 0, "ID_NO": 1},
        {**{c: 0 for c in config.NUMERIC_COLS}, "proto": "tcp", "service": "http",
         "state": "CON", "attack_cat": "Normal", "label": 0, "ID_NO": 2},
        {**{c: 0 for c in config.NUMERIC_COLS}, "proto": "udp", "service": "-",
         "state": "INT", "attack_cat": "DoS", "label": 1, "ID_NO": 3},
    ]
    frame = pd.DataFrame(rows)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)  # exact duplicate of row 0
    for col in config.NUMERIC_COLS:
        frame[col] = frame[col].astype(float)
    frame.loc[1, "sbytes"] = None       # missing value
    frame.loc[2, "dur"] = float("inf")  # inf should be repaired
    return frame
