"""Dataset acquisition and loading for UNSW-NB15."""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from . import config


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    class _Progress(tqdm):
        def update_to(self, blocks=1, block_size=1, total=None):
            if total is not None:
                self.total = total
            self.update(blocks * block_size - self.n)

    with _Progress(unit="B", unit_scale=True, desc=f"downloading {dest.name}") as bar:
        urllib.request.urlretrieve(url, tmp, reporthook=bar.update_to)
    tmp.replace(dest)


def ensure_dataset() -> None:
    """Download any missing raw CSV files."""
    for name, url in config.DATA_URLS.items():
        dest = config.DATA_RAW / name
        if not dest.exists():
            print(f"[data] {name} missing, downloading...")
            _download(url, dest)


def _row_count(path: Path) -> int:
    with open(path, "rb") as fh:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(1 << 20), b""))


def _normalise_splits() -> None:
    """Some mirrors swap train/test filenames; fix by row count."""
    paths = [config.TRAIN_CSV, config.TEST_CSV]
    if not all(p.exists() for p in paths):
        return
    train_rows, test_rows = (_row_count(p) for p in paths)
    if train_rows < test_rows:
        print("[data] train/test filenames appear swapped -> correcting")
        tmp = config.TRAIN_CSV.with_suffix(".swap")
        config.TRAIN_CSV.rename(tmp)
        config.TEST_CSV.rename(config.TRAIN_CSV)
        tmp.rename(config.TEST_CSV)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the official (train, test) dataframes."""
    ensure_dataset()
    _normalise_splits()
    train = pd.read_csv(config.TRAIN_CSV)
    test = pd.read_csv(config.TEST_CSV)
    print(f"[data] train={train.shape} test={test.shape}")
    return train, test


if __name__ == "__main__":
    tr, te = load_raw()
    print(tr[config.TARGET_COL].value_counts().to_dict())
    print(te[config.ATTACK_CAT_COL].value_counts().to_dict())
    sys.exit(0)
