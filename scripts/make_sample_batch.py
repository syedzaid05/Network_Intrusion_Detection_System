"""Create a sample CSV for the dashboard's Batch Analysis tab.

Keeps the 42 raw UNSW-NB15 feature columns (plus ``attack_cat``/``label`` for
reference) so the file can be uploaded as-is.

Usage:
    python scripts/make_sample_batch.py [--rows 2000] [--out data/samples/batch_sample.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a sample batch CSV.")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--out", default=str(config.PROJECT_ROOT / "data" / "samples" / "batch_sample.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not config.TEST_CSV.exists():
        raise SystemExit(f"Test set not found: {config.TEST_CSV}. Run scripts/run_pipeline.py first.")

    df = pd.read_csv(config.TEST_CSV)
    cols = [c for c in config.ALL_FEATURE_COLS if c in df.columns]
    extra = [c for c in (config.ATTACK_CAT_COL, config.TARGET_COL) if c in df.columns]
    sample = df.sample(n=min(args.rows, len(df)), random_state=args.seed)[cols + extra]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    print(f"[sample] wrote {len(sample)} rows x {sample.shape[1]} cols -> {out}")
    print(f"[sample] attack_cat: {sample.get('attack_cat', pd.Series(dtype=str)).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
