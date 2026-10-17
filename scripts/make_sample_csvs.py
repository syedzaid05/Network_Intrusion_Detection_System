"""Generate sample CSV feature files for testing the Batch Analysis tab / API.

Usage:
    python scripts/make_sample_csvs.py [--out data/csvs] [--rows 200]

Sources rows from the raw UNSW-NB15 testing set and writes six files:
    sample_normal_flows.csv   - only normal traffic
    sample_attack_flows.csv   - only attacks
    sample_mixed_flows.csv    - shuffled normal + attacks
    sample_dos_generic.csv    - DoS + Generic attacks
    sample_exploits_recon.csv - Exploits + Reconnaissance attacks
    sample_small_batch.csv    - a small mixed batch (50 rows)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "csvs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample CSV feature files.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    parser.add_argument("--rows", type=int, default=200, help="rows per larger file")
    args = parser.parse_args()

    if not config.TEST_CSV.exists():
        raise SystemExit(f"Raw dataset not found: {config.TEST_CSV}")

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config.TEST_CSV)
    feats = df[config.ALL_FEATURE_COLS]
    label = df["label"]
    cat = df["attack_cat"].fillna("Normal") if "attack_cat" in df else label.map({0: "Normal", 1: "Attack"})

    rng = 42
    n = args.rows

    normal = feats[label == 0].sample(min(n, int((label == 0).sum())), random_state=rng)
    attack = feats[label == 1].sample(min(n, int((label == 1).sum())), random_state=rng)
    mixed = pd.concat([normal, attack]).sample(frac=1.0, random_state=rng)
    dos_generic = feats[cat.isin(["DoS", "Generic"])].sample(n, random_state=rng)
    exp_recon = feats[cat.isin(["Exploits", "Reconnaissance"])].sample(n, random_state=rng)
    small = mixed.sample(min(50, len(mixed)), random_state=rng)

    outputs = {
        "sample_normal_flows.csv": normal,
        "sample_attack_flows.csv": attack,
        "sample_mixed_flows.csv": mixed,
        "sample_dos_generic.csv": dos_generic,
        "sample_exploits_recon.csv": exp_recon,
        "sample_small_batch.csv": small,
    }

    for name, frame in outputs.items():
        path = out_dir / name
        frame.reset_index(drop=True).to_csv(path, index=False)
        print(f"[csv] {len(frame):>4} rows, {frame.shape[1]} cols -> {path}")

    print(f"\nDone. {len(outputs)} CSV files written to {out_dir}")


if __name__ == "__main__":
    main()
