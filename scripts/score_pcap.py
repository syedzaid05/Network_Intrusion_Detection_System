"""Score a PCAP file with the trained IDS model.

Usage:
    python scripts/score_pcap.py capture.pcap [--max-flows 5000] [--out results.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import pcap_extractor, predict  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Score network flows from a PCAP file.")
    parser.add_argument("pcap", help="path to a .pcap/.pcapng file")
    parser.add_argument("--max-flows", type=int, default=None, help="cap on flows to score")
    parser.add_argument("--out", default=None, help="CSV output path")
    args = parser.parse_args()

    if not Path(args.pcap).exists():
        raise SystemExit(f"PCAP not found: {args.pcap}")
    if not predict.model_ready():
        raise SystemExit("No trained model. Run: python scripts/run_pipeline.py")

    print(f"[pcap] extracting flows from {args.pcap} ...")
    df = pcap_extractor.pcap_to_dataframe(args.pcap, max_flows=args.max_flows)
    if df.empty:
        raise SystemExit("No IP flows found in capture.")
    print(f"[pcap] {len(df)} flows extracted; scoring ...")

    scored = predict.predict_dataframe(df)
    attacks = scored[scored["verdict"] == "Attack"]
    out = args.out or str(Path(args.pcap).with_suffix(".predictions.csv"))
    scored.to_csv(out, index=False)

    print("\n=== Summary ===")
    print(f"flows      : {len(scored)}")
    print(f"attacks    : {len(attacks)} ({len(attacks) / len(scored):.1%})")
    print(f"severity   : {scored['severity'].value_counts().to_dict()}")
    if "attack_category" in scored:
        print(f"categories : {scored['attack_category'].value_counts().to_dict()}")
    print(f"results    : {out}")

    if not attacks.empty:
        cols = ["proto", "service", "state", "attack_category", "attack_probability"]
        print("\nTop suspicious flows:")
        print(
            attacks.sort_values("attack_probability", ascending=False)
            .head(10)[cols]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
