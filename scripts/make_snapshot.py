# scripts/make_snapshot.py
# ─────────────────────────────────────────────────────────────────────────────
# Purpose: Freeze a reproducible evaluation snapshot of events as Parquet.
#          - Calls preprocess_events (same filters as the live pipeline)
#          - Saves data/snapshots/events_eval.parquet for evaluation
# Run:
#   python scripts/make_snapshot.py
#   # or with params:
#   python scripts/make_snapshot.py --city Paris --max-records 4000 --lookback-days 365 --out data/snapshots/events_eval.parquet
# Notes: Requires pyarrow or fastparquet for Parquet I/O.
# ─────────────────────────────────────────────────────────────────────────────

import os, sys
from pathlib import Path
import argparse

# Ensure project root is on sys.path so "src" imports work when run as a script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.preprocess_openagenda import preprocess_events  # now safe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Paris")
    ap.add_argument("--max-records", type=int, default=4000)
    ap.add_argument("--lookback-days", type=int, default=365)
    ap.add_argument("--out", default="data/snapshots/events_eval.parquet")
    args = ap.parse_args()

    df = preprocess_events(city=args.city, max_records=args.max_records, lookback_days=args.lookback_days)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    # quick summary
    if not df.empty:
        print(f"Rows: {len(df)}  |  City: {args.city}  |  Range: {df['start_utc'].min()} → {df['start_utc'].max()}")
    print(f"Saved snapshot with {len(df)} rows to {out}")

if __name__ == "__main__":
    main()
