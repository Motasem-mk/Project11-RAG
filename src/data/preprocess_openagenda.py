# src/data/preprocess_openagenda.py
"""
Fetch OpenAgenda (Opendatasoft) events for an optional city, then clean them:
- Normalize fields (uid, title, city, postal_code, venue, website, permalink, tags, start_utc, end_utc, text)
- Parse datetimes to UTC
- Keep only events from the last N days (lookback_days, default 365)
- Strip HTML from descriptions

Quick test:
    python -m src.data.preprocess_openagenda --city Paris --max-records 1500 --lookback-days 365
"""

# from __future__ import annotations
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List

import warnings
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

import requests
import pandas as pd

ODATASET = "evenements-publics-openagenda"
API_URL = "https://public.opendatasoft.com/api/records/1.0/search/"
DEFAULT_ROWS = 500  # per page


# ---------------------------
# 1) Helpers: Fetch & Normalize
# ---------------------------
def _fetch_page(city: Optional[str], rows: int, start: int) -> Dict[str, Any]:
    """Call one API page of the Opendatasoft dataset for OpenAgenda events."""
    params = {
        "dataset": ODATASET,
        "rows": rows,
        "start": start,
        **({"refine.location_city": city} if city else {}),
        "sort": "firstdate_begin",
    }
    resp = requests.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _records_to_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map raw 'records' into a normalized row dict we control."""
    rows: List[Dict[str, Any]] = []
    for rec in records:
        f = rec.get("fields", {})
        rows.append(
            {
                "uid": f.get("uid"),
                "title": f.get("title_fr") or "",
                "city": f.get("location_city") or "",
                "postal_code": f.get("location_postalcode") or "",
                "venue": f.get("location_name") or "",
                "website": f.get("canonicalurl") or "",
                "permalink": f.get("canonicalurl") or "",
                "tags": f.get("keywords_fr") or [],
                "start_utc": f.get("firstdate_begin"),
                "end_utc": f.get("firstdate_end"),
                "text": (f.get("longdescription_fr") or f.get("description_fr") or ""),
            }
        )
    return rows


def _strip_html(html_text: str) -> str:
    """Turn HTML into plain text; safe on non-strings."""
    if not isinstance(html_text, str):
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text(separator=" ").strip()


# ---------------------------
# 2) Public function
# ---------------------------
def preprocess_events(
    city: Optional[str],
    max_records: int = 9000,
    rows: int = DEFAULT_ROWS,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """
    Fetch up to `max_records` events (optionally filtered by city) and return a clean DataFrame:
      - timezone-aware datetimes
      - last `lookback_days` days
      - description text cleaned from HTML
      - tags joined as a comma-separated string
    """
    pages = math.ceil(max_records / rows)
    all_rows: List[Dict[str, Any]] = []
    start = 0

    for _ in range(pages):
        data = _fetch_page(city=city, rows=rows, start=start)
        recs = data.get("records", [])
        if not recs:
            break
        all_rows.extend(_records_to_rows(recs))
        start += rows
        time.sleep(0.2)  # be polite

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    # Ensure stable schema even if API misses some fields
    required = [
        "uid", "title", "city", "postal_code", "venue", "website", "permalink",
        "tags", "start_utc", "end_utc", "text"
    ]
    for col in required:
        if col not in df.columns:
            df[col] = None

    # Parse datetimes to UTC & filter by lookback
    df["start_utc"] = pd.to_datetime(df["start_utc"], errors="coerce", utc=True)
    df["end_utc"] = pd.to_datetime(df["end_utc"], errors="coerce", utc=True)
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    df = df[df["start_utc"].notna() & (df["start_utc"] >= cutoff)].copy()

    # Clean HTML + normalize tags
    df["text"] = df["text"].apply(_strip_html)
    df["tags"] = df["tags"].apply(lambda x: ", ".join(x) if isinstance(x, list) else (x or ""))

    # Drop duplicates by uid (if any)
    if "uid" in df.columns:
        df = df.drop_duplicates(subset=["uid"]).reset_index(drop=True)

    return df[required].reset_index(drop=True)


# ---------------------------
# 3) CLI (optional sanity check)
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="Paris")
    parser.add_argument("--max-records", type=int, default=1500)
    parser.add_argument("--lookback-days", type=int, default=365)
    args = parser.parse_args()

    out = preprocess_events(city=args.city, max_records=args.max_records, lookback_days=args.lookback_days)
    print(out.head())
    print(f"Rows after lookback filter ({args.lookback_days} days): {len(out)}")
