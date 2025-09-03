# scripts/eval_retrieval.py
# ──────────────────────────────────────────────────────────────────────────────────
# Purpose: Evaluate retrieval quality (Recall@k) using rule-based gold sets.
#          - Loads frozen snapshot (Parquet) and qa_rules.csv (rules, not UIDs)
#          - For each rule row, filters the snapshot → gold UID set
#          - Queries FAISS with the question → retrieved UIDs (top-k)
#          - Scores a hit if intersection(gold, retrieved) is non-empty
# Inputs:  --qa data/eval/qa_rules.csv
#          --snapshot data/snapshots/events_eval.parquet
#          --index data/index/faiss
#          --klist "10,20"
# Options: --show N (preview N retrieved docs per row)
#          --use-postal (apply postal_code filter when building gold; default off)
# Output:  Per-row hits + overall Recall@k printed to stdout
# Requires: MISTRAL_API_KEY in .env; FAISS index already built
# Run:      python scripts/eval_retrieval.py --klist 10,20 --show 5
#           (add --use-postal to make ZIP code part of gold rules)
# ──────────────────────────────────────────────────────────────────────────────────

import os, sys, re
from pathlib import Path
import argparse
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

# Ensure project root is on sys.path for local imports (if needed)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------- Normalization helpers (accent-insensitive, lowercased) ----------
ACCENTS = str.maketrans(
    "àáâäãåçèéêëìíîïñòóôöõùúûüýÿÀÁÂÄÃÅÇÈÉÊËÌÍÎÏÑÒÓÔÖÕÙÚÛÜÝ",
    "aaaaaaceeeeiiiinooooouuuuyyAAAAAACEEEEIIIINOOOOOUUUUY"
)

def norm_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return s.translate(ACCENTS).lower()

# ---------- Data loading & gold building ----------
def load_snapshot(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # required columns normalized
    df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True, errors="coerce")
    for col in ["tags", "title", "text", "city", "postal_code"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    # normalized searchable projections
    df["norm_tags"] = df["tags"].apply(norm_text)
    df["norm_title"] = df["title"].apply(norm_text)
    df["norm_text"] = df["text"].apply(norm_text)
    df["zip5"] = df["postal_code"].astype(str).str[:5]
    return df

def build_gold_uids(df: pd.DataFrame, row: pd.Series, use_postal: bool) -> set:
    mask = pd.Series(True, index=df.index)

    # city exact match (case-insensitive)
    if row["city"]:
        mask &= df["city"].str.casefold() == str(row["city"]).casefold()

    # date range inclusive
    if row["date_from"]:
        mask &= df["start_utc"] >= pd.Timestamp(row["date_from"], tz="UTC")
    if row["date_to"]:
        mask &= df["start_utc"] <= (pd.Timestamp(row["date_to"]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).tz_localize("UTC")

    # tags-any across tags/title/text, accent-insensitive
    if row["tags_any"]:
        any_tags = [t.strip() for t in str(row["tags_any"]).split("|") if t.strip()]
        if any_tags:
            tmask = False
            for t in any_tags:
                nt = norm_text(t)
                hit = (
                    df["norm_tags"].str.contains(nt, na=False) |
                    df["norm_title"].str.contains(nt, na=False) |
                    df["norm_text"].str.contains(nt, na=False)
                )
                tmask = tmask | hit
            mask &= tmask

    # optional postal filter (OFF by default)
    if use_postal and str(row.get("postal_code") or "").strip():
        zip5 = str(row["postal_code"]).strip()[:5]
        mask &= df["zip5"] == zip5

    return set(df.loc[mask, "uid"].dropna().astype(str).tolist())

# ---------- Retrieval ----------
def retrieve_uids(vs: FAISS, question: str, k: int) -> list:
    docs = vs.similarity_search(question, k=k)
    return [str((d.metadata or {}).get("uid")) for d in docs if (d.metadata or {}).get("uid")]

def maybe_preview(vs: FAISS, question: str, k: int, show: int):
    if show <= 0:
        return
    docs = vs.similarity_search(question, k=k)
    for d in docs[:show]:
        m = d.metadata or {}
        title = (m.get("title") or "").strip() or "(no title)"
        uid = str(m.get("uid") or "")
        print(f"- {title} [{uid}]")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="data/eval/qa_rules.csv")
    ap.add_argument("--snapshot", default="data/snapshots/events_eval.parquet")
    ap.add_argument("--index", default="data/index/faiss")
    ap.add_argument("--klist", default="10,20")
    ap.add_argument("--show", type=int, default=0, help="Preview top-N retrieved docs per row")
    ap.add_argument("--use-postal", action="store_true", help="Apply postal_code in gold building (default: ignore)")
    args = ap.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise SystemExit("MISTRAL_API_KEY missing")

    df = load_snapshot(args.snapshot)
    qa = pd.read_csv(args.qa).fillna("")

    emb = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)
    vs = FAISS.load_local(args.index, embeddings=emb, allow_dangerous_deserialization=True)

    ks = [int(x) for x in args.klist.split(",")]
    hits = {k: 0 for k in ks}
    denom = 0
    per_row = []

    print(f"\nEval: rows={len(qa)} | ks={ks} | show={args.show} | use_postal={args.use_postal}\n")

    for _, r in qa.iterrows():
        gold = build_gold_uids(df, r, use_postal=args.use_postal)
        # quick preview
        if args.show:
            print(f"[{r['id']}] Q: {r['question_fr']}")
            print(f"  gold(|UIDs|={len(gold)}), retrieved@{ks[0]} {ks[0]}")
            maybe_preview(vs, r["question_fr"], ks[0], args.show)
            print()

        if not gold:
            per_row.append((r["id"], 0, {k: 0 for k in ks}, "EMPTY_GOLD"))
            continue

        denom += 1
        row_hits = {}
        for k in ks:
            ru = set(retrieve_uids(vs, r["question_fr"], k))
            hit = bool(ru & gold)
            row_hits[k] = int(hit)
            if hit:
                hits[k] += 1
        per_row.append((r["id"], len(gold), row_hits, ""))

    print("\nPer-row summary (id, |gold|, hits_by_k, note):")
    for rid, gold_sz, row_hits, note in per_row:
        print(f"{rid:>4}  gold={gold_sz:>3}  hits={row_hits}  {note}")

    print("\nRecall:")
    for k in ks:
        if denom == 0:
            print(f"Recall@{k}: n/a (no non-empty gold rows)")
        else:
            print(f"Recall@{k}: {hits[k]}/{denom} = {hits[k]/denom:.3f}")

if __name__ == "__main__":
    main()
