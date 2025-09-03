# ############################################################################################ #

# src/index/build_faiss.py
"""
Build and persist a FAISS index from preprocessed events.
- Adds normalized Year/Month tokens to page_content (improves date-aware retrieval)
- Adds PostalCode token for postcode-aware retrieval
- Splits with RecursiveCharacterTextSplitter
- Embeds with Mistral embeddings
- Saves FAISS locally

Run:
    python -m src.index.build_faiss --city Paris --max-records 9000 \
        --lookback-days 365 --chunk-size 800 --chunk-overlap 120 \
        --index-out data/index/faiss
"""

from __future__ import annotations
import os
import argparse
from typing import List, Tuple
import time

import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.data.preprocess_openagenda import preprocess_events


def _norm_parts(ts) -> Tuple[str, str]:
    """Return (YYYY, MM) strings or ('','') if ts is NaT/None."""
    if ts is None or pd.isna(ts):
        return "", ""
    try:
        return str(ts.year), f"{ts.month:02d}"
    except Exception:
        return "", ""


def _df_to_documents(df: pd.DataFrame) -> List[Document]:
    """Create Documents and inject normalized date & postal code tokens in page_content."""
    docs: List[Document] = []
    for _, r in df.iterrows():
        y_s, m_s = _norm_parts(r.get("start_utc"))
        y_e, m_e = _norm_parts(r.get("end_utc"))
        page = (
            f"Title: {r.get('title','')}\n"
            f"City: {r.get('city','')}\n"
            f"PostalCode: {r.get('postal_code','')}\n"
            f"Venue: {r.get('venue','')}\n"
            f"Start: {r.get('start_utc')}\n"
            f"End: {r.get('end_utc')}\n"
            f"YearStart: {y_s}\n"
            f"MonthStart: {m_s}\n"
            f"YearEnd: {y_e}\n"
            f"MonthEnd: {m_e}\n"
            f"Tags: {r.get('tags','')}\n\n"
            f"{r.get('text','')}"
        )
        meta = {
            "uid": r.get("uid"),
            "title": r.get("title"),
            "city": r.get("city"),
            "postal_code": r.get("postal_code"),
            "venue": r.get("venue"),
            "website": r.get("website"),
            "permalink": r.get("permalink"),
            "start_utc": r.get("start_utc"),
            "end_utc": r.get("end_utc"),
            "tags": r.get("tags"),
        }
        docs.append(Document(page_content=page, metadata=meta))
    return docs


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="Paris")
    parser.add_argument("--max-records", type=int, default=9000)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--index-out", type=str, default="data/index/faiss")
    args = parser.parse_args()

    print(f"Fetching & preprocessing events for city={args.city} (lookback={args.lookback_days} days) ...")
    df = preprocess_events(city=args.city, max_records=args.max_records, lookback_days=args.lookback_days)
    if df.empty:
        raise SystemExit("No events found after preprocessing. Try another city, increase max-records, or adjust lookback-days.")

    print("Converting to Documents and splitting into chunks ...")
    docs = _df_to_documents(df)
    splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = splitter.split_documents(docs)

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise SystemExit("MISTRAL_API_KEY is missing. Put it in .env or export it.")

    print("Embedding and building FAISS index ...")
    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)

    os.makedirs(args.index_out, exist_ok=True)
    vectordb.save_local(args.index_out)
    print(f"Saved FAISS index to: {args.index_out} (chunks={len(chunks)}, docs={len(docs)}, rows={len(df)})")


if __name__ == "__main__":
    main()
