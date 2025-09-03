# tests/test_vector_time_city.py
"""
Minimal compliance test:
- Build FAISS index from preprocessed events (monkeypatched API).
- Assert retrieved docs are (a) in the selected city, (b) not older than 1 year.

Run:
    PYTHONPATH=. pytest -q
"""

from datetime import datetime, timedelta, timezone
import json
import pandas as pd
import pytest

from langchain_core.embeddings import Embeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from src.data.preprocess_openagenda import preprocess_events


# ---- Minimal fake embeddings (no network calls) ----
class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(t) % 7), 1.0] for t in texts]
    def embed_query(self, text):
        return [float(len(text) % 7), 1.0]


# ---- Helpers to fake the OpenDataSoft API ----
def _fake_payload(city: str):
    now = datetime.now(timezone.utc)
    def rec(days_ago: int, uid: str):
        start = (now - timedelta(days=days_ago)).isoformat()
        end = (now - timedelta(days=days_ago - 1)).isoformat()
        return {
            "fields": {
                "uid": uid,
                "title_fr": f"Événement {uid}",
                "location_city": city,
                "location_postalcode": "75001",
                "location_name": "Lieu",
                "canonicalurl": f"https://example.org/{uid}",
                "keywords_fr": ["famille"],       # simple tag so we can query "famille"
                "firstdate_begin": start,
                "firstdate_end": end,
                "longdescription_fr": "<p>desc</p>",
            }
        }

    # One recent event (30 days ago) -> should pass filter
    # One old event (500 days ago)  -> should be filtered out by lookback
    return {
        "records": [
            {"fields": rec(30, "u_recent")["fields"]},
            {"fields": rec(500, "u_old")["fields"]},
        ]
    }


class _Resp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.text = json.dumps(payload)
    def raise_for_status(self): pass
    def json(self): return self._payload


def test_vector_db_respects_city_and_lookback(monkeypatch):
    city = "Paris"
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=365)

    # Patch requests.get so preprocess_events pulls our fixed payload
    import requests
    monkeypatch.setattr(
        requests, "get",
        lambda *a, **k: _Resp(_fake_payload(city))
    )

    # 1) Preprocess (applies lookback filter)
    df = preprocess_events(city=city, max_records=50, rows=50, lookback_days=365)
    assert not df.empty, "Expected at least one event after lookback filter"
    # Ensure old item was filtered out
    assert (df["start_utc"] >= cutoff).all(), "Found an event older than 1 year after preprocessing"
    assert (df["city"] == city).all(), "Found an event outside the selected city after preprocessing"

    # 2) Build a minimal FAISS index from the preprocessed rows (no network)
    docs = []
    for _, r in df.iterrows():
        docs.append(
            Document(
                page_content=f"Title: {r['title']}\nCity: {r['city']}\nTags: {r['tags']}\n",
                metadata={
                    "uid": r["uid"],
                    "city": r["city"],
                    "start_utc": r["start_utc"],
                },
            )
        )
    vs = FAISS.from_documents(docs, embedding=FakeEmbeddings())

    # 3) Retrieve and assert constraints on the actual vector DB results
    hits = vs.similarity_search("famille", k=5)
    assert hits, "Expected at least one retrieval result from FAISS"

    for d in hits:
        m = d.metadata or {}
        assert m.get("city") == city, "Retrieved a doc from a different city"
        su = pd.to_datetime(m.get("start_utc"), utc=True, errors="coerce")
        assert su is not None and su >= cutoff, "Retrieved a doc older than 1 year"
