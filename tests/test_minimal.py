# tests/test_minimal.py

'''
These tests cover (minimal but meaningful):

Time window & city rule (≤ 1 year; city filter).

Schema & cleaning (columns exist; HTML stripped; tags are strings).

FAISS round-trip (save → load → retrieve) with a fake, offline embedder.
-----------------------------------------------------------------------------------
How to Run the test 

— one-off run
From your project root:
PYTHONPATH=. pytest -q
'''

from datetime import datetime, timedelta, timezone
import types
import json
import pandas as pd
import pytest

from src.data.preprocess_openagenda import preprocess_events

from langchain_core.embeddings import Embeddings

try:
    from langchain_community.vectorstores import FAISS
except ModuleNotFoundError:
    from langchain.vectorstores import FAISS

# ---------- Helpers ----------

def _fake_api_json(city: str, days_ago: int = 30):
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=days_ago)).isoformat()
    end = (now - timedelta(days=days_ago-1)).isoformat()
    return {
        "records": [
            {
                "fields": {
                    "uid": "u1",
                    "title_fr": "Événement test",
                    "location_city": city,
                    "location_postalcode": "75001",
                    "location_name": "Lieu Test",
                    "canonicalurl": "https://example.org/e1",
                    "keywords_fr": ["famille", "enfants"],
                    "firstdate_begin": start,
                    "firstdate_end": end,
                    "longdescription_fr": "<p>Bonjour <b>monde</b></p>",
                }
            }
        ]
    }

class _Resp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.text = json.dumps(payload)
    def raise_for_status(self): pass
    def json(self): return self._payload

# Fake embeddings so we don't call any API
class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        # deterministic tiny vectors
        return [[float(len(t) % 7), 1.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text) % 7), 1.0]



# ---------- Tests ----------

def test_preprocess_timewindow_and_city(monkeypatch):
    city = "Paris"
    payload = _fake_api_json(city=city, days_ago=30)

    def _fake_get(url, params=None, timeout=30):
        return _Resp(payload)

    import requests
    monkeypatch.setattr(requests, "get", _fake_get)

    df = preprocess_events(city=city, max_records=200, lookback_days=365)
    assert not df.empty
    assert (df["city"] == city).all()

    cutoff = datetime.now(timezone.utc) - timedelta(days=365)
    assert (df["start_utc"] >= cutoff).all()

def test_preprocess_schema_and_cleaning(monkeypatch):
    city = "Paris"
    payload = _fake_api_json(city=city, days_ago=10)

    def _fake_get(url, params=None, timeout=30):
        return _Resp(payload)

    import requests
    monkeypatch.setattr(requests, "get", _fake_get)

    df = preprocess_events(city=city, max_records=50, lookback_days=365)

    # required columns exist
    required = [
        "uid","title","city","postal_code","venue","website","permalink",
        "tags","start_utc","end_utc","text"
    ]
    for col in required:
        assert col in df.columns

    # HTML stripped, tags normalized to string
    assert ">" not in df.loc[0, "text"]
    assert isinstance(df.loc[0, "tags"], str)

def test_faiss_save_load(tmp_path):
    # build a tiny FAISS index with FakeEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.schema import Document

    docs = [
        Document(page_content="Title: Fête des enfants\nCity: Paris\nTexte: jeux en famille.", metadata={"title":"Fête des enfants"}),
        Document(page_content="Title: Concert de femmes\nCity: Paris\nTexte: chorale féminine.", metadata={"title":"Concert de femmes"}),
    ]

    vec = FAISS.from_documents(docs, embedding=FakeEmbeddings())
    outdir = tmp_path / "faiss"
    outdir.mkdir(parents=True, exist_ok=True)
    vec.save_local(str(outdir))

    # reload & retrieve
    loaded = FAISS.load_local(str(outdir), embeddings=FakeEmbeddings(), allow_dangerous_deserialization=True)
    retriever = loaded.as_retriever(search_kwargs={"k": 2})

    hits = retriever.invoke("famille")
    assert len(hits) >= 1
