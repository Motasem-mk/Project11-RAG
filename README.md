
# Concevez et déployez un système RAG — POC (OpenAgenda → FAISS → LangChain + Mistral)

A Retrieval-Augmented Generation (RAG) assistant that recommends cultural events from **OpenAgenda** data.
Stack: **Python · LangChain · FAISS · Mistral**.

* **Preprocess** OpenAgenda events (≤ 1 year, city filter, HTML strip)
* **Index** with **FAISS** using **mistral-embed**
* **Chatbot** answers strictly from retrieved context
* **Evaluate** retrieval (Recall\@k) and generation (Exact/Contains) on a frozen snapshot

> Optional artifacts (if present later): `docs/technical_report.pdf`, `slides/presentation.pptx`.

---

## Architecture / Workflow

```mermaid
flowchart LR
  subgraph A[Offline build]
    OA[(OpenAgenda API)] --> PP[Preprocess\n≤ 1 year · city · HTML strip]
    PP --> SNAP[Parquet snapshot\n(data/snapshots/events_eval.parquet)]
    SNAP --> CHUNK[Chunk texts]
    CHUNK --> EMB[Embeddings\nmistral-embed]
    EMB --> IDX[FAISS index\n(data/index/faiss)]
  end

  subgraph B[Chat runtime (RAG)]
    Q[User question (FR/EN)] --> RET[Retriever (FAISS)\nMMR · k]
    IDX -. load .-> RET
    RET --> CTX[Context formatting]
    CTX --> LLM[Mistral Chat\n(medium→large→small fallback)]
    LLM --> ANS[Answer strictly from context\n(or “I don't know.”)]
  end

  subgraph C[Evaluation]
    QR[qa_rules.csv] --> ER[Retrieval eval\nRecall@k]
    SNAP --> ER
    QA[qa_annotated.csv] --> EG[Generation eval\nExact/Contains + snapping]
    IDX -. load .-> EG
    EG --> RPT[gen_eval_report.csv]
  end

  ENV[.env (MISTRAL_API_KEY)] -.-> EMB
  ENV -.-> LLM
```

ASCII fallback:

```
OpenAgenda API → Preprocess (≤1 year, city, HTML strip) → Snapshot (Parquet)
→ Chunk → Embeddings (mistral-embed) → FAISS index

User Question → Retriever (FAISS, MMR, k) → Context → Mistral Chat → Answer
Retrieval eval: qa_rules.csv + Snapshot → Recall@k
Generation eval: qa_annotated.csv + Index → Exact/Contains (+ snapping) → gen_eval_report.csv
.env → MISTRAL_API_KEY for embeddings & chat
```

---

## Project structure

```
project_root/
├── .env.example
├── requirements.txt
├── README.md
│
├── data/
│   ├── eval/
│   │   ├── qa_rules.csv              # retrieval gold rules (20)
│   │   ├── qa_annotated.csv          # 15 FR Q/A for generation eval
│   │   └── gen_eval_report.csv       # (generated) gen eval results
│   ├── index/
│   │   └── faiss/                    # (generated) FAISS index
│   └── snapshots/
│       └── events_eval.parquet       # (generated) frozen snapshot
│
├── scripts/
│   ├── make_snapshot.py              # build snapshot (Parquet)
│   ├── eval_retrieval.py             # Recall@k on qa_rules.csv
│   └── eval_generation.py            # Gen eval (Exact/Contains) + snapping + fallback
│
├── src/
│   ├── data/preprocess_openagenda.py # fetch & clean OpenAgenda
│   ├── index/build_faiss.py          # chunk → embed (Mistral) → FAISS
│   └── rag/rag_pipeline.py           # CLI chatbot (FAISS + Mistral)
│
└── tests/
    └── test_vector_time_city.py      # unit test: ≤1 year + city, FAISS round-trip
```

---

## Setup

**Requirements**

* Python **3.9+**
* A **Mistral** API key

**Install**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env:
# MISTRAL_API_KEY=your_key_here
```

> You may see a tokenizer warning from `langchain_mistralai`; it’s harmless. (Optional) set `HF_TOKEN` for accurate batch sizing.

---

## 1) Build a reproducible snapshot (Parquet)

```bash
python scripts/make_snapshot.py \
  --city Paris \
  --max-records 4000 \
  --lookback-days 365 \
  --out data/snapshots/events_eval.parquet
```

---

## 2) Build the FAISS index

```bash
python -m src.index.build_faiss \
  --city Paris \
  --max-records 9000 \
  --lookback-days 365 \
  --chunk-size 800 \
  --chunk-overlap 120 \
  --index-out data/index/faiss
```

---

## 3) Retrieval evaluation (Recall\@k)

```bash
python scripts/eval_retrieval.py \
  --qa data/eval/qa_rules.csv \
  --snapshot data/snapshots/events_eval.parquet \
  --index data/index/faiss \
  --klist 10,20
```

**Expected (sample run):**

```
Recall@10: 20/20 = 1.000
Recall@20: 20/20 = 1.000
```

---

## 4) Generation evaluation (Exact / Contains)

Robust gen eval with:

* **MMR** retrieval (`k=30`, `fetch_k=60`, λ=0.5)
* **Snapping** LLM output to nearest title/venue/ZIP from retrieved context
* **Normalization** (accents/emoji/punctuation)
* **Multiple golds** via `A||B`
* **Model fallback** (medium → large → small) for capacity bursts

```bash
python scripts/eval_generation.py \
  --qa data/eval/qa_annotated.csv \
  --index data/index/faiss \
  --k 30 \
  --out data/eval/gen_eval_report.csv
```

**Expected (sample run):**

```
ExactMatch: 15/15 = 1.000
ContainsGold: 15/15 = 1.000
```

Output: `data/eval/gen_eval_report.csv`

---

## 5) Run the chatbot (CLI)

```bash
python -m src.rag.rag_pipeline \
  --index data/index/faiss \
  --k 12 \
  --chat-size medium
```

The bot responds **only** from retrieved context (FR/EN). If missing, it says **“I don't know.”**

---

## 6) Tests

```bash
pytest -q
```

Covers:

* **≤ 1 year** & **city** filter correctness
* Schema & HTML stripping
* FAISS save → load → retrieve (with a fake embedder)

---

## Repro commands (quick copy-paste)

```bash
# 1) Snapshot
python scripts/make_snapshot.py --city Paris --max-records 4000 --lookback-days 365 --out data/snapshots/events_eval.parquet

# 2) Index
python -m src.index.build_faiss --city Paris --max-records 9000 --lookback-days 365 --chunk-size 800 --chunk-overlap 120 --index-out data/index/faiss

# 3) Retrieval eval
python scripts/eval_retrieval.py --qa data/eval/qa_rules.csv --snapshot data/snapshots/events_eval.parquet --index data/index/faiss --klist 10,20

# 4) Generation eval
python scripts/eval_generation.py --qa data/eval/qa_annotated.csv --index data/index/faiss --k 30 --out data/eval/gen_eval_report.csv

# 5) Chat
python -m src.rag.rag_pipeline --index data/index/faiss --k 12 --chat-size medium
```

---

## Troubleshooting

* **`service_tier_capacity_exceeded (3505)` / rate limit**
  The gen eval uses model fallback + retries. Re-run the command; it will switch tiers automatically.

* **Tokenizer warning (Hugging Face)**
  Harmless; set `HF_TOKEN` to improve batch sizing (optional).

* **“MISTRAL\_API\_KEY not set”**
  Ensure `.env` contains `MISTRAL_API_KEY` and your shell has the venv activated.

---

## Deliverables checklist


* Code (`src/`, `scripts/`, `tests/`)
* `.env.example`, `requirements.txt`, `README.md`
* `data/snapshots/events_eval.parquet`
* `data/eval/qa_rules.csv`
* `data/eval/qa_annotated.csv`
* `data/eval/gen_eval_report.csv`
* `docs/technical_report.pdf` (5–10 pp)

---

## License

Internal educational POC.

---
