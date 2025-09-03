# scripts/eval_generation.p
#!/usr/bin/env python3
"""
Generation eval (scalable, no per-item tweaks):
- Loads FAISS index and qa_annotated.csv
- Retrieves top-k context and asks Mistral strictly from CONTEXT
- Post-process: snap model answer to the nearest TITLE/VENUE/ZIP found in retrieved context
- Scores:
    * ExactMatch (normalized)
    * ContainsGold (substring)
    * Postal codes: digits-only
- Accepts multiple golds via "A||B"
- Robust retrieval (MMR) + model fallback (medium→large→small)

Run:
  python scripts/eval_generation.py \
    --qa data/eval/qa_annotated.csv \
    --index data/index/faiss \
    --k 30 \
    --out data/eval/gen_eval_report.csv
"""

import os
import csv
import argparse
import re
import time
import difflib
from typing import List, Tuple, Set

from dotenv import load_dotenv, find_dotenv
from unidecode import unidecode
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate


# ---------- normalization helpers ----------
EMOJI_RE = re.compile(
    "["                     # general emoji block
    "\U0001F300-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]", flags=re.UNICODE
)

def strip_emoji(s: str) -> str:
    return EMOJI_RE.sub("", s or "")

def norm_basic(s: str) -> str:
    """Casefold + remove accents + collapse whitespace + keep simple punctuation."""
    if not isinstance(s, str):
        return ""
    s = strip_emoji(s)
    s = unidecode(s)                   # remove accents
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\-’'!/.:]", "", s)
    return s

def norm_ultra(s: str) -> str:
    """Stronger: letters/digits/spaces only."""
    if not isinstance(s, str):
        return ""
    s = strip_emoji(s)
    s = unidecode(s)
    s = s.strip().lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def strip_trailing_punct(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"[!?.:;]+$", "", s.strip())

def extract_postal5(s: str) -> str:
    m = re.search(r"\b\d{5}\b", s or "")
    return m.group(0) if m else ""


# ---------- LLM prompt ----------
BASE_SYSTEM = (
    "You are an events assistant.\n"
    "Use ONLY the provided CONTEXT to answer.\n"
    "If the answer is not in the CONTEXT, say: \"I don't know.\""
)

USER_PROMPT = (
    "Question: {question}\n\n"
    "CONTEXT:\n{context}\n\n"
    "Answer concisely using only facts from CONTEXT.\n"
    "Reply with a SINGLE short phrase: the exact title, venue name, or 5-digit postal code ONLY. "
    "Do not add any extra words or explanations."
)

def format_context(docs, max_chars=550):
    parts = []
    for d in docs:
        m = d.metadata or {}
        snippet = (d.page_content or "")[:max_chars]
        parts.append(
            f"Title: {m.get('title')}\n"
            f"City: {m.get('city')}  PostalCode: {m.get('postal_code')}\n"
            f"Start: {m.get('start_utc')}  End: {m.get('end_utc')}\n"
            f"Link: {m.get('permalink') or m.get('website')}\n"
            f"{snippet}"
        )
    return "\n\n---\n\n".join(parts)


# ---------- Answer snapping to context ----------
def candidate_strings_from_docs(docs) -> Set[str]:
    cands: Set[str] = set()
    for d in docs:
        m = d.metadata or {}
        for k in ("title", "venue"):
            v = (m.get(k) or "").strip()
            if v:
                cands.add(strip_trailing_punct(v))
        # postal code
        z = extract_postal5(str(m.get("postal_code") or ""))
        if z:
            cands.add(z)
    return cands

def snap_to_context(ans: str, docs, threshold: float = 0.88) -> str:
    """
    Map model answer to the nearest canonical candidate from retrieved context.
    Uses difflib ratio on ultra-normalized strings. Returns the snapped candidate or original ans.
    """
    if not ans:
        return ans
    cands = candidate_strings_from_docs(docs)
    if not cands:
        return ans

    a_u = norm_ultra(ans)
    best: Tuple[float, str] = (0.0, ans)
    for c in cands:
        r = difflib.SequenceMatcher(None, a_u, norm_ultra(c)).ratio()
        if r > best[0]:
            best = (r, c)
    return best[1] if best[0] >= threshold else ans


# ---------- LLM fallback ----------
def call_llm_with_fallback(msgs, api_key: str, models: List[str], max_retries: int = 2, sleep_s: float = 1.0) -> str:
    last_err = None
    for model in models:
        for attempt in range(max_retries):
            try:
                llm = ChatMistralAI(model=model, temperature=0.0, api_key=api_key)
                resp = llm.invoke(msgs)
                return (resp.content or "").strip()
            except Exception as e:
                emsg = str(e).lower()
                if any(k in emsg for k in ["capacity", "rate limit", "3505", "service_tier_capacity_exceeded", "429"]):
                    last_err = e
                    time.sleep(sleep_s * (attempt + 1))
                    continue
                raise
    if last_err:
        raise last_err
    raise RuntimeError("All models failed.")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="data/eval/qa_annotated.csv")
    ap.add_argument("--index", default="data/index/faiss")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--out", default="data/eval/gen_eval_report.csv")
    ap.add_argument("--retrieval", choices=["mmr", "similarity"], default="mmr")
    ap.add_argument("--fetch-k", type=int, default=60)
    ap.add_argument("--mmr-lambda", type=float, default=0.5)
    ap.add_argument("--models", default="mistral-medium-latest,mistral-large-latest,mistral-small-latest")
    ap.add_argument("--snap-threshold", type=float, default=0.88, help="0–1 similarity threshold for snapping")
    args = ap.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise SystemExit("MISTRAL_API_KEY not set")

    emb = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)
    vs = FAISS.load_local(args.index, embeddings=emb, allow_dangerous_deserialization=True)

    if args.retrieval == "mmr":
        retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": args.k, "fetch_k": max(args.fetch_k, args.k * 3), "lambda_mult": args.mmr_lambda},
        )
    else:
        retriever = vs.as_retriever(search_kwargs={"k": args.k})

    prompt = ChatPromptTemplate.from_messages([("system", BASE_SYSTEM), ("human", USER_PROMPT)])
    models_chain = [m.strip() for m in args.models.split(",") if m.strip()]

    # read QA
    rows = []
    with open(args.qa, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            q = (r.get("question") or "").strip()
            gold = (r.get("answer_gold") or r.get("gold") or "").strip()
            if not q:
                continue
            rows.append({"id": r.get("id") or "", "question": q, "gold": gold})

    out_rows = []
    n_eval = 0
    exact_hits = 0
    contains_hits = 0

    for r in rows:
        q, gold = r["question"], r["gold"]
        if not gold:
            out_rows.append({**r, "answer": "", "exact_match": "", "contains_gold": "", "note": "SKIPPED_EMPTY_GOLD"})
            continue

        docs = retriever.invoke(q)
        if not docs:
            out_rows.append({**r, "answer": "I don't know.", "exact_match": 0, "contains_gold": 0, "note": "NO_CONTEXT"})
            n_eval += 1
            continue

        ctx = format_context(docs)
        msgs = prompt.format_messages(question=q, context=ctx)
        ans_raw = call_llm_with_fallback(msgs, api_key, models_chain)

        # snap to nearest canonical value from retrieved docs
        ans = snap_to_context(ans_raw, docs, threshold=args.snap_threshold)

        # ----- scoring -----
        n_eval += 1

        # postal: digits only
        if re.fullmatch(r"\s*\d{5}\s*", gold or ""):
            gold5 = re.search(r"\d{5}", gold).group(0)
            ans5 = extract_postal5(ans)
            exact = int(ans5 == gold5)
            contains = exact
        else:
            raw_variants = [g.strip() for g in (gold.split("||")) if g.strip()] or [gold.strip()]
            all_gold_variants = set()
            for gv in raw_variants:
                all_gold_variants.add(gv)
                all_gold_variants.add(strip_trailing_punct(gv))

            gold_norms = {norm_basic(gv) for gv in all_gold_variants if gv}
            gold_ultras = {norm_ultra(gv) for gv in all_gold_variants if gv}

            ans_norm = norm_basic(ans)
            ans_ultra = norm_ultra(ans)

            exact = int((ans_norm in gold_norms) or (ans_ultra in gold_ultras))
            contains = int(any(gv in ans_norm for gv in gold_norms) or any(gv in ans_ultra for gv in gold_ultras))

        exact_hits += exact
        contains_hits += contains

        out_rows.append({
            "id": r["id"], "question": q, "gold": gold,
            "answer": ans, "exact_match": exact, "contains_gold": contains, "note": "" if exact or contains else "MISS"
        })

    # write CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","question","gold","answer","exact_match","contains_gold","note"])
        w.writeheader()
        w.writerows(out_rows)

    if n_eval > 0:
        print(f"ExactMatch: {exact_hits}/{n_eval} = {exact_hits / n_eval:.3f}")
        print(f"ContainsGold: {contains_hits}/{n_eval} = {contains_hits / n_eval:.3f}")
    else:
        print("No evaluable rows (empty gold answers).")

if __name__ == "__main__":
    main()
