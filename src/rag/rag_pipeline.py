# src/rag/rag_pipeline.py

"""
RAG CLI for cultural events (FAISS + Mistral)

- Loads a FAISS index (built for one city with a bounded time window enforced at build time)
  and answers ONLY from retrieved context.
- Replies in the question’s language (FR/EN and others). Includes a small model-fallback chain.

Quick start:
    # 1) Build an index first (handled by src/index/build_faiss.py)
    #    Example (1-year lookback, includes future events present in the dataset):
    #    python -m src.index.build_faiss --city Paris --lookback-days 365 --index-out data/index/faiss

    # 2) Chat (point to the saved index):
    python -m src.rag.rag_pipeline --index data/index/faiss --k 12

Defaults:
  index=data/index/faiss, k=12, chat-size=medium

Note: “I don’t know” = the answer is not present in the retrieved CONTEXT (or outside the indexed time window).
"""

import os
import argparse
from typing import List

import httpx
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

BASE_SYSTEM = (
    "You are an events assistant.\n"
    "Use ONLY the provided CONTEXT to answer. "
    "If the answer is not present in the CONTEXT, say: \"I don't know.\" "
    "Reply strictly in the language of the question, even if the CONTEXT is in another language."
)

USER_PROMPT = (
    "Question: {question}\n\n"
    "CONTEXT:\n{context}\n\n"
    "Respond concisely with facts from the CONTEXT. Include titles, dates, venues, and links only if present."
)

SIZE_TO_MODEL = {"large": "mistral-large-latest", "medium": "mistral-medium-latest"}
EMERGENCY_SMALL = "mistral-small-latest"


def _format_context(docs) -> str:
    """Compact context: key metadata + small snippet."""
    chunks: List[str] = []
    for d in docs:
        m = d.metadata or {}
        snippet = d.page_content[:500]
        chunks.append(
            f"Title: {m.get('title')}\n"
            f"City: {m.get('city')}  PostalCode: {m.get('postal_code')}\n"
            f"Start: {m.get('start_utc')}  End: {m.get('end_utc')}\n"
            f"Link: {m.get('permalink') or m.get('website')}\n"
            f"{snippet}"
        )
    return "\n\n---\n\n".join(chunks)


def _make_llm(model_id: str, api_key: str) -> ChatMistralAI:
    return ChatMistralAI(
        model=model_id,
        temperature=0.1,
        api_key=api_key,
        max_retries=6,
        timeout=90,
    )


def main():
    # Robust .env loading from the current working directory (CLI, IDE, heredocs)
    load_dotenv(find_dotenv(usecwd=True))

    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, default="data/index/faiss")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--chat-size", choices=["large", "medium"], default="medium")
    args = ap.parse_args()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key or api_key.strip().lower() in {"your_mistral_api_key_here", "xxxx", "xxx"}:
        raise SystemExit("MISTRAL_API_KEY not set. Put it in .env or export it.")

    # embeddings must match build-time model family
    emb = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)
    vectordb = FAISS.load_local(args.index, embeddings=emb, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_kwargs={"k": args.k})

    # simple fallback chain: chosen size -> other size -> small
    primary = SIZE_TO_MODEL[args.chat_size]
    secondary = SIZE_TO_MODEL["large" if args.chat_size == "medium" else "medium"]
    models_chain = [primary, secondary, EMERGENCY_SMALL]

    print("\nType your questions (Ctrl+C to quit)\n")
    print(f"(Model order: {', '.join(models_chain)})\n")

    prompt = ChatPromptTemplate.from_messages([("system", BASE_SYSTEM), ("human", USER_PROMPT)])

    while True:
        try:
            q = input("You: ").strip()
            if not q:
                continue

            docs = retriever.invoke(q)
            if not docs:
                print("\nBot: No matching results in the current index (we keep a bounded time window). Try another phrasing.\n")
                continue

            ctx = _format_context(docs)
            msgs = prompt.format_messages(question=q, context=ctx)

            last_err = None
            for i, model in enumerate(models_chain):
                if i > 0:
                    print(f"\n[info] Switching to {model} and retrying...\n")
                llm = _make_llm(model, api_key)
                try:
                    resp = llm.invoke(msgs)
                    print(f"\nBot: {resp.content}\n")
                    break
                except httpx.HTTPStatusError as e:
                    last_err = e
                    status = getattr(e, "response", None).status_code if getattr(e, "response", None) else None
                    text = (getattr(e, "response", None).text or "").lower() if getattr(e, "response", None) else str(e).lower()
                    if status == 429 or "capacity" in text or "rate limit" in text:
                        continue  # try next model
                    raise  # non-capacity error
            else:
                print("\nBot: Sorry, all chat tiers are temporarily at capacity. Please try again shortly.\n")

        except KeyboardInterrupt:
            print("\nBye!")
            break


if __name__ == "__main__":
    main()
