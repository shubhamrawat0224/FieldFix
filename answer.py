#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib import error, request

from bm25 import build_chunks, rank_chunks
from hybrid import build_rrf_ranking, chunk_key
from loader import load_corpus
from vector_search import vector_search

SCORECARD_PATH = Path(__file__).resolve().parent / "scorecard.csv"


def load_env_file() -> None:
    dotenv_path = Path(__file__).resolve().parent / ".env"
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_env_file()


def openrouter_completion(prompt: str, model: str | None = None) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    payload = {
        "model": model or os.environ.get("OPENAI_MODEL", "openai/gpt-5-mini"),
        "messages": [
            {"role": "system", "content": "You answer only from the provided context. Cite the chunk IDs and document titles. If the context does not contain the answer, say exactly 'I don't have enough information'. Do not use external knowledge or invent facts."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }

    req = request.Request(
        f"{os.environ.get('OPENAI_BASE_URL', 'https://openrouter.ai/api/v1').rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Completion request failed: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Completion connection failed: {exc.reason}") from exc

    return data["choices"][0]["message"]["content"].strip()


def get_doc_titles(corpus_dir: Path) -> dict[str, str]:
    return {doc["doc_id"]: doc["title"] for doc in load_corpus(corpus_dir)}


def pick_best_strategy(scorecard_path: Path = SCORECARD_PATH) -> str:
    if not scorecard_path.exists():
        return "hybrid"

    with scorecard_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("question", "").strip() != "BEST_STRATEGY":
                continue
            for key, value in row.items():
                if not key or key == "question":
                    continue
                if value and ":" in value:
                    strategy_key, _ = value.split(":", 1)
                    return strategy_key.split("@", 1)[0]
    return "hybrid"


def select_grounded_evidence(question: str, top_n: int = 5) -> list[dict[str, Any]]:
    corpus_dir = Path(__file__).resolve().parent / "corpus"
    docs = load_corpus(corpus_dir)
    chunk_lookup = {
        chunk_key(chunk["doc_id"], chunk["chunk_id"]): chunk
        for doc in docs
        for chunk in build_chunks(corpus_dir, strategy="fixed", size=60, overlap=10)
        if chunk["doc_id"] == doc["doc_id"]
    }

    chunks = build_chunks(corpus_dir, strategy="fixed", size=60, overlap=10)
    bm25_results = rank_chunks(question, chunks, k=10)
    vector_results = vector_search(question, k=10, collection_name="fieldfix_corpus")
    fused = build_rrf_ranking(bm25_results, vector_results, k=60)
    preferred = pick_best_strategy()

    if preferred == "bm25":
        candidate_items = [
            {"key": chunk_key(str(chunk["doc_id"]), chunk["chunk_id"]), "doc_id": chunk["doc_id"], "chunk_id": chunk["chunk_id"], "title": chunk.get("title", ""), "preview": chunk["text"][:180]}
            for _, chunk in bm25_results
        ]
    elif preferred == "vector":
        candidate_items = [
            {
                "key": chunk_key(str(hit.get("doc_id", "")), hit.get("chunk_id")),
                "doc_id": hit.get("doc_id"),
                "chunk_id": hit.get("chunk_id"),
                "title": hit.get("title", ""),
                "preview": str(hit.get("preview", "")),
            }
            for hit in vector_results
        ]
    elif preferred == "rerank":
        candidate_items = [
            {
                "key": chunk_key(str(hit.get("doc_id", "")), hit.get("chunk_id")),
                "doc_id": hit.get("doc_id"),
                "chunk_id": hit.get("chunk_id"),
                "title": hit.get("title", ""),
                "preview": str(hit.get("preview", "")),
            }
            for hit in vector_results
        ]
    else:
        candidate_items = [
            {
                "key": item["key"],
                "doc_id": item.get("doc_id"),
                "chunk_id": item.get("chunk_id"),
                "title": item.get("title", ""),
                "preview": item.get("preview", ""),
            }
            for item in fused
        ]

    doc_titles = get_doc_titles(corpus_dir)
    evidence: list[dict[str, Any]] = []
    for item in candidate_items[:top_n]:
        key = item["key"]
        chunk = chunk_lookup.get(key)
        if chunk is None:
            continue
        evidence.append(
            {
                "doc_id": item.get("doc_id") or chunk["doc_id"],
                "chunk_id": item.get("chunk_id") or chunk["chunk_id"],
                "title": item.get("title") or doc_titles.get(item.get("doc_id") or chunk["doc_id"], ""),
                "text": chunk["text"],
            }
        )
    return evidence


def grounded_answer(question: str, top_n: int = 5) -> tuple[str, list[dict[str, Any]]]:
    evidence = select_grounded_evidence(question, top_n=top_n)
    if not evidence:
        return "I don't have enough information", []

    context_parts = []
    for idx, item in enumerate(evidence, start=1):
        context_parts.append(
            f"[{idx}] Document: {item['title']} | Chunk ID: {item['doc_id']}:{item['chunk_id']}\n{item['text']}"
        )

    prompt = (
        f"Question: {question}\n\n"
        "Use ONLY the context below. Answer from the context. Cite chunk IDs and document titles. "
        "If the context does not contain the answer, say exactly: I don't have enough information.\n\n"
        + "\n\n---\n\n".join(context_parts)
    )

    answer = openrouter_completion(prompt)
    return answer, evidence


def print_answer_result(question: str, answer: str, evidence: list[dict[str, Any]]) -> None:
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print("Evidence used:")
    for item in evidence:
        print(f"- {item['title']} | {item['doc_id']}:{item['chunk_id']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Grounded generation from retrieved chunks")
    parser.add_argument("--question", default="What should the technician do if the site is unsafe and the customer refuses access?", help="Question to answer")
    parser.add_argument("--top-n", type=int, default=5, help="How many hybrid chunks to pass as context")
    args = parser.parse_args()

    answer, evidence = grounded_answer(args.question, top_n=args.top_n)
    print_answer_result(args.question, answer, evidence)


if __name__ == "__main__":
    main()
