#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from urllib import error, request

from hybrid import hybrid_search


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
            {"role": "system", "content": "You are a strict reranker. Return only valid JSON in the exact shape requested."},
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
        raise RuntimeError(f"Reranker request failed: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Reranker connection failed: {exc.reason}") from exc

    return data["choices"][0]["message"]["content"].strip()


def rank_candidates_by_llm(query: str, candidates: list[dict]) -> list[dict]:
    if len(candidates) <= 1:
        return candidates

    label_list = []
    for idx, item in enumerate(candidates, start=1):
        label = f"{item['doc_id']}:{item['chunk_id']}"
        label_list.append(f"{idx}. {label} | title={item.get('title','')} | preview={item.get('preview','')[:200].replace(chr(10), ' ')}")

    prompt = (
        "Rank the candidate chunks by how directly they answer the user's query. "
        "Focus on policy correctness and operational relevance, not just keyword overlap. "
        "Return only valid JSON: an array of candidate labels in ranked order, for example [\"doc_a:3\", \"doc_b:1\"].\n\n"
        f"Query: {query}\n\nCandidates:\n" + "\n".join(label_list)
    )

    raw = openrouter_completion(prompt)
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    try:
        ordered_labels = json.loads(text)
    except json.JSONDecodeError:
        print(f"LLM rerank returned non-JSON: {raw}")
        return candidates

    if not isinstance(ordered_labels, list):
        print(f"LLM rerank returned unexpected JSON shape: {raw}")
        return candidates

    label_to_item = {f"{item['doc_id']}:{item['chunk_id']}": item for item in candidates}
    reranked = []
    for label in ordered_labels:
        if label in label_to_item:
            reranked.append(label_to_item[label])

    if len(reranked) != len(candidates):
        for item in candidates:
            label = f"{item['doc_id']}:{item['chunk_id']}"
            if label not in {f"{x['doc_id']}:{x['chunk_id']}" for x in reranked}:
                reranked.append(item)

    return reranked


def before_after_ranking(query: str, n: int = 10) -> tuple[list[dict], list[dict]]:
    hybrid = hybrid_search(
        query=query,
        bm25_k=n,
        vector_k=n,
        strategy="fixed",
        size=60,
        overlap=10,
        collection_name="fieldfix_corpus",
        k=60,
    )
    before = hybrid[:n]
    after = rank_candidates_by_llm(query, before)
    return before, after


def print_ranking(label: str, ranking: list[dict]) -> None:
    print(f"{label}")
    print(f"{'Rank':>4} | {'Doc ID':<30} | {'Chunk':>6} | {'Score':>8} | {'Preview'}")
    print("-" * 120)
    for idx, item in enumerate(ranking, start=1):
        preview = str(item.get("preview", ""))
        print(f"{idx:>4} | {str(item.get('doc_id')):<30} | {str(item.get('chunk_id')):>6} | {item.get('score', 0.0):>8.4f} | {preview[:120]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM reranking over the top hybrid hits")
    parser.add_argument("--query", default="what should technician do if customer refuses a signature after repair", help="Query to rerank")
    parser.add_argument("--n", type=int, default=10, help="How many top hybrid results to rerank")
    args = parser.parse_args()

    before, after = before_after_ranking(args.query, n=args.n)
    print_ranking("Before reranking", before)
    print("\n")
    print_ranking("After reranking", after)


if __name__ == "__main__":
    main()
