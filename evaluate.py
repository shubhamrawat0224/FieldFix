#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from bm25 import build_chunks, rank_chunks
from hybrid import hybrid_search
from rerank import before_after_ranking
from vector_search import vector_search

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.json"
SCORECARD_CSV_PATH = Path(__file__).resolve().parent / "scorecard.csv"
SCORECARD_MD_PATH = Path(__file__).resolve().parent / "scorecard.md"

STRATEGIES = ["bm25", "vector", "hybrid", "rerank"]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def fact_matches(fact: str, chunk_text: str) -> bool:
    return normalize_text(fact) in normalize_text(chunk_text)


def build_chunk_lookup(corpus_dir: Path) -> dict[str, dict[str, Any]]:
    chunks = build_chunks(corpus_dir, strategy="fixed", size=60, overlap=10)
    lookup: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        key = f"{chunk.get('doc_id')}:{chunk.get('chunk_id')}"
        lookup[key] = chunk
    return lookup


def resolve_chunks_from_results(results: list[Any], chunk_lookup: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for result in results:
        if isinstance(result, tuple):
            _, chunk = result
            if isinstance(chunk, dict):
                resolved.append(chunk)
                continue
        if isinstance(result, dict):
            doc_id = result.get("doc_id")
            chunk_id = result.get("chunk_id")
            if doc_id is not None and chunk_id is not None:
                key = f"{doc_id}:{chunk_id}"
                chunk = chunk_lookup.get(key)
                if chunk is not None:
                    resolved.append(chunk)
                    continue
                text = str(result.get("text") or result.get("preview") or "")
                resolved.append({"doc_id": doc_id, "chunk_id": chunk_id, "text": text})
    return resolved


def evaluate_query_at_k(question: str, required_facts: list[str], strategy: str, k: int, corpus_dir: Path) -> tuple[int, int]:
    if strategy == "bm25":
        chunks = build_chunks(corpus_dir, strategy="fixed", size=60, overlap=10)
        ranking = rank_chunks(question, chunks, k=k)
        retrieved = [chunk for _, chunk in ranking]
    elif strategy == "vector":
        hits = vector_search(question, k=k, collection_name="fieldfix_corpus")
        retrieved = resolve_chunks_from_results(hits, build_chunk_lookup(corpus_dir))
    elif strategy == "hybrid":
        hits = hybrid_search(
            query=question,
            bm25_k=10,
            vector_k=10,
            strategy="fixed",
            size=60,
            overlap=10,
            collection_name="fieldfix_corpus",
            k=k,
        )
        retrieved = resolve_chunks_from_results(hits, build_chunk_lookup(corpus_dir))
    elif strategy == "rerank":
        before, after = before_after_ranking(question, n=max(10, k))
        retrieved = resolve_chunks_from_results(after[:k], build_chunk_lookup(corpus_dir))
        if not retrieved:
            retrieved = resolve_chunks_from_results(before[:k], build_chunk_lookup(corpus_dir))
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    matched = set()
    for fact in required_facts:
        for chunk in retrieved:
            chunk_text = str(chunk.get("text") or "")
            if fact_matches(fact, chunk_text):
                matched.add(fact)
                break

    return len(matched), len(required_facts)


def load_eval_set(path: Path = EVAL_SET_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_question_rows(eval_set: list[dict[str, Any]], corpus_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in eval_set:
        question = item["question"]
        required_facts = item["required_facts"]
        row: dict[str, Any] = {"question": question}
        for strategy in STRATEGIES:
            for k in (3, 5):
                matched, total = evaluate_query_at_k(question, required_facts, strategy, k, corpus_dir)
                row[f"{strategy}@{k}"] = f"{matched}/{total}"
        rows.append(row)
    return rows


def compute_totals(rows: list[dict[str, Any]]) -> dict[str, str]:
    totals: dict[str, str] = {}
    for strategy in STRATEGIES:
        for k in (3, 5):
            key = f"{strategy}@{k}"
            matched = 0
            total = 0
            for row in rows:
                value = row.get(key, "0/0")
                if "/" not in value:
                    continue
                left, right = value.split("/", 1)
                matched += int(left)
                total += int(right)
            totals[key] = f"{matched}/{total}"
    return totals


def total_summary_label(strategy: str, k: int, totals: dict[str, str]) -> str:
    value = totals.get(f"{strategy}@{k}", "0/0")
    return f"{strategy}: {value} facts @ {k}"


def pick_best_strategy(scorecard_path: Path = SCORECARD_CSV_PATH) -> str:
    if not scorecard_path.exists():
        raise FileNotFoundError(f"Scorecard not found: {scorecard_path}")

    with scorecard_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        best_name = "hybrid"
        best_score = -1
        for row in reader:
            if row.get("question", "").strip() != "BEST_STRATEGY":
                continue
            for strategy_name in STRATEGIES:
                value = row.get(f"{strategy_name}@5", "")
                if not value or ":" not in value:
                    continue
                _, score = value.split(":", 1)
                score_value = int(score.split("/", 1)[0])
                if score_value > best_score:
                    best_score = score_value
                    best_name = strategy_name
        return best_name


def write_csv(rows: list[dict[str, Any]], totals: dict[str, str], output_path: Path = SCORECARD_CSV_PATH) -> None:
    fieldnames = ["question", "bm25@3", "bm25@5", "vector@3", "vector@5", "hybrid@3", "hybrid@5", "rerank@3", "rerank@5"]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

        totals_row = {"question": "TOTAL"}
        for strategy in STRATEGIES:
            for k in (3, 5):
                totals_row[f"{strategy}@{k}"] = totals.get(f"{strategy}@{k}", "0/0")
        writer.writerow(totals_row)

        best_key = max(
            [f"{strategy}@5" for strategy in STRATEGIES],
            key=lambda key: int(totals.get(key, "0/0").split("/", 1)[0]),
        )
        best_name = best_key.split("@", 1)[0]
        best_value = totals.get(best_key, "0/0")
        summary_row = {"question": f"BEST_STRATEGY"}
        for key in fieldnames[1:]:
            summary_row[key] = ""
        summary_row[best_key] = f"{best_name}: {best_value} facts @ 5"
        writer.writerow(summary_row)


def write_markdown(rows: list[dict[str, Any]], totals: dict[str, str], output_path: Path = SCORECARD_MD_PATH) -> None:
    best_key = max(
        [f"{strategy}@5" for strategy in STRATEGIES],
        key=lambda key: int(totals.get(key, "0/0").split("/", 1)[0]),
    )
    strategy_label = best_key.split("@", 1)[0].capitalize()
    best_value = totals.get(best_key, "0/0")

    lines: list[str] = []
    lines.append("# Retrieval evaluation scorecard")
    lines.append("")
    lines.append("| Question | BM25@3 | BM25@5 | Vector@3 | Vector@5 | Hybrid@3 | Hybrid@5 | Rerank@3 | Rerank@5 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            f"| {row['question']} | {row.get('bm25@3', '0/0')} | {row.get('bm25@5', '0/0')} | {row.get('vector@3', '0/0')} | {row.get('vector@5', '0/0')} | {row.get('hybrid@3', '0/0')} | {row.get('hybrid@5', '0/0')} | {row.get('rerank@3', '0/0')} | {row.get('rerank@5', '0/0')} |"
        )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    lines.append(
        f"| TOTAL | {totals.get('bm25@3', '0/0')} | {totals.get('bm25@5', '0/0')} | {totals.get('vector@3', '0/0')} | {totals.get('vector@5', '0/0')} | {totals.get('hybrid@3', '0/0')} | {totals.get('hybrid@5', '0/0')} | {totals.get('rerank@3', '0/0')} | {totals.get('rerank@5', '0/0')} |"
    )
    lines.append("")
    lines.append(f"- {total_summary_label('bm25', 5, totals)}")
    lines.append(f"- {total_summary_label('vector', 5, totals)}")
    lines.append(f"- {total_summary_label('hybrid', 5, totals)}")
    lines.append(f"- {total_summary_label('rerank', 5, totals)}")
    lines.append("")
    lines.append(f"Best strategy at top-5: {strategy_label} ({best_value} facts)")
    lines.append("")
    lines.append("## Why a small eval set can mislead you")
    lines.append("- A tiny eval set is extremely sensitive to wording, synonyms, and exact fact phrasing.")
    lines.append("- One or two questions can overstate a strategy's quality when the corpus contains a few highly similar policies.")
    lines.append("- A small set can hide edge cases like missing signature rules, access restrictions, or contract exceptions that only appear under different phrasings.")
    lines.append("- The right approach is to treat the eval set as a quick sanity check, then add more realistic technician prompts and review retrieval failures before trusting a winner.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    corpus_dir = Path(__file__).resolve().parent / "corpus"
    rows = compute_question_rows(load_eval_set(), corpus_dir)
    totals = compute_totals(rows)
    write_csv(rows, totals)
    write_markdown(rows, totals)
    print(f"Wrote {SCORECARD_CSV_PATH}")
    print(f"Wrote {SCORECARD_MD_PATH}")
    best_key = max(
        [f"{strategy}@5" for strategy in STRATEGIES],
        key=lambda key: int(totals.get(key, "0/0").split("/", 1)[0]),
    )
    print(f"Best strategy at top-5: {best_key} -> {totals.get(best_key, '0/0')}")


if __name__ == "__main__":
    main()
