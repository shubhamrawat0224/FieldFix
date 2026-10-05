#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from bm25 import build_chunks, rank_chunks
from vector_search import vector_search


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


def chunk_key(doc_id: str, chunk_id: object) -> str:
    return f"{doc_id}:{chunk_id}"


def build_rrf_ranking(
    bm25_results: list[tuple[float, dict]],
    vector_results: list[dict],
    k: int = 60,
) -> list[dict]:
    scores: dict[str, float] = {}
    metadata: dict[str, dict] = {}

    for rank, (_, chunk) in enumerate(bm25_results, start=1):
        key = chunk_key(str(chunk["doc_id"]), chunk["chunk_id"])
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        metadata[key] = {
            "doc_id": chunk["doc_id"],
            "chunk_id": chunk["chunk_id"],
            "title": chunk.get("title", ""),
            "preview": chunk["text"][:180] + ("..." if len(chunk["text"]) > 180 else ""),
        }

    for rank, hit in enumerate(vector_results, start=1):
        key = chunk_key(str(hit.get("doc_id", "")), hit.get("chunk_id"))
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        metadata.setdefault(
            key,
            {
                "doc_id": hit.get("doc_id"),
                "chunk_id": hit.get("chunk_id"),
                "title": hit.get("title", ""),
                "preview": str(hit.get("preview", "")),
            },
        )

    ranked = []
    for key, score in scores.items():
        info = metadata.get(key, {})
        ranked.append(
            {
                "key": key,
                "score": score,
                "doc_id": info.get("doc_id"),
                "chunk_id": info.get("chunk_id"),
                "title": info.get("title", ""),
                "preview": info.get("preview", ""),
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def hybrid_search(
    query: str,
    bm25_k: int = 10,
    vector_k: int = 10,
    strategy: str = "fixed",
    size: int = 60,
    overlap: int = 10,
    collection_name: str = "fieldfix_corpus",
    k: int = 60,
):
    chunks = build_chunks(Path(__file__).resolve().parent / "corpus", strategy=strategy, size=size, overlap=overlap)
    bm25_results = rank_chunks(query, chunks, k=bm25_k)
    vector_results = vector_search(query, k=vector_k, collection_name=collection_name)
    return build_rrf_ranking(bm25_results, vector_results, k=k)


def print_rrf_results(results: list[dict], query: str) -> None:
    print(f"Query: {query}")
    print(f"{'Rank':>4} | {'Score':>8} | {'Doc ID':<30} | {'Chunk':>6} | {'Preview'}")
    print("-" * 120)
    for idx, hit in enumerate(results[:10], start=1):
        preview = str(hit.get("preview", ""))
        print(f"{idx:>4} | {hit['score']:>8.4f} | {str(hit.get('doc_id')):<30} | {str(hit.get('chunk_id')):>6} | {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reciprocal Rank Fusion (RRF) over BM25 and vector results")
    parser.add_argument("--query", default="how long is the power module covered if it dies?", help="Query to search")
    parser.add_argument("--bm25-k", type=int, default=10, help="How many BM25 results to consider")
    parser.add_argument("--vector-k", type=int, default=10, help="How many vector results to consider")
    parser.add_argument("--strategy", choices=["fixed", "recursive", "semantic"], default="fixed")
    parser.add_argument("--size", type=int, default=60, help="Chunk size for the selected BM25 chunking strategy")
    parser.add_argument("--overlap", type=int, default=10, help="Word overlap for fixed-size chunks")
    parser.add_argument("--collection-name", default="fieldfix_corpus", help="Chroma collection name")
    parser.add_argument("--k", type=int, default=60, help="RRF k parameter")
    args = parser.parse_args()

    results = hybrid_search(
        query=args.query,
        bm25_k=args.bm25_k,
        vector_k=args.vector_k,
        strategy=args.strategy,
        size=args.size,
        overlap=args.overlap,
        collection_name=args.collection_name,
        k=args.k,
    )
    print_rrf_results(results, args.query)


if __name__ == "__main__":
    main()
