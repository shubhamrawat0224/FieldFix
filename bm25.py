#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from chunking import fixed_size_chunks, load_corpus, recursive_structure_chunks, semantic_chunks

WORD_RE = re.compile(r"\b[\w'-]+\b")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def compute_idf(term: str, n_chunks: int, df: int) -> float:
    if df == 0:
        return 0.0
    return math.log(((n_chunks - df + 0.5) / (df + 0.5)) + 1.0)


def build_inverted_index(chunks: list[dict[str, Any]]) -> tuple[dict[str, dict[int, int]], dict[str, int]]:
    index: dict[str, dict[int, int]] = defaultdict(dict)
    df: dict[str, int] = defaultdict(int)

    for chunk_idx, chunk in enumerate(chunks):
        counts = Counter(tokenize(chunk["text"]))
        for term, tf in counts.items():
            index[term][chunk_idx] = tf
            df[term] += 1

    return index, df


def build_chunks(corpus_dir: Path, strategy: str = "fixed", size: int = 60, overlap: int = 10) -> list[dict[str, Any]]:
    docs = load_corpus(corpus_dir)
    chunks: list[dict[str, Any]] = []

    for doc in docs:
        if strategy == "fixed":
            generated = fixed_size_chunks(doc, size=size, overlap=overlap, strategy="fixed")
        elif strategy == "recursive":
            generated = recursive_structure_chunks(doc, max_words=size, strategy="recursive")
        elif strategy == "semantic":
            generated = semantic_chunks(doc, max_words=size, threshold=0.18, strategy="semantic")
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
        chunks.extend(generated)

    return chunks


def bm25_score(
    chunk_text: str,
    chunk_index: int,
    query_terms: list[str],
    index: dict[str, dict[int, int]],
    idf_map: dict[str, float],
    avgdl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not query_terms:
        return 0.0

    document_length = len(tokenize(chunk_text))
    if document_length == 0:
        return 0.0

    terms_in_chunk = Counter(tokenize(chunk_text))
    score = 0.0

    for term in set(query_terms):
        if term not in index or chunk_index not in index[term]:
            continue

        tf = index[term][chunk_index]
        idf = idf_map.get(term, 0.0)
        if idf == 0.0:
            continue

        tf_component = tf * (k1 + 1.0)
        length_component = tf + k1 * (1.0 - b + b * (document_length / avgdl if avgdl > 0 else 1.0))
        score += idf * (tf_component / length_component)

    return score


def rank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    k: int = 5,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[float, dict[str, Any]]]:
    if not chunks:
        return []

    query_terms = [term for term in tokenize(query) if term]
    if not query_terms:
        return []

    index, df = build_inverted_index(chunks)
    n_chunks = len(chunks)
    idf_map = {term: compute_idf(term, n_chunks, df[term]) for term in index}
    avgdl = sum(len(tokenize(chunk["text"])) for chunk in chunks) / n_chunks

    scored: list[tuple[float, dict[str, Any]]] = []
    for chunk_index, chunk in enumerate(chunks):
        score = bm25_score(
            chunk["text"],
            chunk_index,
            query_terms,
            index=index,
            idf_map=idf_map,
            avgdl=avgdl,
            k1=k1,
            b=b,
        )
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:k]


def print_ranked_results(results: list[tuple[float, dict[str, Any]]]) -> None:
    if not results:
        print("No matches found.")
        return

    print(f"{'Score':>10} | {'Doc':<30} | {'Chunk':<16} | {'Words':>5}")
    print("-" * 80)
    for score, chunk in results:
        print(f"{score:>10.4f} | {chunk['doc_id']:<30} | {str(chunk['chunk_id']):<16} | {chunk['word_count']:>5}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 retrieval over the project corpus")
    parser.add_argument("--query", default="safety access and customer refusal", help="Search query")
    parser.add_argument("--k", type=int, default=5, help="Number of top chunks to return")
    parser.add_argument("--strategy", choices=["fixed", "recursive", "semantic"], default="fixed")
    parser.add_argument("--size", type=int, default=60, help="Chunk size to use for fixed/recursive/semantic strategies")
    parser.add_argument("--overlap", type=int, default=10, help="Overlap for fixed-size chunks")
    parser.add_argument("--corpus-dir", default=str(Path(__file__).resolve().parent / "corpus"), help="Directory of .md/.txt documents")
    args = parser.parse_args()

    chunks = build_chunks(Path(args.corpus_dir), strategy=args.strategy, size=args.size, overlap=args.overlap)
    results = rank_chunks(args.query, chunks, k=args.k)
    print(f"Corpus chunks: {len(chunks)}")
    print(f"Query: {args.query}")
    print_ranked_results(results)


if __name__ == "__main__":
    main()
