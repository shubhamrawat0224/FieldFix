#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib import error, request

import chromadb


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

EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "openai/text-embedding-3-small")


def sha256_of_corpus(corpus_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(corpus_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\n")
            digest.update(path.read_bytes())
            digest.update(b"\n")
    return digest.hexdigest()


def embed_text(text: str) -> list[float]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    payload = {"model": EMBEDDING_MODEL, "input": text}
    req = request.Request(
        f"{os.environ.get('OPENAI_BASE_URL', 'https://openrouter.ai/api/v1').rstrip('/')}/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=60) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Embedding request failed: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Embedding connection failed: {exc.reason}") from exc

    data = json.loads(body)
    return data["data"][0]["embedding"]


def get_collection(collection_name: str = "fieldfix_corpus"):
    client = chromadb.PersistentClient(path=str(Path(__file__).resolve().parent / "vector_store"))
    return client.get_collection(collection_name)


def vector_search(query: str, k: int = 5, collection_name: str = "fieldfix_corpus") -> list[dict[str, object]]:
    corpus_id = sha256_of_corpus(Path(__file__).resolve().parent / "corpus")
    collection = get_collection(collection_name)
    query_embedding = embed_text(query)

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"corpus_id": corpus_id},
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict[str, object]] = []
    for idx, doc in enumerate(result.get("documents", [[]])[0]):
        metadata = result.get("metadatas", [[{}]])[0][idx]
        distance = result.get("distances", [[0.0]])[0][idx]
        hits.append(
            {
                "rank": idx + 1,
                "score": round(max(0.0, 1.0 - float(distance)), 6),
                "doc_id": metadata.get("doc_id"),
                "title": metadata.get("title"),
                "chunk_id": metadata.get("chunk_id"),
                "preview": (doc[:180] + "...") if len(doc) > 180 else doc,
            }
        )
    return hits


def print_hits(hits: list[dict[str, object]], query: str) -> None:
    print(f"Query: {query}")
    print(f"{'Rank':>4} | {'Score':>8} | {'Doc ID':<30} | {'Chunk':>6} | {'Preview'}")
    print("-" * 120)
    for hit in hits:
        print(f"{hit['rank']:>4} | {hit['score']:>8.4f} | {str(hit['doc_id']):<30} | {str(hit['chunk_id']):>6} | {hit['preview']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic retrieval over the Chroma corpus")
    parser.add_argument("--query", default="How should a technician handle an unsafe refrigeration site?", help="Query to embed")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest chunks to return")
    parser.add_argument("--collection-name", default="fieldfix_corpus", help="Chroma collection name")
    args = parser.parse_args()

    hits = vector_search(args.query, k=args.k, collection_name=args.collection_name)
    print_hits(hits, args.query)


if __name__ == "__main__":
    main()
