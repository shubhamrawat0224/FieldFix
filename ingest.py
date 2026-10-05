#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

import chromadb

from chunking import recursive_structure_chunks
from loader import load_corpus
from observability import estimate_cost, estimate_token_count, log_event, new_correlation_id, stage_timer


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


EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


def sha256_of_corpus(corpus_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(corpus_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\n")
            digest.update(path.read_bytes())
            digest.update(b"\n")
    return digest.hexdigest()


def embed_texts(texts: list[str]) -> list[list[float]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "openai/text-embedding-3-small")

    payload = {"model": embedding_model, "input": texts}
    req = request.Request(
        f"{base_url.rstrip('/')}/embeddings",
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
    embeddings = []
    for item in data.get("data", []):
        embeddings.append(item["embedding"])
    if len(embeddings) != len(texts):
        raise RuntimeError(f"Expected {len(texts)} embeddings, got {len(embeddings)}")
    return embeddings


def build_records(corpus_dir: Path, strategy: str, size: int) -> tuple[str, list[dict[str, object]]]:
    docs = load_corpus(corpus_dir)
    corpus_id = sha256_of_corpus(corpus_dir)
    records: list[dict[str, object]] = []

    for doc in docs:
        if strategy == "recursive":
            chunks = recursive_structure_chunks(doc, max_words=size, strategy="recursive")
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")

        for chunk in chunks:
            records.append(
                {
                    "id": f"{doc['doc_id']}-{chunk['chunk_id']}",
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": doc["doc_id"],
                    "corpus_id": corpus_id,
                    "title": doc["title"],
                    "content": chunk["text"],
                    "metadata": {
                        "strategy": strategy,
                        "heading": chunk["heading"],
                        "word_count": chunk["word_count"],
                    },
                }
            )

    return corpus_id, records


def write_corpus_manifest(corpus_dir: Path, strategy: str, collection_name: str, corpus_id: str, records: list[dict[str, object]]) -> dict[str, object]:
    root = Path(__file__).resolve().parent
    manifest_path = root / "corpus_versions.json"
    manifest: dict[str, object] = {"active_corpus_id": corpus_id, "history": []}

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {"active_corpus_id": corpus_id, "history": []}

    docs = sorted({str(record["doc_id"]) for record in records})
    entry = {
        "corpus_id": corpus_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "collection_name": collection_name,
        "doc_count": len(docs),
        "chunk_count": len(records),
        "document_ids": docs,
        "source_dir": str(corpus_dir),
        "rollback_ready": True,
    }

    history = list(manifest.get("history", []))
    existing_ids = {str(item.get("corpus_id")) for item in history}
    if corpus_id not in existing_ids:
        history.insert(0, entry)
    else:
        for idx, item in enumerate(history):
            if str(item.get("corpus_id")) == corpus_id:
                history[idx] = entry
                break
    manifest["history"] = history[:10]
    manifest["active_corpus_id"] = corpus_id
    manifest["rollback_target"] = corpus_id
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def ingest_records(records: list[dict[str, object]], collection_name: str = "fieldfix_corpus") -> None:
    correlation_id = new_correlation_id()
    with stage_timer("ingest_records", correlation_id=correlation_id, record_count=len(records), collection_name=collection_name):
        client = chromadb.PersistentClient(path=str(Path(__file__).resolve().parent / "vector_store"))
        existing = [c.name for c in client.list_collections()]
        if collection_name in existing:
            collection = client.get_collection(collection_name)
            collection.delete(ids=[record["id"] for record in records])
        else:
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        texts = [str(record["content"]) for record in records]
        embeddings = embed_texts(texts)

        collection.upsert(
            ids=[str(record["id"]) for record in records],
            embeddings=embeddings,
            metadatas=[
                {
                    "chunk_id": record["chunk_id"],
                    "doc_id": record["doc_id"],
                    "corpus_id": record["corpus_id"],
                    "title": record["title"],
                    "metadata": json.dumps(record["metadata"]),
                }
                for record in records
            ],
            documents=texts,
        )

        print(f"Collection: {collection_name}")
        print(f"Inserted: {len(records)} chunk records")

        sample = collection.get(limit=3, include=["metadatas", "documents", "embeddings"])
        print("Sample records:")
        for i, record_id in enumerate(sample["ids"][:3]):
            idx = sample["ids"].index(record_id)
            print({
                "id": record_id,
                "chunk_id": sample["metadatas"][idx].get("chunk_id"),
                "doc_id": sample["metadatas"][idx].get("doc_id"),
                "corpus_id": sample["metadatas"][idx].get("corpus_id"),
                "title": sample["metadatas"][idx].get("title"),
                "metadata": json.loads(sample["metadatas"][idx].get("metadata") or "{}"),
                "content_preview": (sample["documents"][idx][:120] + "...") if len(sample["documents"][idx]) > 120 else sample["documents"][idx],
            })

        aggregate_tokens = sum(estimate_token_count(str(record["content"])) for record in records)
        total_cost = estimate_cost("openai/text-embedding-3-small", aggregate_tokens, 0)
        log_event("ingest_embedding_cost", "ok", correlation_id=correlation_id, model="openai/text-embedding-3-small", input_tokens=aggregate_tokens, cost_usd=round(total_cost, 8))


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed and ingest chunked corpus into Chroma")
    parser.add_argument("--corpus-dir", default=str(Path(__file__).resolve().parent / "corpus"), help="Directory with markdown/txt documents")
    parser.add_argument("--strategy", default="recursive", choices=["recursive"], help="Chunking strategy to use")
    parser.add_argument("--size", type=int, default=200, help="Chunk size in words for the chosen strategy")
    parser.add_argument("--collection-name", default="fieldfix_corpus", help="Chroma collection name")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    corpus_id, records = build_records(corpus_dir, strategy=args.strategy, size=args.size)
    print(f"Corpus ID: {corpus_id}")
    ingest_records(records, collection_name=args.collection_name)
    manifest = write_corpus_manifest(corpus_dir, args.strategy, args.collection_name, corpus_id, records)
    print(f"Corpus manifest updated: {manifest['active_corpus_id']}")


if __name__ == "__main__":
    main()
