#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_corpus(corpus_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(corpus_dir)
    docs: list[dict[str, Any]] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".txt"}:
            continue

        text = path.read_text(encoding="utf-8")
        doc_id = path.stem
        title = path.stem.replace("_", " ").title()
        docs.append({"doc_id": doc_id, "title": title, "text": text})
        word_count = len(text.split())
        print(f"Loaded: {doc_id} | title={title} | words={word_count}")

    print(f"\nDocument count: {len(docs)}")
    return docs


if __name__ == "__main__":
    corpus_dir = Path(__file__).resolve().parent / "corpus"
    print(f"Corpus directory: {corpus_dir}")
    load_corpus(corpus_dir)
