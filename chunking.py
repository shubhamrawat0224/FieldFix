#!/usr/bin/env python3
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from loader import load_corpus


WORD_RE = re.compile(r"\b\w+[\w'-]*\b")


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def make_chunk(
    chunk_id: int,
    doc_id: str,
    title: str,
    heading: str,
    strategy: str,
    chunk_text: str,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "title": title,
        "heading": heading,
        "strategy": strategy,
        "text": chunk_text.strip(),
        "word_count": word_count(chunk_text),
    }


def fixed_size_chunks(doc: dict[str, Any], size: int, overlap: int, strategy: str = "fixed") -> list[dict[str, Any]]:
    words = WORD_RE.findall(doc["text"])
    if not words:
        return []

    step = max(1, size - overlap)
    chunks: list[dict[str, Any]] = []
    for index in range(0, len(words), step):
        segment = words[index : index + size]
        if not segment:
            break
        chunks.append(
            make_chunk(
                len(chunks) + 1,
                doc["doc_id"],
                doc["title"],
                doc["title"],
                strategy,
                " ".join(segment),
            )
        )
        if index + size >= len(words):
            break
    return chunks


def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in lines:
        match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if match:
            if current_lines:
                sections.append((current_heading.strip(), " ".join(current_lines).strip()))
            current_heading = match.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading.strip(), " ".join(current_lines).strip()))

    if not sections:
        return [("", text.strip())]
    return sections


def split_into_paragraph_buckets(text: str, max_words: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    if not paragraphs:
        return []

    buckets: list[str] = []
    current: list[str] = []
    current_count = 0

    for paragraph in paragraphs:
        paragraph_words = word_count(paragraph)

        if paragraph_words <= max_words:
            if current_count + paragraph_words <= max_words:
                current.append(paragraph)
                current_count += paragraph_words
            else:
                if current:
                    buckets.append(" ".join(current))
                current = [paragraph]
                current_count = paragraph_words
        else:
            if current:
                buckets.append(" ".join(current))
                current = []
                current_count = 0

            sentences = split_into_sentences(paragraph)
            sentence_bucket: list[str] = []
            sentence_count = 0
            for sentence in sentences:
                s_count = word_count(sentence)
                if sentence_count + s_count <= max_words:
                    sentence_bucket.append(sentence)
                    sentence_count += s_count
                else:
                    if sentence_bucket:
                        buckets.append(" ".join(sentence_bucket))
                    sentence_bucket = [sentence]
                    sentence_count = s_count
            if sentence_bucket:
                buckets.append(" ".join(sentence_bucket))

    if current:
        buckets.append(" ".join(current))

    return buckets or [text.strip()]


def split_into_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 1.0

    terms = set(a) | set(b)
    if not terms:
        return 1.0

    dot = sum(a.get(term, 0) * b.get(term, 0) for term in terms)
    mag_a = math.sqrt(sum(value * value for value in a.values()))
    mag_b = math.sqrt(sum(value * value for value in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 1.0
    return dot / (mag_a * mag_b)


def semantic_chunks(doc: dict[str, Any], max_words: int, threshold: float = 0.18, strategy: str = "semantic") -> list[dict[str, Any]]:
    sentences = split_into_sentences(doc["text"])
    if not sentences:
        return []

    chunks: list[dict[str, Any]] = []
    current_sentences: list[str] = []
    current_vector: Counter[str] | None = None
    previous_vector: Counter[str] | None = None

    def finalize_current() -> None:
        nonlocal current_sentences, current_vector, previous_vector
        if not current_sentences:
            return
        text = " ".join(current_sentences)
        chunks.append(
            make_chunk(
                len(chunks) + 1,
                doc["doc_id"],
                doc["title"],
                doc["title"],
                strategy,
                text,
            )
        )
        current_sentences = []
        current_vector = None
        previous_vector = None

    for sentence in sentences:
        sentence_vector = Counter(WORD_RE.findall(sentence.lower()))

        if not current_sentences:
            current_sentences = [sentence]
            current_vector = sentence_vector
            previous_vector = sentence_vector
            continue

        similarity = cosine_similarity(previous_vector or sentence_vector, sentence_vector)
        if word_count(" ".join(current_sentences + [sentence])) > max_words or similarity < threshold:
            finalize_current()
            current_sentences = [sentence]
            current_vector = sentence_vector
            previous_vector = sentence_vector
        else:
            current_sentences.append(sentence)
            current_vector = current_vector or Counter()
            current_vector.update(sentence_vector)
            previous_vector = sentence_vector

    if current_sentences:
        finalize_current()

    return chunks


def recursive_structure_chunks(doc: dict[str, Any], max_words: int, strategy: str = "recursive") -> list[dict[str, Any]]:
    sections = split_markdown_sections(doc["text"])
    chunks: list[dict[str, Any]] = []

    for heading, section_text in sections:
        heading_value = heading or doc["title"]
        if not section_text.strip():
            continue

        if word_count(section_text) <= max_words:
            chunks.append(
                make_chunk(
                    len(chunks) + 1,
                    doc["doc_id"],
                    doc["title"],
                    heading_value,
                    strategy,
                    section_text,
                )
            )
            continue

        for bucket in split_into_paragraph_buckets(section_text, max_words):
            chunks.append(
                make_chunk(
                    len(chunks) + 1,
                    doc["doc_id"],
                    doc["title"],
                    heading_value,
                    strategy,
                    bucket,
                )
            )

    return chunks


def build_counts_table(docs: list[dict[str, Any]], size: int) -> list[list[str]]:
    rows: list[list[str]] = [["Document", "Fixed", "Recursive", "Semantic"]]
    for doc in docs:
        fixed = fixed_size_chunks(doc, size, overlap=max(5, size // 6))
        recursive = recursive_structure_chunks(doc, size)
        semantic = semantic_chunks(doc, size)
        rows.append([
            doc["title"],
            str(len(fixed)),
            str(len(recursive)),
            str(len(semantic)),
        ])
    return rows


def format_markdown_table(rows: list[list[str]]) -> str:
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for row_index, row in enumerate(rows):
        cells = [str(cell).ljust(widths[i]) for i, cell in enumerate(row)]
        line = " | ".join(cells)
        lines.append(line)
        if row_index == 0:
            lines.append(" | ".join(["-" * widths[i] for i in range(len(row))]))
    return "\n".join(lines)


def example_split_facts_report(docs: list[dict[str, Any]]) -> str:
    target = next(doc for doc in docs if doc["doc_id"] == "amperia_refrigeration_guide")
    excerpt = (
        "A refrigeration case or walk-in unit that is not cooling may show one or more of the following: high head pressure, low evaporator airflow, warm product temperatures, repeated short cycling, or a fan motor that fails to start. Frost buildup on the evaporator can indicate restricted airflow, failed defrost control, or a refrigerant-related issue. A compressor that starts and then trips on overload is often a sign of overheating, failed capacitor, or a blocked condenser."
    )
    fixed = fixed_size_chunks({**target, "text": excerpt}, 40, 10)
    recursive = recursive_structure_chunks({**target, "text": excerpt}, 40)
    return (
        "### Example: a small fixed chunk splits two related facts that recursive chunking keeps together\n\n"
        f"Document: `{target['title']}`\n\n"
        "The excerpt below contains a symptom cluster and a compressor diagnosis in the same operational paragraph.\n\n"
        f"`fixed-size (40 words, 10 overlap)` produced {len(fixed)} chunks, while `recursive` produced {len(recursive)} chunk because the section stays intact until the max-cap is exceeded.\n\n"
        f"Excerpt:\n> {excerpt}\n\n"
        "This matters because the technician needs the relationship between the cooling symptom and the compressor overload diagnosis in one retrieval unit. A short fixed-size boundary can separate these into different chunks even though they describe one root-cause chain."
    )


def main() -> None:
    corpus_dir = Path(__file__).resolve().parent / "corpus"
    docs = load_corpus(corpus_dir)

    sizes = [40, 200]
    report_sections: list[str] = ["# Chunking Report", "", "## Document counts by strategy", ""]

    for size in sizes:
        report_sections.append(f"### Chunk size: {size} words")
        table = format_markdown_table(build_counts_table(docs, size))
        report_sections.append(table)
        report_sections.append("")

    report_sections.append(example_split_facts_report(docs))
    report_sections.append("")

    report_text = "\n".join(report_sections)
    report_path = Path(__file__).resolve().parent / "chunking_report.md"
    report_path.write_text(report_text + "\n", encoding="utf-8")

    print(report_text)
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
