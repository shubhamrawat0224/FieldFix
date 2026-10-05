#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "FieldFix_Learning_Dashboard_Report.pdf"


def build_report() -> None:
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    title = "FieldFix Learning Dashboard Report"
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 20))

    body = [
        "Project: FieldFix",
        "Repository: https://github.com/shubhamrawat0224/FieldFix.git",
        "",
        "Summary:",
        "This project builds a retrieval-augmented field support assistant using a hosted OpenAI/OpenRouter-compatible LLM, local Chroma vector storage, a BM25 keyword retrieval layer, and grounded answer generation from retrieved evidence.",
        "",
        "Key milestones completed:",
        "- Switched from local Ollama usage to a hosted OpenAI-compatible API flow.",
        "- Added .env-based configuration for API credentials and model selection.",
        "- Loaded and indexed the six-document Amperia corpus.",
        "- Implemented fixed-size, recursive, and semantic chunking strategies.",
        "- Built manual BM25 ranking and vector-based retrieval.",
        "- Combined retrieval via hybrid search and reranking.",
        "- Created a grounded answer pipeline that cites retrieved evidence.",
        "- Validated a PDF-based knowledge pipeline using a PDF-only fact answer.",
        "",
        "Validation note:",
        "The repository was validated with the current working environment and ingestion pipeline from the project root. The PDF-only fact retrieval was confirmed with a permit number answer grounded in extracted PDF evidence.",
        "",
        "Submission note:",
        "This report should be submitted before the due date. Late submissions follow the programme policy.",
    ]

    for text in body:
        if not text:
            story.append(Spacer(1, 12))
            continue
        story.append(Paragraph(text, styles["BodyText"]))

    doc.build(story)
    print(f"Created report PDF: {OUTPUT}")


if __name__ == "__main__":
    build_report()
