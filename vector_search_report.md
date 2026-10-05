# Vector Search vs BM25 comparison

## Query A: semantic retrieval wins

Query: "how long is the power module covered if it dies?"

Why vector search should win:
- This question is conceptually about a warranty/coverage policy, not a literal keyword match.
- The corpus contains policy language about warranty, contract coverage, and service windows rather than the exact phrase "power module" or "how long".
- Semantic retrieval can match on the idea of warranty duration and replacement coverage even when the terms differ.

Expected result:
- Vector search should surface the warranty / customer escalation / policy chunks more strongly than BM25.
- BM25 may rank generic policy text highly but may miss the conversational phrasing.

## Query B: BM25 wins

Query: "E-217"

Why BM25 should win:
- This is an exact code-like token; lexical retrieval is strongest here.
- A semantic model may match related concepts but will usually not be as precise for an exact diagnostic code or equipment tag.
- BM25 is good at exact-token retrieval and code matching.

Expected result:
- BM25 should rank the exact code-bearing chunk significantly above semantically similar text.

## Side-by-side practical interpretation

- Semantic search is better for natural-language questions that paraphrase procedure or policy.
- BM25 is better for exact codes, IDs, version strings, model numbers, and literal tokens such as "E-217" or "fw 4.3".

This is why a hybrid system is often best in practice: vector search for intent and BM25 for exact matches.
