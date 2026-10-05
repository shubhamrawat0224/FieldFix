# FieldFix Retrieval Notes

## BM25 toy worked example

Consider this tiny corpus with two chunks:

- Chunk A: "DC-150 warranty warranty"
- Chunk B: "standard warranty policy"

Let the query be: "warranty".

The term counts are:

- For Chunk A, `warranty` appears twice, so `tf = 2`
- For Chunk B, `warranty` appears once, so `tf = 1`

The document frequency is:

- `warranty` appears in both chunks, so `df = 2`
- `dc-150` appears only in Chunk A, so `df = 1`

Using the requested IDF formula:

$$
\mathrm{IDF}(t) = \log\left(\frac{N - df + 0.5}{df + 0.5} + 1\right)
$$

with `N = 2`:

- For `warranty`: 
  $$
  \log\left(\frac{2 - 2 + 0.5}{2 + 0.5} + 1\right)
  = \log\left(\frac{0.5}{2.5} + 1\right)
  = \log(1.2)
  \approx 0.182
  $$

- For `dc-150`:
  $$
  \log\left(\frac{2 - 1 + 0.5}{1 + 0.5} + 1\right)
  = \log\left(\frac{1.5}{1.5} + 1\right)
  = \log(2)
  \approx 0.693
  $$

The rarer term (`dc-150`) gets the higher IDF because it appears in fewer chunks. In a real query, that gives it more weighting than a common word like `warranty`.

This is the exact behaviour BM25 relies on: common terms are down-weighted, rare terms are up-weighted.

## Running BM25

```bash
cd /Users/shubhamrawat/Desktop/projects/FieldFix
python3 bm25.py --query "safety access and customer refusal" --k 5 --strategy fixed --size 60 --overlap 10
```

This prints the top-ranked chunks with BM25 scores.

## Embeddings + vector database

Chosen embedding model: `openai/text-embedding-3-small`
Dimension count: `1536`

The OpenRouter setup requires provider-prefixed model names, such as `openai/gpt-5-mini` or `openai/text-embedding-3-small`, and the base URL must be `https://openrouter.ai/api/v1`.

Vector database: Chroma (local persistent store) in `vector_store/` at the project root.

Each stored record includes:
- `id`
- `chunk_id`
- `doc_id`
- `corpus_id`
- `title`
- `content`
- `metadata`
- `vector`

The `corpus_id` is a SHA-256 fingerprint of the full corpus contents at ingestion time and is stored on every chunk as the watermark for that corpus version.

Run ingestion:

```bash
cd /Users/shubhamrawat/Desktop/projects/FieldFix
python3 ingest.py --strategy recursive --size 200
```

This upserts the recursive chunks into Chroma in one batch and prints a few sample stored records.

## Clean clone setup

From a fresh clone of the repo:

```bash
cd FieldFix
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
cp .env.example .env
# then edit .env and add your OpenRouter API key
python3 ingest.py --corpus-dir corpus --strategy recursive --size 200 --collection-name fieldfix_corpus
python3 answer.py --question "What should a technician do if the site is unsafe and the customer refuses access?"
```

Important:
- `.env` is required for the hosted LLM and embeddings to work.
- The project expects an OpenRouter-compatible base URL and a model such as `openai/gpt-5-mini`.
- Do not submit late work; late submissions follow the programme policy.
