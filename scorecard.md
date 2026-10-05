# Retrieval evaluation scorecard

| Question | BM25@3 | BM25@5 | Vector@3 | Vector@5 | Hybrid@3 | Hybrid@5 | Rerank@3 | Rerank@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| What should the technician do if the site is unsafe and the customer refuses access? | 2/3 | 2/3 | 0/3 | 0/3 | 3/3 | 3/3 | 2/3 | 2/3 |
| When do we escalate a customer refusal or missing signature? | 3/3 | 3/3 | 2/3 | 2/3 | 3/3 | 3/3 | 1/3 | 2/3 |
| What is the rule for a customer who refuses to sign the final work after repair? | 1/3 | 2/3 | 2/3 | 2/3 | 3/3 | 3/3 | 2/3 | 3/3 |
| How should a technician proceed when access is blocked? | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 |
| Can the technician continue work if the site is unsafe? | 3/3 | 3/3 | 1/3 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| If a compressor fails and the customer will not provide a signature, what happens? | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 2/3 |
| Is a repair covered by warranty automatically? | 2/3 | 2/3 | 0/3 | 2/3 | 3/3 | 3/3 | 2/3 | 2/3 |
| What triggers an escalation for out-of-contract work? | 2/3 | 2/3 | 1/3 | 1/3 | 2/3 | 2/3 | 2/3 | 2/3 |
| Which parts can a technician install? | 2/3 | 2/3 | 1/3 | 1/3 | 2/3 | 2/3 | 1/3 | 1/3 |
| What should the technician do when a required part is not in the vehicle inventory? | 2/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 |
| What does the company require if the customer refuses work or access? | 3/3 | 3/3 | 2/3 | 2/3 | 3/3 | 3/3 | 1/3 | 3/3 |
| What should a technician do for a gas odor or smoke on site? | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TOTAL | 29/36 | 30/36 | 21/36 | 23/36 | 34/36 | 34/36 | 21/36 | 29/36 |

- bm25: 30/36 facts @ 5
- vector: 23/36 facts @ 5
- hybrid: 34/36 facts @ 5
- rerank: 29/36 facts @ 5

Best strategy at top-5: Hybrid (34/36 facts)

## Why a small eval set can mislead you
- A tiny eval set is extremely sensitive to wording, synonyms, and exact fact phrasing.
- One or two questions can overstate a strategy's quality when the corpus contains a few highly similar policies.
- A small set can hide edge cases like missing signature rules, access restrictions, or contract exceptions that only appear under different phrasings.
- The right approach is to treat the eval set as a quick sanity check, then add more realistic technician prompts and review retrieval failures before trusting a winner.
