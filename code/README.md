# Support Triage Agent — Setup & Run

## Architecture

```
Ticket IN
   │
   ├─► [Stage 1] Hard escalation rules (rule-based, NO LLM)
   │       Fraud / security / legal / outage keywords
   │       → if triggered: status=escalated, skip LLM decision
   │
   ├─► [Stage 2] Injection detection
   │       Prompt injection patterns → escalated + invalid
   │
   ├─► [Stage 3] Company-aware RAG retrieval
   │       HuggingFace embeddings + FAISS
   │       Filters to company corpus when known, global fallback
   │
   └─► [Stage 4] Gemini LLM triage (temperature=0.0)
           Grounded strictly in retrieved corpus chunks
           Structured JSON output → validated → output.csv
```

**Modules:**

| File | Responsibility |
|---|---|
| `main.py` | Entry point, orchestrates full pipeline |
| `corpus_loader.py` | Loads + chunks `data/` into FAISS with company metadata |
| `retriever.py` | Company-aware similarity search |
| `escalation.py` | Hard rule-based escalation (no LLM) |
| `agent.py` | Gemini API call, prompt construction, JSON validation |

## Requirements

Python 3.9+

## Install dependencies

```bash
pip install google-generativeai pandas tqdm \
  langchain langchain-community langchain-text-splitters \
  langchain-huggingface sentence-transformers \
  faiss-cpu
```

## Set API key

Get a free Gemini key at: https://aistudio.google.com/app/apikey

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your_key_here"

# Mac/Linux
export GEMINI_API_KEY=your_key_here
```

## Run

From the **repo root** (not from inside `code/`):

```bash
python code/main.py
```

First run downloads the embedding model (~50MB). Subsequent runs are instant.

## Output

Written to `support_tickets/output.csv` with columns:

| Column | Values |
|---|---|
| `status` | `replied` or `escalated` |
| `product_area` | short corpus-grounded category (e.g. `screen`, `privacy`) |
| `response` | user-facing answer grounded in `data/` only |
| `justification` | internal routing reasoning |
| `request_type` | `product_issue`, `feature_request`, `bug`, or `invalid` |

## Design decisions

- **HuggingFace embeddings** — no API key needed, runs locally, no rate limits
- **Company-aware retrieval** — filters FAISS to the ticket's company corpus first, falls back to global search if insufficient results
- **Two-tier escalation** — hard keyword rules run before LLM; LLM cannot be prompted into skipping them
- **temperature=0.0** — fully deterministic output, required by evaluation criteria
- **Injection detection** — tickets with prompt injection patterns are flagged and escalated
- **`company=None`** — runs global search; LLM infers best domain from content
- **JSON validation** — every LLM output field is validated against allowed values; invalid responses fall back to safe escalation