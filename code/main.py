"""
main.py
-------
Entry point for the HackerRank Orchestrate support triage agent.

Pipeline per ticket:
  1. Hard escalation check (rule-based, no LLM)
  2. Injection / malicious content check
  3. RAG retrieval (company-aware)
  4. LLM triage via Gemini (grounded in corpus only)
  5. Schema validation
  6. Write to output.csv

Run:
    python code/main.py

Requirements:
    NVIDIA Llama-3.3-70B  environment variable must be set.
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Allow imports from code/ when running from repo root
sys.path.insert(0, str(Path(__file__).parent))
from agent import init_nvidia as init_gemini, triage
from corpus_loader import load_corpus
from escalation    import is_injection_attempt, should_escalate
from retriever     import format_context, retrieve

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT    = Path(__file__).parent.parent
TICKETS_PATH = REPO_ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_PATH  = REPO_ROOT / "support_tickets" / "output.csv"

# ── Output column order (must match problem_statement.md) ─────────────────────

OUTPUT_COLUMNS = ["status", "product_area", "response", "justification", "request_type"]


# ── Per-ticket processing ─────────────────────────────────────────────────────

def process_ticket(row, vectorstore, gemini_model) -> dict:
    """
    Full pipeline for a single support ticket row.
    Returns a dict with all 5 required output fields.
    """
    issue   = str(row.get("Issue",   "") or "").strip()
    subject = str(row.get("Subject", "") or "").strip()
    company = str(row.get("Company", "") or "").strip()

    # Normalize company
    if company.lower() in ("none", "nan", ""):
        company = "None"

    # ── Guard: empty ticket ───────────────────────────────────────────────────
    if not issue:
        return {
            "status":        "escalated",
            "product_area":  "general_support",
            "response":      "Escalate to a human",
            "justification": "Ticket body is empty — cannot process.",
            "request_type":  "invalid",
        }

    # ── Guard: prompt injection attempt ──────────────────────────────────────
    if is_injection_attempt(issue):
        print(f"  [SECURITY] Injection attempt detected — escalating.")
        return {
            "status":        "escalated",
            "product_area":  "security",
            "response":      "Escalate to a human",
            "justification": "Ticket contains prompt injection patterns — escalated for safety.",
            "request_type":  "invalid",
        }

    # ── Stage 1: Hard rule-based escalation (no LLM) ─────────────────────────
    escalate, reason = should_escalate(issue)
    if escalate:
        print(f"  [RULE] Hard escalation: {reason}")
        # Still retrieve context so product_area can be inferred accurately
        docs    = retrieve(vectorstore, issue, company)
        context = format_context(docs)
        # Use LLM only to classify product_area and request_type, not to decide status
        result = triage(gemini_model, issue, subject, company, context)
        result["status"]        = "escalated"
        result["response"]      = "Escalate to a human"
        result["justification"] = reason
        return result

    # ── Stage 2: RAG retrieval ────────────────────────────────────────────────
    docs    = retrieve(vectorstore, issue, company)
    context = format_context(docs)

    # ── Stage 3: LLM triage ───────────────────────────────────────────────────
    result = triage(gemini_model, issue, subject, company, context)
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  HackerRank Orchestrate — Support Triage Agent")
    print("=" * 60)

    # Validate API key early
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("\n[ERROR] GEMINI_API_KEY or GOOGLE_API_KEY not set.")
        print("  Get a free key: https://aistudio.google.com/app/apikey")
        print("  Then run: set GEMINI_API_KEY=your_key_here  (Windows)")
        print("            export GEMINI_API_KEY=your_key_here  (Mac/Linux)")
        sys.exit(1)

    # Load tickets
    print(f"\n[1/4] Loading tickets from {TICKETS_PATH.name}...")
    if not TICKETS_PATH.exists():
        print(f"[ERROR] File not found: {TICKETS_PATH}")
        sys.exit(1)
    tickets = pd.read_csv(TICKETS_PATH)
    print(f"      Found {len(tickets)} tickets")

    # Build corpus vectorstore
    print("\n[2/4] Building corpus index...")
    vectorstore = load_corpus()

    # Init Gemini
    print("[3/4] Initialising Gemini model...")
    gemini_model = init_gemini()
    print("      Gemini ready (gemini-2.0-flash, temperature=0.0)\n")

    # Process tickets
    print(f"[4/4] Processing tickets...\n")
    results = []

    for idx, row in tqdm(tickets.iterrows(), total=len(tickets), desc="Triaging"):
        issue = str(row.get("issue", "") or "")[:80].replace("\n", " ")
        print(f"\n  Ticket {idx + 1}: {issue}...")

        result = process_ticket(row, vectorstore, gemini_model)
        results.append(result)

        print(f"  -> status={result['status']} | type={result['request_type']} | area={result['product_area']}")
        time.sleep(4.5)
    # Write output
    output_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Done! Results written to {OUTPUT_PATH}")
    print(f"  Replied:   {(output_df['status'] == 'replied').sum()}")
    print(f"  Escalated: {(output_df['status'] == 'escalated').sum()}")
    replied_types = output_df[output_df['status'] == 'replied']['request_type'].value_counts().to_dict()
    print(f"  Breakdown: {replied_types}")
    print("=" * 60)


if __name__ == "__main__":
    main()
