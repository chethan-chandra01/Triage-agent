"""
retriever.py
------------
Company-aware retrieval from the FAISS vectorstore.

Strategy:
- If company is HackerRank / Claude / Visa → filter chunks to that company first,
  fall back to global search if not enough results.
- If company is None → search all chunks, let similarity decide.
"""

from langchain_community.vectorstores import FAISS

TOP_K = 5  # number of chunks to retrieve per query


def retrieve(vectorstore: FAISS, issue: str, company: str) -> list:
    """
    Returns up to TOP_K document chunks most relevant to the issue.
    Uses company metadata to narrow the search when possible.
    """
    known_companies = {"HackerRank", "Claude", "Visa"}

    if company in known_companies:
        # Attempt filtered retrieval first
        try:
            results = vectorstore.similarity_search(
                query=f"{company} support: {issue}",
                k=TOP_K,
                filter={"company": company},
            )
            if len(results) >= 2:
                return results
        except Exception:
            pass  # FAISS filter not supported — fall through to global

    # Global retrieval (company=None or filter returned too few results)
    results = vectorstore.similarity_search(
        query=issue,
        k=TOP_K,
    )
    return results


def format_context(docs: list) -> str:
    """
    Formats retrieved chunks into a single context string for the LLM prompt.
    Each chunk is labeled with its source file for traceability.
    """
    if not docs:
        return "No relevant documentation found."

    sections = []
    for i, doc in enumerate(docs, 1):
        source  = doc.metadata.get("source", "unknown")
        company = doc.metadata.get("company", "unknown")
        content = doc.page_content.strip()
        sections.append(
            f"[Source {i} | {company} | {source}]\n{content}"
        )

    return "\n\n---\n\n".join(sections)