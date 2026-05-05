"""
corpus_loader.py
----------------
Loads all .md files from data/ into a FAISS vector store.
Each chunk is tagged with its company (hackerrank/claude/visa)
so retriever.py can filter by company when the ticket specifies one.
"""

import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CORPUS_DIR  = Path(__file__).parent.parent / "data"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE  = 500
CHUNK_OVERLAP = 100

# Maps subfolder name → canonical company label used in tickets
COMPANY_FOLDER_MAP = {
    "hackerrank": "HackerRank",
    "claude":     "Claude",
    "visa":       "Visa",
}


def load_corpus() -> FAISS:
    """
    Walk data/hackerrank/, data/claude/, data/visa/ and load every .md file.
    Each chunk gets metadata: {source, company, filename}.
    Returns a FAISS vectorstore ready for similarity search.
    """
    print("[corpus_loader] Scanning corpus...")

    all_docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    for folder, company_label in COMPANY_FOLDER_MAP.items():
        folder_path = CORPUS_DIR / folder
        if not folder_path.exists():
            print(f"  [WARN] Folder not found: {folder_path}")
            continue

        md_files = list(folder_path.rglob("*.md"))
        print(f"  {company_label}: {len(md_files)} files")

        for md_file in md_files:
            try:
                loader = TextLoader(str(md_file), encoding="utf-8")
                raw_docs = loader.load()
                chunks = splitter.split_documents(raw_docs)

                # Tag every chunk with company so we can filter later
                for chunk in chunks:
                    chunk.metadata["company"]  = company_label
                    chunk.metadata["filename"] = md_file.name
                    chunk.metadata["source"]   = str(
                        md_file.relative_to(CORPUS_DIR)
                    )

                all_docs.extend(chunks)
            except Exception as e:
                print(f"  [WARN] Could not load {md_file.name}: {e}")

    print(f"[corpus_loader] Total chunks: {len(all_docs)}")
    print("[corpus_loader] Building FAISS index (downloads ~50MB on first run)...")

    embeddings  = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(all_docs, embeddings)

    print("[corpus_loader] Vector index ready.\n")
    return vectorstore