"""
Retrieval-Augmented Generation (RAG) over the bank's policy knowledge base.

Loads markdown documents from the ``knowledge/`` directory, splits them into
sections, and retrieves the most relevant sections for a query using TF-IDF
cosine similarity. This is fully local — no external embedding service or
network call — so the policy-aware advisor works out of the box using the same
scikit-learn that powers the model.

(If you later want embedding-based retrieval, ChromaDB can be swapped in behind
``retrieve_policy`` without changing the agent or backend.)
"""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"

# Cached (vectorizer, document-term matrix, chunks) built lazily on first query.
_index: tuple | None = None


def _load_chunks() -> list[dict]:
    """Read every markdown doc and split it into heading-delimited sections."""
    chunks: list[dict] = []
    if not KNOWLEDGE_DIR.exists():
        return chunks
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        # Split before each markdown heading so each chunk is a coherent section.
        sections = re.split(r"\n(?=#{1,6}\s)", text)
        for sec in sections:
            sec = sec.strip()
            if len(sec) < 30:  # skip tiny fragments / front-matter
                continue
            chunks.append({"source": path.stem, "text": sec})
    return chunks


def _build_index():
    """Build and cache the TF-IDF index over policy chunks."""
    global _index
    if _index is not None:
        return _index
    chunks = _load_chunks()
    if not chunks:
        _index = (None, None, [])
        return _index
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Bigrams + sublinear tf so phrases like "credit utilization" or
    # "prime subprime" rank by meaning rather than single-word overlap.
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vec.fit_transform([c["text"] for c in chunks])
    _index = (vec, matrix, chunks)
    return _index


def retrieve_policy(query: str, n: int = 3) -> list[dict]:
    """Return the top-``n`` most relevant policy sections for ``query``.

    Each result is ``{"source": <doc name>, "text": <section>, "score": float}``.
    Returns an empty list if there is no knowledge base or nothing matches.
    """
    vec, matrix, chunks = _build_index()
    if not chunks or vec is None:
        return []
    from sklearn.metrics.pairwise import cosine_similarity

    q_vec = vec.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    order = sims.argsort()[::-1][:n]
    results = []
    for i in order:
        if sims[i] <= 0:
            continue
        results.append(
            {"source": chunks[i]["source"], "text": chunks[i]["text"], "score": float(sims[i])}
        )
    return results


def knowledge_status() -> str:
    """Human-readable status of the knowledge base (for diagnostics/UI)."""
    chunks = _load_chunks()
    if not chunks:
        return "No knowledge base found"
    sources = sorted({c["source"] for c in chunks})
    return f"{len(chunks)} chunks from {len(sources)} docs: {', '.join(sources)}"


def knowledge_info() -> dict:
    """Structured knowledge-base status for the API/UI."""
    chunks = _load_chunks()
    sources = sorted({c["source"] for c in chunks})
    return {
        "available": bool(chunks),
        "doc_count": len(sources),
        "chunk_count": len(chunks),
        "docs": [s.replace("_", " ") for s in sources],
    }
