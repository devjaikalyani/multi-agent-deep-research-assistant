"""
Vector Memory (ChromaDB)
- Stores full research sessions
- Sidebar buttons load past sessions directly
"""

import os
import hashlib
from datetime import datetime

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "../data/chroma_db")


def _get_collection():
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        ef = embedding_functions.DefaultEmbeddingFunction()
        return client.get_or_create_collection(
            name="research_memory",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception:
        return None


def save_research(query: str, report: str, metadata: dict = None) -> bool:
    collection = _get_collection()
    if not collection:
        return False
    try:
        doc_id = hashlib.md5(query.encode()).hexdigest()[:12]
        meta = {
            "query": query[:200],
            "timestamp": datetime.now().isoformat(),
            "report_length": len(report),
            "doc_id": doc_id,
            **(metadata or {})
        }
        # Store full report in document field (ChromaDB supports large text)
        document = f"FULL_REPORT|||{query}|||{report}"
        collection.upsert(ids=[doc_id], documents=[document], metadatas=[meta])
        return True
    except Exception as e:
        print(f"Memory save error: {e}")
        return False


def get_session_report(query: str) -> str | None:
    """Retrieve the full saved report for an exact query."""
    collection = _get_collection()
    if not collection:
        return None
    try:
        doc_id = hashlib.md5(query.encode()).hexdigest()[:12]
        result = collection.get(ids=[doc_id], include=["documents"])
        if result and result["documents"]:
            doc = result["documents"][0]
            if "FULL_REPORT|||" in doc:
                parts = doc.split("|||", 2)
                return parts[2] if len(parts) == 3 else None
        return None
    except Exception:
        return None


def retrieve_similar(query: str, n_results: int = 3) -> list[dict]:
    collection = _get_collection()
    if not collection:
        return []
    try:
        count = collection.count()
        if count == 0:
            return []
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, count)
        )
        items = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            items.append({
                "query": meta.get("query", "Unknown"),
                "timestamp": meta.get("timestamp", ""),
                "preview": doc[:300],
                "distance": results["distances"][0][i] if "distances" in results else 1.0
            })
        return items
    except Exception as e:
        print(f"Memory retrieve error: {e}")
        return []


def get_all_sessions() -> list[dict]:
    collection = _get_collection()
    if not collection:
        return []
    try:
        count = collection.count()
        if count == 0:
            return []
        results = collection.get(include=["metadatas"])
        sessions = []
        for meta in results["metadatas"]:
            sessions.append({
                "query": meta.get("query", "Unknown"),
                "timestamp": meta.get("timestamp", ""),
                "report_length": meta.get("report_length", 0),
                "doc_id": meta.get("doc_id", "")
            })
        sessions.sort(key=lambda x: x["timestamp"], reverse=True)
        return sessions
    except Exception:
        return []


def memory_available() -> bool:
    return _get_collection() is not None