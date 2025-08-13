# rag/vectorstore.py
from typing import List, Dict, Any
import os, logging
import chromadb
from chromadb.config import Settings

# Global kill switches (belt-and-suspenders)
os.environ["CHROMA_TELEMETRY__ENABLED"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["POSTHOG_DISABLED"] = "true"
os.environ["PH_DISABLED"] = "true"
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

def _kill_chroma_telemetry():
    """Monkey-patch Chroma telemetry to a no-op across 0.5.x shapes."""
    try:
        from chromadb.telemetry import telemetry as _t
        def _noop(*args, **kwargs):
            return None
        for attr in ("capture", "flush", "identify", "alias", "group"):
            if hasattr(_t, attr):
                try:
                    setattr(_t, attr, _noop)
                except Exception:
                    pass
        if hasattr(_t, "_telemetry"):
            try:
                _t._telemetry = type("Noop", (), {"capture": _noop, "flush": _noop})()
            except Exception:
                pass
    except Exception:
        pass

def get_all(self):
    """Return all docs & metadatas currently in the collection (0.5.x-safe)."""
    try:
        return self.collection.get(include=["documents", "metadatas"])
    except TypeError:
        return self.collection.get()



_kill_chroma_telemetry()

class VectorStore:
    def __init__(self, persist_dir: str):
        # Prefer the 0.5.x PersistentClient if available
        client = None
        try:
            # 0.5.x
            client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        except Exception:
            # Fallback for some builds
            client = chromadb.Client(
                Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=persist_dir,
                    anonymized_telemetry=False,
                )
            )
        self.client = client
        _kill_chroma_telemetry()
        self.collection = self.client.get_or_create_collection("edu_rag")

    def add(self, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]], documents: List[str]):
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def query(self, query_embedding: List[float], top_k: int = 4):
        _kill_chroma_telemetry()
        return self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

    def reset(self):
        self.client.delete_collection("edu_rag")
        self.collection = self.client.get_or_create_collection("edu_rag")
    
    
