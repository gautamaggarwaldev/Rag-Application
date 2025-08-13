from typing import List, Dict, Any
from .embeddings import Embedder, EmbedConfig
from .vectorstore import VectorStore

class Retriever:
    def __init__(self, vectorstore: VectorStore, embedder: Embedder, top_k: int = 4):
        self.vs = vectorstore
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str) -> Dict[str, Any]:
        q_emb = self.embedder.embed([query])[0]
        res = self.vs.query(query_embedding=q_emb, top_k=self.top_k)
        return res
