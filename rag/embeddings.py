# rag/embeddings.py  (replace the existing file’s content with this snippet’s changed parts only)

import os
from typing import List
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class EmbedConfig:
    provider: str  # "openai" | "sentence-transformers" | "gemini"
    model: str     # e.g., "models/text-embedding-004"

def _normalize_gemini_model(name: str) -> str:
    if name.startswith("models/") or name.startswith("tunedModels/"):
        return name
    return f"models/{name}"

class Embedder:
    def __init__(self, config: EmbedConfig):
        self.config = config
        prov = (config.provider or "").lower()
        if prov == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif prov == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.genai = genai
            # store normalized model id
            self.gemini_embed_model = _normalize_gemini_model(config.model)
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(config.model)

    def embed(self, texts: List[str]) -> List[List[float]]:
        prov = (self.config.provider or "").lower()
        if prov == "openai":
            resp = self.client.embeddings.create(model=self.config.model, input=texts)
            return [d.embedding for d in resp.data]
        elif prov == "gemini":
            out = []
            for t in texts:
                r = self.genai.embed_content(model=self.gemini_embed_model, content=t)
                # google-generativeai returns dict-like or object depending on version
                out.append(r["embedding"] if isinstance(r, dict) else r.embedding)
            return out
        else:
            return self.model.encode(texts, convert_to_numpy=False).tolist()
