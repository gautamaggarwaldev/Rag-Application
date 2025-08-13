# rag/generator.py  (edit only the changed parts)

import os
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

SYSTEM_PROMPT = "You are a helpful tutor. Answer using the provided context. Cite the source filenames when helpful."

def format_context(docs: List[str], metas: List[Dict]) -> str:
    ...

def _normalize_gemini_model(name: str) -> str:
    if name.startswith("models/") or name.startswith("tunedModels/"):
        return name
    return f"models/{name}"

class Generator:
    def __init__(self, provider: str = "flan-t5-small"):
        self.provider = provider.lower()
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.openai_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.genai = genai
            raw = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
            self.gemini_model = _normalize_gemini_model(raw)
            self.model = genai.GenerativeModel(self.gemini_model)
        else:
            from transformers import pipeline
            model_name = provider  # e.g., flan-t5-small (HF only if you choose it)
            self.pipe = pipeline("text2text-generation", model=model_name)

    def generate(self, question: str, docs: List[str], metas: List[Dict], max_tokens: int = 256) -> str:
        ...
        if self.provider == "openai":
            ...
        elif self.provider == "gemini":
            resp = self.model.generate_content(prompt)
            try:
                return resp.text
            except Exception:
                return str(resp)
        else:
            ...
