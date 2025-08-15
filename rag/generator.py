import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful tutor. Answer using ONLY the provided context. "
    "If the answer isn't in the context, say you don't have enough information. "
    "When helpful, mention the source filenames briefly."
)

def format_context(docs: List[str], metas: List[Dict]) -> str:
    lines = []
    for d, m in zip(docs, metas):
        src = m.get("source", "unknown")
        lines.append(f"[Source: {src}]\n{d}")
    return "\n\n".join(lines)

def _normalize_gemini_model(name: str) -> str:
    return name if name.startswith(("models/", "tunedModels/")) else f"models/{name}"

class Generator:
    def __init__(self, provider: str = "gemini"):
        self.provider = (provider or "gemini").lower()

        if self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            raw = os.getenv("GEMINI_CHAT_MODEL", "models/gemini-1.5-flash")
            model_name = _normalize_gemini_model(raw)
            # Attach the system prompt as system_instruction so user prompts stay clean
            self.model = genai.GenerativeModel(model_name, system_instruction=SYSTEM_PROMPT)

        elif self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.openai_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

        else:
            # Local HF fallback ONLY if you explicitly select it in the UI
            from transformers import pipeline
            self.pipe = pipeline("text2text-generation", model=provider)

    def generate(self, question: str, docs: List[str], metas: List[Dict], max_tokens: int = 256) -> str:
        ctx = format_context(docs, metas)

        user_prompt = (
            "Use the following context to answer the question. "
            "Be concise (5–8 sentences) and include short bullet points when useful.\n\n"
            f"Context:\n{ctx}\n\nQuestion: {question}\n"
        )

        if self.provider == "gemini":
            try:
                resp = self.model.generate_content(
                    user_prompt,
                    generation_config={"max_output_tokens": max_tokens, "temperature": 0.2}
                )
                # Prefer .text; fall back to stitching parts
                if hasattr(resp, "text") and resp.text:
                    return resp.text
                cand0 = (getattr(resp, "candidates", None) or [None])[0]
                content = getattr(cand0, "content", None)
                parts = getattr(content, "parts", None)
                if parts:
                    return "".join(getattr(p, "text", "") for p in parts if hasattr(p, "text"))
                return str(resp)
            except Exception as e:
                return f"(Generation error: {e})"

        elif self.provider == "openai":
            try:
                resp = self.client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content
            except Exception as e:
                return f"(OpenAI error: {e})"

        else:
            # HF pipeline
            try:
                out = self.pipe(user_prompt, max_new_tokens=max_tokens, do_sample=False)[0]["generated_text"]
                return out
            except Exception as e:
                return f"(HF generation error: {e})"
