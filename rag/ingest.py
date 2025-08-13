import os, uuid, time, json
from pathlib import Path
from typing import Dict, List, Tuple
from PyPDF2 import PdfReader

def read_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    txt = []
    for page in reader.pages:
        txt.append(page.extract_text() or "")
    return "\n".join(txt)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += (chunk_size - overlap)
    return chunks

def load_and_chunk(paths: List[Path], chunk_size: int, overlap: int) -> List[Tuple[str, Dict]]:
    results = []
    for p in paths:
        if p.suffix.lower() == ".pdf":
            raw = read_text_from_pdf(p)
        else:
            raw = p.read_text(encoding="utf-8", errors="ignore")
        for ch in chunk_text(raw, chunk_size, overlap):
            results.append((ch, {"source": str(p), "type": p.suffix.lower().lstrip("."), "ts": time.time()}))
    return results

def prepare_documents(content_dir: Path) -> List[Path]:
    paths = []
    for ext in ("*.txt", "*.md", "*.pdf"):
        paths += list(content_dir.rglob(ext))
    return paths
