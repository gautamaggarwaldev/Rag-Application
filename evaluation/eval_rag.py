import time, ast, csv
from pathlib import Path
from typing import List, Dict
import numpy as np

def precision_recall_at_k(relevant: List[str], retrieved_ids: List[str], k: int = 4):
    retrieved_k = retrieved_ids[:k]
    rel_set = set(relevant)
    hit = sum(1 for r in retrieved_k if r in rel_set)
    precision = hit / max(1, len(retrieved_k))
    recall = hit / max(1, len(rel_set))
    return precision, recall

def run_eval(evaluator, eval_csv: Path, k: int = 4) -> Dict[str, float]:
    rows = list(csv.DictReader(open(eval_csv, "r", encoding="utf-8")))
    precs, recs, lats = [], [], []
    for row in rows:
        q = row["query"]
        relevant = ast.literal_eval(row["relevant_doc_ids"])
        t0 = time.time()
        ids = evaluator.retrieve_ids(q, k=k)
        lats.append(time.time() - t0)
        p, r = precision_recall_at_k(relevant, ids, k=k)
        precs.append(p); recs.append(r)
    return {
        "precision@k": float(np.mean(precs)),
        "recall@k": float(np.mean(recs)),
        "avg_latency_sec": float(np.mean(lats)),
    }
