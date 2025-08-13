import yaml
from typing import List, Dict, Any
from collections import defaultdict, deque

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def topo_sort(competencies: List[Dict[str, Any]]) -> List[str]:
    graph = defaultdict(list)
    indeg = defaultdict(int)
    for c in competencies:
        cid = c["id"]
        indeg.setdefault(cid, 0)
        for p in c.get("prerequisites", []):
            graph[p].append(cid)
            indeg[cid] += 1
    q = deque([c["id"] for c in competencies if indeg[c["id"]] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in graph[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order

def plan(competencies: List[Dict[str, Any]], level: str) -> List[Dict[str, Any]]:
    # Basic: lower level shifts easier items earlier; keep topo order
    order = topo_sort(competencies)
    level_bias = {"Beginner": -1, "Intermediate": 0, "Advanced": +1}
    lb = level_bias.get(level, 0)
    def diff_score(c):
        base = {"easy": 0, "medium": 1, "hard": 2}.get(c.get("difficulty","medium"),1)
        return base + (-0.2 if lb < 0 else (0.2 if lb > 0 else 0))
    ranked = sorted([c for c in competencies], key=lambda x: (order.index(x["id"]), diff_score(x)))
    return ranked
