from dataclasses import dataclass
from typing import Dict

QUESTIONS = [
    ("I prefer diagrams, flowcharts, or maps when learning", "V"),
    ("I remember best when I hear explanations or discussions", "A"),
    ("I like reading/writing detailed notes and handouts", "R"),
    ("I learn by doing: hands-on tasks and experiments", "K"),
    ("Visual examples help me more than text-only descriptions", "V"),
    ("Podcasts or lectures are effective for me", "A"),
    ("I summarize concepts in my own written words", "R"),
    ("Practicing with exercises helps me internalize concepts", "K"),
]

@dataclass
class StyleScore:
    V: int = 0
    A: int = 0
    R: int = 0
    K: int = 0

def score(responses: Dict[int, int]) -> StyleScore:
    # responses: {question_index: 0/1}, where 1 = agree/true
    s = StyleScore()
    for i, val in responses.items():
        if val not in (0, 1): continue
        dim = QUESTIONS[i][1]
        if val == 1:
            setattr(s, dim, getattr(s, dim) + 1)
    return s

def primary_style(st: StyleScore) -> str:
    best = max(("V","A","R","K"), key=lambda k: getattr(st, k))
    mapping = {"V":"Visual","A":"Auditory","R":"Read/Write","K":"Kinesthetic"}
    return mapping[best]
