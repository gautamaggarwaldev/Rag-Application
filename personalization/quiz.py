import random
from typing import List, Dict

SAMPLE_QUESTIONS = {
    "intro": [
        {"q": "Python is statically typed. True or False?", "ans": "False"},
        {"q": "Name one common use of Python.", "ans": "web development"}
    ],
    "control_flow": [
        {"q": "Which keyword creates a loop over a sequence?", "ans": "for"},
        {"q": "Blocks in Python are defined by ______.", "ans": "indentation"}
    ],
    "functions": [
        {"q": "Which keyword returns a value from a function?", "ans": "return"},
        {"q": "What is the output of add(2,3) if add returns a+b?", "ans": "5"}
    ]
}

def generate_quiz(competency_id: str, n: int = 2) -> List[Dict]:
    qs = SAMPLE_QUESTIONS.get(competency_id, [])
    random.shuffle(qs)
    return qs[:n]

def grade_quiz(quiz: List[Dict], responses: List[str]) -> float:
    correct = 0
    for item, resp in zip(quiz, responses):
        gold = item["ans"].strip().lower()
        if gold in resp.strip().lower():
            correct += 1
    return correct / max(1, len(quiz))
