"""MMLU evaluator — real multiple-choice accuracy on ``cais/mmlu``.

Loads the requested MMLU subject test splits, formats each item as a zero-shot
letter-answer prompt, fans the items across a thread pool (each call is I/O-bound
on the model), extracts the predicted ``A``/``B``/``C``/``D``, and reduces to
overall and per-subject accuracy. Thinking-model ``<think>...</think>`` prefixes
are stripped before parsing so reasoning models are scored fairly.
"""
from __future__ import annotations

import concurrent.futures
import random
import re

from src.base_evaluator import BaseEvaluator

# A diverse default slice of MMLU (valid ``cais/mmlu`` subject configs).
DEFAULT_SUBJECTS = [
    "machine_learning",
    "computer_security",
    "college_computer_science",
    "high_school_mathematics",
    "anatomy",
    "professional_medicine",
    "high_school_macroeconomics",
    "philosophy",
]

_LETTERS = "ABCD"
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_PROMPT = (
    "Answer the following multiple choice question. Respond with ONLY the letter "
    "of the correct option (A, B, C, or D).\n\n"
    "Question: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:"
)


class MMLUEvaluator(BaseEvaluator):
    def __init__(self, model, subjects=None, max_tokens=8, prompt_suffix=""):
        super().__init__(model)
        self.subjects = subjects or DEFAULT_SUBJECTS
        self.max_tokens = max_tokens
        self.prompt_suffix = prompt_suffix  # e.g. "/no_think" for Qwen3

    def load_dataset(self):
        from datasets import load_dataset

        items = []
        for subject in self.subjects:
            for row in load_dataset("cais/mmlu", subject, split="test"):
                items.append(
                    {
                        "subject": subject,
                        "question": row["question"],
                        "choices": row["choices"],
                        "answer": _LETTERS[row["answer"]],
                    }
                )
        return items

    def _format(self, item):
        c = item["choices"]
        prompt = _PROMPT.format(q=item["question"], a=c[0], b=c[1], c=c[2], d=c[3])
        return prompt + self.prompt_suffix

    @staticmethod
    def _parse_letter(text):
        text = _THINK.sub("", text).strip().upper()
        m = re.search(r"[ABCD]", text)
        return m.group(0) if m else ""

    def evaluate_single(self, item):
        resp = self.model.complete(self._format(item), max_tokens=self.max_tokens)
        pred = self._parse_letter(resp.get("text", ""))
        return {
            "subject": item["subject"],
            "pred": pred,
            "gold": item["answer"],
            "correct": pred == item["answer"],
        }

    def run(self, max_samples=None, seed=42, workers=4):
        data = self.load_dataset()
        if max_samples and max_samples < len(data):
            data = random.Random(seed).sample(data, max_samples)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            self.results = list(ex.map(self.evaluate_single, data))
        return self.aggregate()

    def aggregate(self):
        if not self.results:
            return {"accuracy": 0.0, "n": 0, "by_subject": {}}
        n = len(self.results)
        acc = sum(r["correct"] for r in self.results) / n
        by = {}
        for r in self.results:
            slot = by.setdefault(r["subject"], [0, 0])
            slot[0] += int(r["correct"])
            slot[1] += 1
        by_subject = {
            k: {"accuracy": round(v[0] / v[1], 4), "n": v[1]} for k, v in sorted(by.items())
        }
        return {"accuracy": round(acc, 4), "n": n, "by_subject": by_subject}
