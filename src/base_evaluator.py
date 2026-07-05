"""Base evaluator.

An evaluator owns a dataset, an answer-parsing rule, and an aggregation, and runs
independently of which model adapter scores it. Concrete evaluators implement
``load_dataset``, ``evaluate_single``, and ``aggregate``.
"""
from __future__ import annotations


class BaseEvaluator:
    def __init__(self, model):
        self.model = model
        self.results = []

    def load_dataset(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def evaluate_single(self, item):  # pragma: no cover - overridden
        raise NotImplementedError

    def aggregate(self):  # pragma: no cover - overridden
        raise NotImplementedError
