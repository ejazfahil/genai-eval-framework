"""Ollama local-model adapter — talks to ``POST /api/generate`` via stdlib urllib.

Deterministic by default (``temperature=0``) and honours ``max_tokens`` through
Ollama's ``num_predict`` so letter-answer tasks stay fast.
"""
from __future__ import annotations

import json
import urllib.request

from src.models.base_adapter import BaseModelAdapter


class OllamaAdapter(BaseModelAdapter):
    def __init__(self, model="llama3.2:3b", host="http://localhost:11434", timeout=120,
                 think=None, **kw):
        super().__init__(model, **kw)
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.think = think  # None = model default; False disables reasoning (e.g. Qwen3)

    def _call_api(self, prompt, **kw):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kw.get("temperature", 0.0),
                "num_predict": kw.get("max_tokens", 256),
            },
        }
        if self.think is not None:
            payload["think"] = self.think
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        return {
            "text": data.get("response", ""),
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
        }
