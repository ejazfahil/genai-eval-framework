"""Base model adapter.

Every provider adapter implements ``_call_api(prompt, **kw) -> {"text": ...}``.
``complete()`` is the uniform entry point evaluators call; it transparently uses
an optional content-addressed :class:`~src.utils.cache.EvalCache` so repeated
prompts hit disk instead of the network/model.
"""
from __future__ import annotations


class BaseModelAdapter:
    def __init__(self, model, cache=None, **kwargs):
        self.model_name = model
        self._cache = cache
        self._kwargs = kwargs

    def complete(self, prompt, **kw):
        """Return ``{"text": ...}`` for ``prompt``, via cache when available."""
        if self._cache is not None:
            hit = self._cache.get(prompt, self.model_name)
            if hit is not None:
                return hit
        out = self._call_api(prompt, **kw)
        if self._cache is not None:
            self._cache.set(prompt, self.model_name, out)
        return out

    def _call_api(self, prompt, **kw):  # pragma: no cover - overridden by adapters
        raise NotImplementedError
