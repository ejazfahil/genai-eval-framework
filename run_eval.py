#!/usr/bin/env python3
"""Run the MMLU evaluator over one or more models and write a leaderboard.

Every number comes from a real model call — the content-addressed cache makes
re-runs free and reproducible. Local models run free via Ollama; hosted models
run through the Anthropic adapter with a key.

Examples
--------
    # Free & local (Ollama):
    python run_eval.py --models llama3.2:3b,qwen3:8b --n 120 --seed 42

    # Hosted (needs ANTHROPIC_API_KEY):
    python run_eval.py --provider anthropic --models claude-3-haiku-20240307 --n 120
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.evaluators.mmlu import DEFAULT_SUBJECTS, MMLUEvaluator  # noqa: E402
from src.models.anthropic_adapter import AnthropicAdapter  # noqa: E402
from src.models.ollama_adapter import OllamaAdapter  # noqa: E402
from src.utils.cache import EvalCache  # noqa: E402


def build_adapter(model: str, provider: str, cache: EvalCache):
    if provider == "ollama":
        # Disable the reasoning phase for Qwen3-style models so we get a fast letter.
        think = False if "qwen3" in model.lower() else None
        return OllamaAdapter(model=model, cache=cache, think=think)
    if provider == "anthropic":
        return AnthropicAdapter(model=model, cache=cache)
    raise SystemExit(f"unknown provider '{provider}' (ollama | anthropic)")


def main():
    ap = argparse.ArgumentParser(description="Run MMLU over one or more models.")
    ap.add_argument("--models", required=True, help="comma-separated model ids")
    ap.add_argument("--provider", default="ollama", choices=["ollama", "anthropic"])
    ap.add_argument("--n", type=int, default=120, help="sampled questions per model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = EvalCache(str(out / ".eval_cache"))

    board = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"=== {model} ({args.provider}) ===", flush=True)
        adapter = build_adapter(model, args.provider, cache)
        ev = MMLUEvaluator(adapter, max_tokens=8)
        t0 = time.time()
        res = ev.run(max_samples=args.n, seed=args.seed)
        res.update({"model": model, "provider": args.provider, "seconds": round(time.time() - t0, 1)})
        board.append(res)
        print(f"  accuracy={res['accuracy']:.3f}  n={res['n']}  ({res['seconds']}s)", flush=True)

    board.sort(key=lambda r: r["accuracy"], reverse=True)
    payload = {
        "meta": {
            "benchmark": "MMLU (cais/mmlu, test split)",
            "subjects": DEFAULT_SUBJECTS,
            "n_per_model": args.n,
            "seed": args.seed,
            "temperature": 0.0,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "leaderboard": board,
    }
    (out / "mmlu_leaderboard.json").write_text(json.dumps(payload, indent=2))

    print("\n=== MMLU LEADERBOARD ===")
    print(f"{'model':28s} {'acc':>6s} {'n':>5s} {'sec':>7s}")
    for r in board:
        print(f"{r['model']:28s} {r['accuracy']:6.3f} {r['n']:5d} {r['seconds']:7.0f}")
    print(f"\nwrote {out / 'mmlu_leaderboard.json'}")


if __name__ == "__main__":
    main()
