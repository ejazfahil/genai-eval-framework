# Benchmarks — measured

Real MMLU results produced by this repo's harness (`run_eval.py`), committed under
`results/`. Every number comes from an actual model run; re-runs are free and
reproducible via the content-addressed cache.

## MMLU (`cais/mmlu`, test split) — local models

**Config:** 120 questions sampled with `seed=42` across 8 subjects (machine_learning,
computer_security, college_computer_science, high_school_mathematics, anatomy,
professional_medicine, high_school_macroeconomics, philosophy), `temperature=0`.

| Model | MMLU accuracy | n | Provider |
|-------|:-------------:|:-:|----------|
| **qwen3:8b** | **0.675** | 120 | Ollama (local, free) |
| llama3.2:3b | 0.525 | 120 | Ollama (local, free) |

Random baseline = 0.25. Full per-subject breakdown lives in
[`results/mmlu_leaderboard.json`](../results/mmlu_leaderboard.json), and charts in
[`results/plots/`](../results/plots/).

**Reproduce:**

```bash
pip install -r requirements.txt
python run_eval.py --models llama3.2:3b,qwen3:8b --n 120 --seed 42
```

## Notes

- These are deliberately **small-sample (N=120)** local runs that prove the harness
  end-to-end — not a definitive leaderboard. Widen with `--n` (up to the full ~14k
  MMLU test set).
- Hosted models run through the same harness:
  `python run_eval.py --provider anthropic --models claude-3-haiku-20240307 --n 120`
  (needs `ANTHROPIC_API_KEY`).
- **No external/published figures are reproduced here.** An earlier version of this
  file listed illustrative frontier-model numbers that were *not* produced by this
  code; they have been removed to avoid presenting unverified results.
