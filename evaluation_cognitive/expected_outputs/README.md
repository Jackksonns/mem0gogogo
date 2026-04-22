# expected_outputs/

Placeholder directory. Each ablation run declared in `../configs/` will write
its per-turn outputs to a subdirectory here (for example
`expected_outputs/ablation_full/`).

**Nothing numeric is checked in yet.** Once `scripts/run_ablation.py` is
implemented and produces stable output, each released run will come with:

- `run_manifest.json` — config hash, code SHA, seed values, runtime.
- `per_turn_scores.csv` — one row per `(dialogue_id, query_id, turn_id)`
  with retention score, affective composite, rank of gold turn in retrieved
  top-k.
- `summary.json` — aggregated metrics matching the rows of the paper's
  experiments table.
- `outputs.sha256` — SHA-256 of every other file in the directory, so
  reimplementers can diff numerically.

Until then, this README is the only file under `expected_outputs/`.
