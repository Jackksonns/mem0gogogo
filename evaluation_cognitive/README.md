# evaluation_cognitive — Evaluation Harness for the Cognitive-Memory Paper

This directory holds the evaluation harness that belongs to **this** paper:

> *Emotion-Weighted Forgetting and Sleep Consolidation for Adaptive LLM Memory*

It is **deliberately separated** from `../evaluation_mem0_original/`, which is
the unmodified upstream Mem0 benchmark harness. The split was introduced in
Stage 6 of the repository's public remediation plan to answer a concrete piece
of reviewer feedback: *"it is not clear which evaluation code the paper's
numbers come from."*

---

## Status

**This is a skeleton, not a complete benchmark release.** We are publishing the
directory layout, configuration schema, adapter interfaces, and reproducibility
checklist first — and filling in the runnable pieces in follow-up PRs — rather
than shipping an opaque "results already included" artifact.

Every file in this directory is in exactly one of the following states:

| State                    | Meaning                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| **complete**             | Runs as-is (may still require an API key, documented in the script).    |
| **interface-only**       | Exposes the target public API; all methods raise `NotImplementedError`. |
| **config-only**          | YAML / manifest file; no executable code.                               |
| **placeholder**          | `.gitkeep` / README stub describing what will live there.               |

Each subdirectory's own `README.md` / module docstring states which of the four
it is.

---

## Layout

```
evaluation_cognitive/
├── README.md                 ← this file
├── configs/                  ← config-only: ablation + run YAMLs
│   ├── ablation_full.yaml
│   ├── ablation_no_emotion.yaml
│   ├── ablation_no_consolidation.yaml
│   ├── ablation_no_meta_learner.yaml
│   └── cognitivebench_seeds.yaml
├── generators/               ← interface-only: CognitiveBench generator
│   ├── __init__.py
│   └── cognitivebench.py
├── adapters/                 ← interface-only: dataset adapters
│   ├── __init__.py
│   └── locomo.py
├── scripts/                  ← interface-only: ablation + table scripts
│   ├── __init__.py
│   ├── run_ablation.py
│   └── make_tables.py
└── expected_outputs/         ← placeholder: target output locations
    └── README.md
```

---

## Reproducibility Checklist (honest status)

The paper's experiments section references these components. The checklist
below is what still has to happen before anyone — including us — can
end-to-end reproduce any number in the paper from a clean checkout.

- [ ] **CognitiveBench generator (`generators/cognitivebench.py`)** —
      currently raises `NotImplementedError`. Needs: dialogue simulator with
      the seed manifest in `configs/cognitivebench_seeds.yaml`, emotional-event
      injection, distractor / noise injection, difficulty stratification,
      query taxonomy labelling, leak-check against seed manifest.
- [ ] **LoCoMo adapter (`adapters/locomo.py`)** — currently raises
      `NotImplementedError`. Needs: load `locomo10.json` from
      `evaluation_mem0_original/dataset/`, convert each dialogue into the
      `(session_transcript, query, gold_memory_ids)` tuple that
      `scripts/run_ablation.py` consumes, and document the exact mapping from
      LoCoMo's answer attribution to our *retention-conditioned* retrieval
      metric.
- [ ] **Ablation runner (`scripts/run_ablation.py`)** — currently raises
      `NotImplementedError`. Needs: load any config in `configs/`, instantiate
      `Memory` with the matching `CognitiveHooksConfig`, run the configured
      dataset through `Memory.add` + `Memory.search` + optional
      `Memory.run_sleep_consolidation`, persist per-turn outputs under
      `expected_outputs/<run_id>/`.
- [ ] **Table generator (`scripts/make_tables.py`)** — currently raises
      `NotImplementedError`. Needs: read the runs produced by
      `run_ablation.py`, compute the metrics listed in
      `paper/sections/experiments.tex`, emit the LaTeX fragments referenced
      by `paper/main.tex`.
- [ ] **Expected outputs (`expected_outputs/`)** — no numeric results are
      checked in yet. When `run_ablation.py` starts producing stable output
      we will publish the SHA-256 + row counts of each released run so a
      reimplementer can diff their numbers against ours.
- [ ] **Seed manifest freeze** — `configs/cognitivebench_seeds.yaml` is
      currently a declared schema; the actual seeded values are marked
      `TODO` and must be fixed before any headline number is reported.

**Until every box above is ticked, the paper text must (and does, after
Stage 6) avoid quoting headline numbers whose reproduction requires an
unchecked box.**

---

## Relationship to `evaluation_mem0_original/`

`evaluation_mem0_original/` is deliberately untouched. Do **not** copy or
re-use code from it without also copying its upstream attribution. The
`adapters/locomo.py` stub is the *only* planned cross-reference: it will
read the LoCoMo dataset file that lives under
`evaluation_mem0_original/dataset/`, because that's where users who already
ran upstream Mem0 expect the file to be.

---

## Why skeleton-first?

The reviewer's core complaint about the repository was that *claims and
artifacts are not aligned*. Publishing a skeleton with an explicit,
unchecked-boxes checklist is the strongest possible alignment signal we can
give short of a full release: it tells the reader exactly what is and what
is **not** currently reproducible from this repository.
