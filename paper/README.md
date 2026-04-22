# Paper source

LaTeX writeup of the math and algorithms behind the [dreamfeed](../README.md) fork.
This directory is **documentation**, not a submission artifact: it describes
what the three cognitive modules do and why, but it is not being submitted to
any venue and the experiment tables are intentionally unpopulated.

Working title (mirrored in `\title{...}` in `main.tex`):

> Emotion-Weighted Forgetting and Sleep Consolidation for Adaptive LLM Memory

## File structure

```
paper/
├── main.tex                    # Main document (imports all sections)
├── references.bib              # Bibliography
├── README.md                   # This file
└── sections/
    ├── introduction.tex
    ├── related_work.tex
    ├── methodology.tex
    ├── experiments.tex
    ├── conclusion.tex
    └── appendix.tex
```

> **Note on figures.** Two `\includegraphics` references remain in
> `experiments.tex` (`figures/memory_growth.pdf`) and `appendix.tex`
> (`figures/meta_convergence.pdf`). These PDFs do not exist in the repo;
> compilation emits "missing file" warnings. This is intentional — the
> figures would come from an evaluation harness that is not part of this
> fork (see "Status of experiments" below).

## Compilation

`main.tex` imports `acl_natbib.sty`. Drop the venue-provided style file into
this directory, or upload to Overleaf against an ACL/EMNLP template.

```bash
cd paper/
pdflatex main.tex
bibtex main.aux
pdflatex main.tex
pdflatex main.tex
```

`\usepackage[review]{acl_natbib}` means authors render anonymously; this is a
leftover from an earlier draft and is harmless for documentation use.

## Section status

Each section is in one of three states. The states mirror the "honest
checklist" convention used elsewhere in the repo.

- **aligned** — text has been audited against the code in `mem0_cognitive/`
  and cross-references are correct.
- **narrowed-in-stage-N** — text was narrowed in a specific remediation
  stage and the narrowing is fully in place.
- **work-in-progress** — section still overclaims or under-specifies
  something; follow-ups listed below.

| Section | State | Notes |
|---------|-------|-------|
| **Introduction** | narrowed | Collapsed the top-level "Active Inference" framing to a mechanism-level description (`emotion-weighted retention + asynchronous consolidation + adaptive reweighting`). Dropped the `55% / 29%` headline numbers; contribution bullets match the empty experiments tables. |
| **Related Work** | narrowed | Replaced the unbounded `"To our knowledge, we are the first ..."` claim with a bounded novelty statement scoped to a named set of surveyed memory layers (LangChain, MemGPT, Generative Agents, Mem0, Zep). Explicitly disclaims priority over the individual cognitive-science ideas. |
| **Methodology** | aligned | Retention formula pinned to Eq. \ref{eq:retention}; Gaussian-Process / Expected-Improvement language replaced with a top-$k$ reward-weighted-averaging heuristic; cluster step labelled "greedy compatible-cluster assignment" rather than DBSCAN; `Emotion Extractor: Implementation and Reliability Caveats` subsection in place. |
| **Experiments** | work-in-progress | Artifact-first structure (research questions, protocol, metric definitions, baseline set, ablation matrix) is in place, but **every numeric cell is intentionally empty** (`---`). No evaluation harness is shipped with this fork. |
| **Conclusion** | narrowed | Dropped the `29% / 55% / sub-50ms` headline numbers and the "active inference" positioning. Limitations section explicitly states the extractor is not a validated measurement instrument, that there is no belief update / generative model, and that the clustering algorithm is greedy rather than DBSCAN. |
| **Appendix** | work-in-progress | Prompt templates and case studies describe what the implementation does; the CognitiveBench generation protocol describes a dataset that has not been generated. |

## Claim-to-code mapping

Each claim in the writeup maps to a concrete location in `mem0_cognitive/`.
This table is the single source of truth for "where is this implemented?".

| Claim | Code path |
|-------|-----------|
| Eq. \ref{eq:retention} monotone in $E$, Ebbinghaus-collapse when $\lambda = 0$ | `mem0_cognitive/retention/scorer.py`, pinned by `tests/cognitive/test_retention_direction.py` (49 tests) |
| Offline consolidation with writeback + audit log + non-recursive fallback | `mem0_cognitive/consolidation/engine.py`, pinned by `tests/cognitive/test_sleep_consolidator.py` (14 tests) |
| Top-$k$-weighted adaptive parameter tuner | `mem0_cognitive/meta_learner/optimizer.py`, with `MetaLearnerConfig(update_strategy="topk_weighted_mean")`, pinned by `tests/cognitive/test_meta_learner.py` (24 tests) |
| Opt-in integration with `Memory.add()` / `Memory.search()` | `mem0_cognitive/integration/hooks.py`, pinned by `tests/cognitive/test_integration_hooks.py` (24 tests) |
| Emotion extractor two-path (LLM + lexicon fallback) with method logging | `mem0_cognitive/emotion/analyzer.py`, pinned by `tests/cognitive/test_emotion_analyzer.py` (26 tests) |

## Status of experiments

There is no evaluation harness in this fork that produces the numbers the
experiments section would need. The tests in `tests/cognitive/` pin
algorithmic properties (monotonicity, Ebbinghaus fallback, writeback
correctness, top-$k$ averaging math) but do **not** claim retrieval quality
gains on any public benchmark. If you want to turn this writeup into a paper,
that evaluation harness is the work that still needs to happen.

## Repository

Companion repository: <https://github.com/Jackksonns/mem0gogogo>
