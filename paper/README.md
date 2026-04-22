# Paper Source Files

This directory contains the LaTeX source for the ACL/EMNLP 2026 submission:

> **"Emotion-Weighted Forgetting and Sleep Consolidation for Adaptive LLM Memory"**

The title here is kept in sync with the `\title{...}` declared in `main.tex`
and with the companion repository's root `README.md`. Earlier drafts of this
file used different working titles ("Cognitively-Inspired Dynamic Memory
Evolution for Long-Term Dialogue Systems",
"Biologically-Inspired Memory Management for LLM Agents"); those drafts have
been retired and the title above is authoritative across the whole repository.

## File Structure

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
> (`figures/meta_convergence.pdf`). The corresponding PDFs have not yet been
> regenerated against the Stage 2 retention-direction fix or the Stage 4
> consolidator rewrite, and the evaluation harness in `evaluation_cognitive/`
> that will produce them is still a skeleton (see its
> [README](../evaluation_cognitive/README.md)). Compilation currently emits
> "missing file" warnings; this is intentional and will not be silenced with
> placeholder images.

## Compilation

The paper uses `acl_natbib.sty` (from the ACL/EMNLP author kit). You will need
to drop the venue-provided style file into this directory, or upload the
project to Overleaf against an ACL/EMNLP template.

```bash
cd paper/
pdflatex main.tex
bibtex main.aux
pdflatex main.tex
pdflatex main.tex
# Output: main.pdf
```

`main.tex` is configured with `\usepackage[review]{acl_natbib}` so that authors
are rendered anonymously. For a preprint, switch to `[preprint]` and add an
explicit `\author{...}` block.

## Section Status

> These are **honest** status markers. They mirror the Stage-6 reproducibility
> checklist convention used in
> [`../evaluation_cognitive/README.md`](../evaluation_cognitive/README.md):
> each section is in exactly one of three states.
>
> - **aligned** — text has been audited against the companion-repository code
>   path and cross-references are correct.
> - **narrowed-in-stage-N** — text was narrowed in a specific remediation
>   stage and the narrowing is fully in place.
> - **work-in-progress** — section is acknowledged to still overclaim or
>   under-specify something, with the concrete follow-up described in the
>   checklist below.

| Section | State | What was done / what still needs to happen |
|---------|-------|--------------------------------------------|
| **Introduction** | narrowed-in-stage-7 | Collapsed the top-level "Active Inference" framing to a mechanism-level description (`emotion-weighted retention + asynchronous consolidation + adaptive reweighting`). Dropped the `55% / 29%` headline numbers; the contribution bullets now match the (empty) experiments tables in the Stage-6 artifact-first section. |
| **Related Work** | narrowed-in-stage-7 | Replaced the unbounded `"To our knowledge, we are the first ..."` claim with a bounded novelty statement scoped to a named set of surveyed open-source memory layers (LangChain, MemGPT, Generative Agents, Mem0, Zep). Explicitly disclaims priority over the individual cognitive-science ideas each module draws from. |
| **Methodology** | narrowed-in-stage-{2,3,7}, partially aligned | Stage 2 locked in the retention-formula direction (Eq. \ref{eq:retention} is monotone non-decreasing in $E$ at every fixed $t$). Stage 3 retracted the Gaussian-Process Bayesian-Optimisation claim and relabelled the update rule as a top-$k$ reward-weighted-averaging heuristic. Stage 7 (this stage) scrubbed the residual "Active Inference" framing at the top of the section, reclassified the consolidation cluster step as `greedy compatible-cluster assignment` (not DBSCAN), and added an `Emotion Extractor: Implementation and Reliability Caveats` subsection. |
| **Experiments** | narrowed-in-stage-6 | Rewritten as an artifact-first section: research questions, evaluation protocol, metric definitions, baseline set, and the ablation matrix are all kept and cross-referenced to `evaluation_cognitive/`; every numeric cell is intentionally emptied (`---`) until the runner in `evaluation_cognitive/scripts/run_ablation.py` emits stable outputs under `evaluation_cognitive/expected_outputs/`. See the checklist in `evaluation_cognitive/README.md`. |
| **Conclusion** | narrowed-in-stage-7 | Dropped the `29% / 55% / sub-50ms` headline numbers and the "active inference" positioning. Limitations section now explicitly states the extractor is not a validated measurement instrument, that there is no belief update / generative model, and that the clustering algorithm is greedy rather than DBSCAN. Future-directions list now calls out real GP-BO and gold-annotated emotion calibration as the two most prominent deferred items. |
| **Appendix** | work-in-progress | Prompt templates + CognitiveBench generation protocol + case studies. Will be audited once `evaluation_cognitive/generators/cognitivebench.py` is implemented and the seed manifest in `evaluation_cognitive/configs/cognitivebench_seeds.yaml` is frozen (both are currently `NotImplementedError` / `TODO`). |

## Claim-to-Artifact Mapping

Each claim in the paper maps to a specific location in the companion
repository. This table is the single source of truth for "where is this
implemented?" questions.

| Paper claim | Code path / config |
|-------------|--------------------|
| Eq. \ref{eq:retention} monotone in $E$, Ebbinghaus-collapse when $\lambda = 0$ | `mem0_cognitive/retention/scorer.py`, pinned by `tests/cognitive/test_retention_direction.py` (49 tests) |
| Offline consolidation with writeback + audit log + non-recursive fallback | `mem0_cognitive/consolidation/engine.py`, pinned by `tests/cognitive/test_sleep_consolidator.py` (14 tests) |
| Top-$k$-weighted adaptive parameter tuner | `mem0_cognitive/meta_learner/optimizer.py`, configured via `mem0_cognitive/meta_learner/configs.MetaLearnerConfig` (`update_strategy="topk_weighted_mean"`) |
| Opt-in integration with `Memory.add()` / `Memory.search()` | `mem0_cognitive/integration/hooks.py`, pinned by `tests/cognitive/test_integration_hooks.py` (24 tests) |
| Emotion extractor two-path with LLM + lexicon fallback and method logging | `mem0_cognitive/emotion/analyzer.py` (method field on every stored memory) |
| Ablation matrix (full, no-emotion, no-consolidation, no-meta-learner) | `evaluation_cognitive/configs/ablation_*.yaml` |
| CognitiveBench seed manifest schema (values are `TODO`) | `evaluation_cognitive/configs/cognitivebench_seeds.yaml` |
| LoCoMo metric-definition pin (answer-accuracy vs retrieval-faithfulness vs attribution-correctness) | `evaluation_cognitive/adapters/locomo.py::LoCoMoAdapter.describe_metric` (currently `NotImplementedError`) |

Any paper claim that does not appear in this table is an oversight; please
open an issue on the companion repository.

## Submission Checklist

- [ ] Run the `evaluation_cognitive/` harness (once the generator and runner are implemented) and verify every numeric cell in `experiments.tex` and `appendix.tex` matches what the scripts produce. Until then, numeric cells remain `---`.
- [ ] Regenerate `figures/memory_growth.pdf` and `figures/meta_convergence.pdf` from real run logs.
- [ ] Verify all bibliography entries render correctly with `acl_natbib`.
- [ ] Confirm page count against the target venue's limit.
- [ ] Provide anonymized companion-repository link in submission materials (not embedded in the PDF).
- [x] `\usepackage[review]{acl_natbib}` with no `\author{...}` block — anonymization baseline is in place (Stage 1).
- [x] Repository identity, package name, clone URL, and paper title are unified across root `README.md`, `paper/README.md`, `paper/main.tex`, `ARCHITECTURE.md`, and `mem0_cognitive/README.md` (Stage 1).
- [x] Retention-formula direction matches the paper's own description (Stage 2).
- [x] Paper's adaptive-tuner description matches the implemented algorithm (Stage 3).
- [x] Consolidation engine writes back consolidated records, deletes sources, and emits an audit log (Stage 4).
- [x] Cognitive modules are wired into `Memory.add()` / `Memory.search()` behind an opt-in flag (Stage 5).
- [x] Evaluation harness split into `evaluation_mem0_original/` (upstream, untouched) and `evaluation_cognitive/` (ours, skeleton) (Stage 6).
- [x] Narrative scrubbed of unsupported active-inference framing and unbounded novelty claims (Stage 7, this stage).

## Repository

Companion repository: <https://github.com/Jackksonns/mem0gogogo>

For double-blind review, please refer to the anonymised repository link that
will be provided in the submission system. Do not link this GitHub URL directly
in the PDF.
