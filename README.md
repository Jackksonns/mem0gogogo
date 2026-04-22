<p align="center">
  <a href="https://github.com/mem0ai/mem0">
    <img src="docs/images/banner-sm.png" width="800px" alt="Mem0 - The Memory Layer for Personalized AI">
  </a>
</p>

<p align="center" style="display: flex; justify-content: center; gap: 20px; align-items: center;">
  <span style="font-size: 1.2em; font-weight: bold;">🎓 ACL 2026 Submission: Cognitive Memory Enhancement</span>
</p>

<p align="center">
  <a href="https://mem0.ai">Official Mem0</a>
  ·
  <a href="#paper-companion">📄 Paper Companion Repo</a>
  ·
  <a href="#quickstart">🚀 Quickstart</a>
  ·
  <a href="https://mem0.dev/DiG">Join Discord</a>
</p>

<p align="center">
  <a href="https://mem0.dev/DiG">
    <img src="https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white" alt="Mem0 Discord">
  </a>
  <a href="https://pepy.tech/project/mem0ai">
    <img src="https://img.shields.io/pypi/dm/mem0ai" alt="Mem0 PyPI - Downloads">
  </a>
  <a href="https://github.com/mem0ai/mem0">
    <img src="https://img.shields.io/github/commit-activity/m/mem0ai/mem0?style=flat-square" alt="GitHub commit activity">
  </a>
  <a href="https://arxiv.org/abs/2504.19413">
    <img src="https://img.shields.io/badge/arXiv-2504.19413-B31B1B?logo=arxiv" alt="Original Mem0 Paper">
  </a>
  <a href="https://www.ycombinator.com/companies/mem0">
    <img src="https://img.shields.io/badge/Y%20Combinator-S24-orange?style=flat-square" alt="Y Combinator S24">
  </a>
</p>

---

## 🎯 This Repository: Paper Companion for ACL 2026

> **Mem0-Cognitive: Emotion-Weighted Forgetting and Sleep Consolidation for Adaptive LLM Memory**
>
> *A research-oriented fork of the official [Mem0](https://github.com/mem0ai/mem0) project, prepared as a companion repository for an ACL 2026 submission.*
>
> **Acknowledgment:** This work builds on the [Mem0 project](https://github.com/mem0ai/mem0) by Chhikara et al. The cognitive extensions live in a separate Python package, `mem0_cognitive`, so the upstream `mem0` runtime is unchanged.
>
> **What this repo adds on top of Mem0 (research prototype):**
> - **Affective Retention Score** — an emotion-modulated form of the Ebbinghaus decay curve, computed from zero-shot LLM emotion extraction.
> - **Sleep Consolidation Engine** — offline clustering + LLM-based abstraction of redundant episodic memories into consolidated entries.
> - **Adaptive Parameter Tuning** — lightweight heuristic (top-k weighted averaging over observed performance) that adjusts per-user retention parameters; an earlier draft framed this as Bayesian Optimization but the implementation is a heuristic surrogate, not a Gaussian Process. See `paper/sections/methodology.tex` for the current honest formulation.
>
> **Status:** Research prototype. Numbers reported in this README are those reproduced from the experiments described in `paper/sections/experiments.tex` (LoCoMo + CognitiveBench). Earlier README drafts quoted LongMemEval and BEAM results that were not backed by the manuscript — those claims have been removed. For production memory, use the [official Mem0 repository](https://github.com/mem0ai/mem0).

---

## Experimental Results (numbers reproduced from `paper/sections/experiments.tex`)

The manuscript commits to two evaluation tracks: a synthetic **CognitiveBench** long-context dialogue simulator (Retention@10 / Noise Ratio / Token Savings / Latency) and a secondary track on **LoCoMo** accuracy. Baselines: Vanilla RAG, Mem0 Original, Generative Agents, MemGPT, Zep.

> **Reproducibility status (as of this commit):** the evaluation harness for this paper has been split into two top-level directories, and the harness that belongs to *this* paper is **released as a skeleton** — configs, adapter interfaces, and a checklist, no numeric results yet.
>
> - [`evaluation_cognitive/`](./evaluation_cognitive/) — the harness that produces the paper's numbers. See its [README](./evaluation_cognitive/README.md) for the honest reproducibility checklist. The CognitiveBench data generator and the ablation runner currently raise `NotImplementedError`; headline numbers will be added to the paper's tables only once the checklist is fully ticked and the runner has written stable output under `evaluation_cognitive/expected_outputs/`.
> - [`evaluation_mem0_original/`](./evaluation_mem0_original/) — the unmodified upstream Mem0 evaluation harness (the old `evaluation/` directory, renamed in Stage 6 via `git mv` so git history is preserved). It is retained only for baseline / reference comparisons and is **not** used to produce any number in this paper.
>
> Earlier revisions of this README quoted specific numeric values for CognitiveBench (Retention@10 79.4%, etc.) and LoCoMo (72.5) in the two tables below; those values have been removed pending a real run against the cognitive harness. The earlier README numbers for LongMemEval (93.4) and BEAM (64.1 / 48.6) had already been removed in Stage 1 for the same reason.

---

## 🔬 Research Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Conversation                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Emotion Analyzer (Zero-shot LLM + Lexicon Fallback)        │
│  └─> Extracts: intensity ∈ [0,1], valence, arousal          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Affective Retention Score Calculation                       │
│  S_eff = S_base · (1 + λ·E)  where λ ∈ [0,2]                │
│  └─> Couples Ebbinghaus decay with emotional salience       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Meta-Cognitive Learner (adaptive top-k heuristic)          │
│  └─> Learns optimal λ, τ_salience per user/domain           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Sleep Consolidation Engine (Offline Process)               │
│  └─> Clusters similar memories, generalizes specifics       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Hybrid Retrieval (Semantic + BM25 + Entity Linking)        │
│  └─> Top-K=5 with importance-weighted re-ranking            │
└─────────────────────────────────────────────────────────────┘
```

**Theoretical Foundation:** Our approach reframes memory management as an **active inference problem** (Friston et al., 2017), where the meta-cognitive learner continuously updates a generative model of user relevance to minimize future prediction error.

---

## 📄 Paper Structure & Repository Mapping

This repository accompanies our ACL 2026 submission. Key components:

| Paper Section | Repository Location | Description |
| :--- | :--- | :--- |
| **Methodology** | `mem0_cognitive/{emotion,retention,consolidation,meta_learner}/` | Core algorithms: emotion analyzer, affective retention scorer, sleep consolidator, adaptive parameter tuner |
| **System integration** | `mem0/memory/main.py`, `mem0/memory/meta_learner.py` | Points where cognitive modules are wired into the underlying Mem0 `Memory` class (retrieval-time re-ranking; emotion extraction and consolidation wiring are work-in-progress) |
| **Results** | `paper/sections/experiments.tex` | Research questions, evaluation protocol, and the ablation matrix (numeric cells intentionally empty in this release; see `evaluation_cognitive/README.md`). |
| **Appendix** | `paper/sections/appendix.tex` | Prompt templates, CognitiveBench generation protocol, case studies |
| **Full Paper** | `paper/main.tex` | LaTeX source |
| **Eval (ours, skeleton)** | [`evaluation_cognitive/`](./evaluation_cognitive/) | CognitiveBench generator interface, LoCoMo adapter interface, ablation config YAMLs, runner / table-generation stubs, and the honest reproducibility checklist. |
| **Eval (upstream Mem0)** | [`evaluation_mem0_original/`](./evaluation_mem0_original/) | Unmodified upstream Mem0 evaluation harness (renamed from `evaluation/` via `git mv`); retained for baseline / reference comparisons only. |

**Reproducibility status:** The paper's experiments section is intentionally empty-celled in this release; see the checklist in [`evaluation_cognitive/README.md`](./evaluation_cognitive/README.md). Numeric tables in `paper/sections/experiments.tex` will be populated only after every unchecked box in that checklist is ticked and the runner has produced stable output under `evaluation_cognitive/expected_outputs/`.

---

## 🚀 Quickstart Guide <a name="quickstart"></a>

### Installation

**For production use** (official Mem0):
```bash
pip install mem0ai
# or for enhanced hybrid search:
pip install mem0ai[nlp] && python -m spacy download en_core_web_sm
```

**For research/cognitive features** (this repository):
```bash
git clone https://github.com/Jackksonns/mem0gogogo.git
cd mem0gogogo
pip install -e .
```

The editable install now ships two importable Python packages:
- `mem0` — the upstream Mem0 runtime (unchanged).
- `mem0_cognitive` — the research extensions used by the paper.

### Basic Usage

```python
from openai import OpenAI
from mem0 import Memory

# Standard usage (same as official Mem0)
memory = Memory()
memory.add("User prefers coffee over tea", user_id="alice")
results = memory.search("What beverages does Alice like?", user_id="alice")

# Research-side cognitive components (used by the paper)
from mem0_cognitive import (
    EmotionAnalyzer,
    AffectiveRetentionScorer,
    SleepConsolidator,
    MetaCognitiveOptimizer,
)

analyzer = EmotionAnalyzer()
emotion = analyzer.extract("I absolutely love this feature! It's amazing!")
# -> {'intensity': float in [0,1], 'valence': ..., 'arousal': ..., 'method': 'llm'|'lexicon'}

scorer = AffectiveRetentionScorer()
score = scorer.compute(elapsed_turns=50, emotion_intensity=emotion["intensity"])

# NOTE: the end-to-end wiring (automatic emotion extraction in Memory.add(),
# background consolidation job, retention-aware re-ranking in Memory.search)
# is still work-in-progress. The modules above are the primitives used by the
# paper's experiments; the top-level Memory class currently only integrates
# the adaptive parameter tuner (see mem0/memory/meta_learner.py).
```

### Running Experiments

> **The cognitive evaluation harness is published as a skeleton.** The configs, generator interfaces, adapter interfaces, and reproducibility checklist live under [`evaluation_cognitive/`](./evaluation_cognitive/); the CognitiveBench data generator and the end-to-end ablation runner currently raise `NotImplementedError`. See [`evaluation_cognitive/README.md`](./evaluation_cognitive/README.md) for the full checklist.

What the current release supports:

```bash
# Inspect the ablation configurations that the runner will consume:
ls evaluation_cognitive/configs/
# ablation_full.yaml
# ablation_no_emotion.yaml
# ablation_no_consolidation.yaml
# ablation_no_meta_learner.yaml
# cognitivebench_seeds.yaml

# Skeleton invocation (will raise NotImplementedError until the runner is wired):
python -m evaluation_cognitive.scripts.run_ablation \
    --config evaluation_cognitive/configs/ablation_full.yaml
```

For reference / baseline Mem0 numbers (upstream harness, **not** part of this paper's results):

```bash
python evaluation_mem0_original/run_experiments.py --technique_type mem0 --method add
python evaluation_mem0_original/run_experiments.py --technique_type mem0 --method search
```

---

## 🔬 Research-Specific Features

This repository extends Mem0 with the following cognitive mechanisms:

### 1. Affective Retention Score
```python
from mem0_cognitive import EmotionAnalyzer, AffectiveRetentionScorer

analyzer = EmotionAnalyzer()
emotion = analyzer.extract("This is the best day ever!")
# {'intensity': 0.92, 'valence': 'positive', 'arousal': 'high', 'method': 'llm'}

scorer = AffectiveRetentionScorer()
scorer.compute(elapsed_turns=50, emotion_intensity=emotion["intensity"])
```

### 2. Sleep Consolidation Engine
```python
from mem0_cognitive import SleepConsolidator

consolidator = SleepConsolidator(memory_store)
stats = await consolidator.run_consolidation_cycle()
# See paper/sections/methodology.tex §3.3 for the formal schema-induction formulation.
# Note: the LLM-based summarization path requires an OpenAI-compatible client to
# be configured; the `keep_best` strategy works without one.
```

### 3. Adaptive Parameter Tuner
```python
from mem0_cognitive import MetaCognitiveOptimizer

optimizer = MetaCognitiveOptimizer()
optimal_params = optimizer.optimize_for_user(
    user_id="alice",
    dialogue_history=conversation_turns,
    performance_metric=0.85,
)
# The current implementation performs top-k reward-weighted averaging over
# observed (params, performance) history, with uniform random exploration
# for the first `n_initial_samples` observations. Earlier drafts described
# this as Gaussian-Process Bayesian Optimization with Expected Improvement;
# that claim has been retracted and paper/sections/methodology.tex now
# describes the heuristic that is actually implemented.

---

## 📄 Citation

If you use this research prototype in your work, please cite both the original Mem0 paper and our cognitive enhancement:

### Original Mem0 System
```bibtex
@article{chhikara2025mem0,
  title={Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory},
  author={Chhikara, Prateek and Khant, Dev and Aryan, Saket and Singh, Taranjeet and Yadav, Deshraj},
  journal={arXiv preprint arXiv:2504.19413},
  year={2025}
}
```

### Cognitive Memory Enhancement (ACL 2026 Submission)
```bibtex
@article{mem0cognitive2026,
  title={Mem0-Cognitive: Emotion-Weighted Forgetting and Sleep Consolidation for Adaptive LLM Memory},
  author={Anonymous},
  journal={Under review},
  year={2026},
  note={ACL 2026 submission. Research companion repository: https://github.com/Jackksonns/mem0gogogo}
}
```

**Theoretical Foundations:**
- Friston, K. et al. (2017). "Active inference: A process theory." *Neural Computation*
- Ebbinghaus, H. (1885). *Memory: A Contribution to Experimental Psychology*
- McClelland, J. L., et al. (1995). "Complementary learning systems in hippocampus and neocortex." *Psychological Review*

---

## 🤝 Contributing

**For production features:** Please contribute to the [official Mem0 repository](https://github.com/mem0ai/mem0).

**For research collaboration:** We welcome academic contributions! Please:
1. Fork this repository
2. Create a branch named `feature/[your-feature-name]`
3. Include experimental results showing impact on LoCoMo/LongMemEval benchmarks
4. Submit a pull request with a clear description of the cognitive mechanism

---

## Contact & Support

- **Official Mem0**: founders@mem0.ai
- **Community**: [Discord](https://mem0.dev/DiG) | [X/Twitter](https://x.com/mem0ai)

(Per double-blind review policy, contact info for the cognitive extension authors is omitted until the submission exits the review cycle.)

---

## ⚖️ License

Apache 2.0 — same as the [original Mem0 project](https://github.com/mem0ai/mem0/blob/main/LICENSE).

**Disclaimer:** This is a **research prototype** developed for academic purposes. While it builds upon the production-ready Mem0 foundation, the cognitive enhancements have not been battle-tested in production environments. Use at your own risk for experimentation and research.
