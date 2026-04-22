# Mem0-Cognitive Repository Architecture

## Overview

This repository is structured as a **paper companion** for our ACL 2026 submission, built on top of the official [Mem0](https://github.com/mem0ai/mem0) project. The architecture separates production-ready code from research prototypes.

```
/workspace
├── 📄 paper/                          # LaTeX paper source (ACL 2026 submission)
│   ├── main.tex                       # Main paper document
│   ├── references.bib                 # Bibliography (96 entries)
│   └── sections/                      # Paper sections
│       ├── introduction.tex           # Section 1: Introduction & Contributions
│       ├── methodology.tex            # Section 2: Technical Approach
│       ├── experiments.tex            # Section 3: Experimental Setup
│       ├── related_work.tex           # Section 4: Related Work
│       ├── conclusion.tex             # Section 5: Conclusion
│       └── appendix.tex               # Appendix: Prompts, Extended Results
│
├── 🔬 mem0_cognitive/                 # CORE RESEARCH MODULE (Hongyi's contribution)
│   ├── __init__.py                    # Package initialization
│   ├── README.md                      # Module documentation
│   │
│   ├── emotion/                       # Section 3.1: Emotion Analysis
│   │   ├── __init__.py
│   │   ├── configs.py                 # EmotionConfig dataclass
│   │   └── analyzer.py                # EmotionAnalyzer class
│   │                                   # - Zero-shot LLM extraction
│   │                                   # - Lexicon fallback
│   │                                   # - Prompt template (Appendix)
│   │
│   ├── retention/                     # Section 3.2: Affective Retention
│   │   ├── __init__.py
│   │   ├── configs.py                 # RetentionConfig
│   │   └── scorer.py                  # AffectiveRetentionScorer
│   │                                   # - S_eff = S_base · (1 + λ·E)
│   │                                   # - Ebbinghaus decay with emotion modulation
│   │                                   # - Decay curve generation (Figure 2)
│   │
│   ├── consolidation/                 # Section 3.3: Sleep Consolidation
│   │   ├── __init__.py
│   │   ├── configs.py                 # ConsolidationConfig
│   │   └── engine.py                  # SleepConsolidator
│   │                                   # - Clustering by embedding similarity
│   │                                   # - LLM-based generalization
│   │                                   # - Memory pruning
│   │
│   ├── meta_learner/                  # Section 3.4: Meta-Cognitive Learning
│   │   ├── __init__.py
│   │   ├── configs.py                 # MetaLearnerConfig
│   │   └── optimizer.py               # MetaCognitiveOptimizer
│   │                                   # - Bayesian optimization
│   │                                   # - User-specific parameter tuning
│   │                                   # - Convergence tracking
│   │
│   └── utils/                         # Utility Functions
│       ├── __init__.py
│       └── helpers.py                 # Convenience wrappers
│                                       # - extract_emotion()
│                                       # - compute_retention_score()
│                                       # - run_consolidation_cycle()
│
├── 🧪 evaluation/                     # Experimental Evaluation
│   ├── README.md                      # Evaluation guide
│   ├── evals.py                       # Core evaluation logic
│   ├── prompts.py                     # LLM judge prompts
│   ├── run_experiments.py             # Main experiment runner
│   ├── generate_scores.py             # Score aggregation
│   │
│   ├── metrics/                       # Evaluation Metrics
│   │   ├── utils.py                   # Metric utilities
│   │   └── llm_judge.py               # LLM-as-a-judge implementation
│   │
│   └── src/                           # Baseline Implementations
│       ├── memzero/                   # Our method (Mem0-Cognitive)
│       │   ├── add.py                 # Memory addition with emotion
│       │   └── search.py              # Retrieval with retention scoring
│       ├── rag.py                     # Vanilla RAG baseline
│       ├── zep/                       # Zep baseline
│       └── openai/                    # OpenAI baseline
│
├── 💻 examples/                       # Usage Examples
│   ├── cognitive_memory_demo.py       # Main demo (Section 4.1)
│   ├── meta_cognitive_demo.py         # Meta-learning demo
│   ├── misc/                          # Miscellaneous examples
│   ├── multiagents/                   # Multi-agent scenarios
│   └── mem0-demo/                     # Interactive demo app
│
├── 📚 mem0/                           # Official Mem0 (production base)
│   ├── memory/                        # Core memory management
│   │   ├── main.py                    # Memory class (138K lines)
│   │   ├── meta_learner.py            # Base meta-learner
│   │   ├── storage.py                 # SQLite storage
│   │   └── utils.py                   # Utilities
│   ├── configs/                       # Configuration schemas
│   ├── embeddings/                    # Embedding providers
│   ├── llms/                          # LLM providers
│   ├── vector_stores/                 # Vector database adapters
│   └── reranker/                      # Reranking models
│
├── 🛠️ scripts/                        # Utility Scripts
│   └── check-llms-txt-coverage.py     # Documentation checker
│
├── 📖 docs/                           # Documentation
│   └── core-concepts/cognitive-memory.md  # Cognitive memory guide
│
└── 📝 Root Files
    ├── README.md                      # Main repository README
    ├── ARCHITECTURE.md                # This file
    ├── LICENSE                        # Apache 2.0
    ├── pyproject.toml                 # Python project configuration
    └── poetry.lock                    # Dependency lock file
```

## Key Design Principles

### 1. Separation of Concerns

- **`mem0/`**: Production-ready, battle-tested code from official Mem0 project
- **`mem0_cognitive/`**: Research prototype with cognitive enhancements (ACL 2026)
- **`evaluation/`**: Reproducible experimental scripts
- **`paper/`**: LaTeX source for academic publication

### 2. Modular Architecture

Each cognitive mechanism is encapsulated in its own module:
- `emotion/`: Independent emotion extraction
- `retention/`: Standalone scoring algorithm
- `consolidation/`: Offline processing pipeline
- `meta_learner/`: Adaptive optimization loop

### 3. Reproducibility

All experiments can be reproduced via:
```bash
python evaluation/run_experiments.py --benchmark locomo --config full
```

Random seeds are fixed (seed=42) and all hyperparameters are logged.

### 4. Paper-Code Alignment

Every paper section maps to specific code files:

| Paper Section | Code Location | Key Classes/Functions |
|--------------|---------------|----------------------|
| §3.1 Emotion Extraction | `mem0_cognitive/emotion/analyzer.py` | `EmotionAnalyzer.extract()` |
| §3.2 Affective Retention | `mem0_cognitive/retention/scorer.py` | `AffectiveRetentionScorer.compute()` |
| §3.3 Sleep Consolidation | `mem0_cognitive/consolidation/engine.py` | `SleepConsolidator.run_consolidation_cycle()` |
| §3.4 Meta-Cognitive Learning | `mem0_cognitive/meta_learner/optimizer.py` | `MetaCognitiveOptimizer.optimize_for_user()` |
| §4.1 LoCoMo Benchmark | `evaluation/run_experiments.py` | `run_locomo_evaluation()` |
| §4.2 Ablation Study | `evaluation/src/memzero/` | Full vs. w/o components |
| Appendix | `mem0_cognitive/emotion/configs.py` | `PROMPT_TEMPLATE` |

## Data Flow

```
User Utterance
    │
    ▼
┌─────────────────────┐
│  EmotionAnalyzer    │  ← Extracts E ∈ [0, 1]
│  (mem0_cognitive/   │
│   emotion/)         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  AffectiveRetention │  ← Computes S_eff = S_base·(1+λE)
│  Scorer             │
│  (mem0_cognitive/   │
│   retention/)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Memory Storage     │  ← Stores with retention score
│  (mem0/memory/)     │
└──────────┬──────────┘
           │
           ▼
    [Periodic Sleep Cycle]
           │
           ▼
┌─────────────────────┐
│  SleepConsolidator  │  ← Clusters & generalizes
│  (mem0_cognitive/   │
│   consolidation/)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  MetaCognitive      │  ← Optimizes λ, τ per user
│  Optimizer          │
│  (mem0_cognitive/   │
│   meta_learner/)    │
└─────────────────────┘
```

## Citation

If you use this code in your research:

```bibtex
@article{zhou2026mem0cognitive,
  title={Mem0-Cognitive: Biologically-Inspired Memory Management with Affective Retention and Sleep Consolidation for LLM Agents},
  author={Zhou, Hongyi},
  journal={arXiv preprint (forthcoming)},
  year={2026},
  note={ACL 2026 Submission}
}
```

Also cite the original Mem0 system:

```bibtex
@article{chhikara2025mem0,
  title={Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory},
  author={Chhikara, Prateek and Khant, Dev and Aryan, Saket and Singh, Taranjeet and Yadav, Deshraj},
  journal={arXiv preprint arXiv:2504.19413},
  year={2025}
}
```
