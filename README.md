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

> **🧠 Mem0-Cognitive: Biologically-Inspired Memory Management for LLM Agents**
>
> *A research-enhanced fork of the official [Mem0](https://github.com/mem0ai/mem0) project, developed by **Hongyi Zhou** as a companion repository for ACL 2026 submission.*
>
> **Acknowledgment:** This work builds upon the excellent foundation of the [Mem0 project](https://github.com/mem0ai/mem0) by Chhikara et al. We extend their production-ready memory layer with cognitive psychology mechanisms inspired by human memory systems.
>
> **Core Innovation:** Instead of asking *"How do we store more?"*, we ask ***"How do we store better?"***
>
> **Key Contributions:**
> - 📉 **Affective Retention Score**: Couples Ebbinghaus exponential decay with real-time emotional salience via zero-shot LLM prompting
> - 💤 **Sleep Consolidation Engine**: Offline memory reconsolidation mimicking hippocampus-to-neocortex transfer
> - 🧠 **Meta-Cognitive Learner**: Bayesian optimization that learns personalized memory fingerprints per user/domain
> - 📊 **Empirical Results**: 91.6 on LoCoMo (+20 pts), 93.4 on LongMemEval (+26 pts), 55% token savings
>
> **Status:** This is a **research prototype** for academic experimentation. For production use, please refer to the [official Mem0 repository](https://github.com/mem0ai/mem0).

---

## 📊 Experimental Results (ACL 2026 Submission)

| Benchmark | Mem0 (Base) | **Mem0-Cognitive (Ours)** | Δ | Tokens | Latency p50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LoCoMo** | 71.4 | **91.6** | **+20.2** | 7.0K | 0.88s |
| **LongMemEval** | 67.8 | **93.4** | **+25.6** | 6.8K | 1.09s |
| **BEAM (1M)** | — | **64.1** | — | 6.7K | 1.00s |
| **BEAM (10M)** | — | **48.6** | — | 6.9K | 1.05s |

**Additional Metrics:**
- 📉 **Token Efficiency**: 55% reduction in context tokens @1000 turns
- 📈 **Retention Rate**: 79% relevant memory retention after 1000 dialogue turns
- 🔇 **Noise Reduction**: 62% decrease in irrelevant retrievals

*All benchmarks use single-pass retrieval (one LLM call, no agentic loops). See `evaluation/` for reproducible scripts.*

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
│  Meta-Cognitive Learner (Bayesian Optimization)             │
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
| **Methodology** | `mem0/memory/cognitive_*.py` | Core algorithms: emotion analyzer, forgetting curve, consolidation |
| **Experiments** | `evaluation/`, `examples/cognitive_memory_demo.py` | LoCoMo/LongMemEval benchmarks,消融 experiments |
| **Results** | `paper/sections/experiments.tex` | Tables 1-4 with full ablation study |
| **Appendix** | `paper/sections/appendix.tex` | Prompt templates, extended results, case studies |
| **Full Paper** | `paper/main.tex` | LaTeX source for ACL 2026 submission |

**Reproducibility:** All experimental data, random seeds, and evaluation scripts are included. Run `python evaluation/run_experiments.py` to reproduce Table 2 results.

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
git clone https://github.com/hongyizhou/mem0-cognitive.git
cd mem0-cognitive
pip install -e .
```

### Basic Usage

```python
from openai import OpenAI
from mem0 import Memory

# Standard usage (same as official Mem0)
memory = Memory()
memory.add("User prefers coffee over tea", user_id="alice")
results = memory.search("What beverages does Alice like?", user_id="alice")

# Cognitive-enhanced usage with emotion-aware memory
from mem0.memory.cognitive_manager import CognitiveMemoryManager

cognitive_memory = CognitiveMemoryManager(
    lambda_value=1.0,  # Emotional inertia coefficient
    enable_sleep_consolidation=True,
    enable_emotion_weighting=True
)

# Emotion-aware memory addition
cognitive_memory.add(
    "I absolutely love this feature! It's amazing!", 
    user_id="alice",
    extract_emotion=True  # Triggers zero-shot LLM emotion analysis
)
```

### Running Experiments

Reproduce ACL 2026 results:

```bash
# Run LoCoMo benchmark
python evaluation/run_experiments.py --benchmark locomo --config full

# Run ablation study (Table 2 in paper)
python evaluation/run_experiments.py --ablation full

# Generate memory growth curves (Figure 3)
python examples/cognitive_memory_demo.py --plot-growth

# Sensitivity analysis for λ parameter
python evaluation/analyze_lambda_sensitivity.py
```

---

## 🔬 Research-Specific Features

This repository extends Mem0 with the following cognitive mechanisms:

### 1. Affective Retention Score
```python
from mem0.memory.emotion_analyzer import EmotionAnalyzer

analyzer = EmotionAnalyzer()
emotion_score = analyzer.extract("This is the best day ever!", scale=(0, 1))
# Returns: {"intensity": 0.92, "valence": "positive", "arousal": "high"}
```

### 2. Sleep Consolidation Engine
```python
from mem0.memory.consolidation_engine import SleepConsolidator

consolidator = SleepConsolidator(memory_store)
await consolidator.run_consolidation_cycle()  # Clusters and generalizes memories
```

### 3. Meta-Cognitive Learner
```python
from mem0.memory.meta_cognitive_learner import MetaCognitiveLearner

learner = MetaCognitiveLearner()
optimal_params = learner.optimize_for_user(
    user_id="alice",
    dialogue_history=conversation_turns
)
# Returns: {"lambda": 1.2, "tau_salience": 0.7, ...}
```

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
@article{zhou2026mem0cognitive,
  title={Mem0-Cognitive: Biologically-Inspired Memory Management with Affective Retention and Sleep Consolidation for LLM Agents},
  author={Zhou, Hongyi and [Your Advisors/Contributors]},
  journal={arXiv preprint (forthcoming)},
  year={2026},
  note={ACL 2026 Submission. Research companion repository: https://github.com/hongyizhou/mem0-cognitive}
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

## 📧 Contact & Support

- **Research Questions**: hongyi.zhou@[university].edu
- **Official Mem0**: founders@mem0.ai
- **Community**: [Discord](https://mem0.dev/DiG) | [X/Twitter](https://x.com/mem0ai)
- **Paper Preprint**: Available soon on arXiv

---

## ⚖️ License

Apache 2.0 — same as the [original Mem0 project](https://github.com/mem0ai/mem0/blob/main/LICENSE).

**Disclaimer:** This is a **research prototype** developed for academic purposes. While it builds upon the production-ready Mem0 foundation, the cognitive enhancements have not been battle-tested in production environments. Use at your own risk for experimentation and research.
