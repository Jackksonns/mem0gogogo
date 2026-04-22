# `mem0_cognitive` — the dreamfeed extensions

This package holds the three cognitively-inspired mechanisms that the [dreamfeed](../README.md) fork adds on top of upstream [mem0](https://github.com/mem0ai/mem0):

- **Affective retention** — emotion-weighted Ebbinghaus decay, used to re-rank retrieval candidates by an `affective_composite = α·similarity + (1-α)·retention` score.
- **Sleep consolidation** — an offline pass that clusters near-duplicate memories, abstracts each cluster with an LLM, writes the consolidated memory back to the vector store, and audits the originals.
- **Meta-cognitive learner** — a per-user top-$k$ reward-weighted-averaging heuristic that slowly nudges the retention-law parameters (`λ`, `τ_base`) toward whatever is empirically working for that user. Not a Bayesian optimizer; see `meta_learner/optimizer.py` for the honest formulation.

All three are independently ablatable and opt-in via `MemoryConfig.cognitive` on the upstream `Memory` class.

## Installation

Installed automatically when you `pip install -e .` from the repo root.

## Quick start

```python
from mem0_cognitive import (
    EmotionAnalyzer,
    AffectiveRetentionScorer,
    SleepConsolidator,
    MetaCognitiveOptimizer,
)

analyzer = EmotionAnalyzer()
emotion = analyzer.extract("I love this feature!")

scorer = AffectiveRetentionScorer()
retention = scorer.compute(elapsed_turns=50, emotion_intensity=emotion["intensity"])

optimizer = MetaCognitiveOptimizer()
params = optimizer.optimize_for_user("alice", history, performance=0.85)
```

## Layout

```
mem0_cognitive/
├── emotion/           # Zero-shot LLM + lexicon-fallback emotion extractor
├── retention/         # Emotion-weighted retention scorer (paper Eq. 2)
├── consolidation/     # Async sleep-consolidation engine
├── meta_learner/      # Top-k reward-weighted-averaging parameter tuner
├── integration/       # Hooks that wire the above into mem0.Memory
└── utils/             # Shared helpers
```

## Math / theory

The long-form writeup of the retention law, the consolidation algorithm, and the averaging heuristic lives in `paper/` at the repo root. That directory is documentation, not a submission artifact.

## License

Apache 2.0 (inherited from upstream mem0).
