# Mem0-Cognitive: Core Research Module

**Emotion-Weighted Forgetting and Sleep Consolidation for Adaptive LLM Memory**

This module implements the cognitively-inspired mechanisms described in the companion paper:
- **Affective Retention Score**: emotion-weighted Ebbinghaus decay (paper Eq. 2)
- **Sleep Consolidation Engine**: offline memory reconsolidation (prototype; LLM summarization path is still being wired in)
- **Meta-Cognitive Learner**: per-user adaptive parameter tuning via a **top-$k$ reward-weighted averaging heuristic** (an earlier draft framed this as GP-BO with Expected Improvement; that claim has been retracted in favour of the heuristic actually implemented — see `meta_learner/optimizer.py`)

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from mem0_cognitive import (
    EmotionAnalyzer,
    AffectiveRetentionScorer,
    SleepConsolidator,
    MetaCognitiveOptimizer
)

# Extract emotion from utterance
analyzer = EmotionAnalyzer()
emotion = analyzer.extract("I love this feature!")
print(f"Intensity: {emotion['intensity']}")  # 0.92

# Compute retention score
scorer = AffectiveRetentionScorer()
score = scorer.compute(elapsed_turns=50, emotion_intensity=0.85)
print(f"Retention: {score:.3f}")  # 0.923

# Run consolidation cycle (async)
# stats = await consolidator.run_consolidation_cycle()

# Optimize parameters per user
optimizer = MetaCognitiveOptimizer()
params = optimizer.optimize_for_user("alice", history, performance=0.85)
```

## Architecture

```
mem0_cognitive/
├── emotion/           # Zero-shot LLM emotion extraction
│   ├── analyzer.py    # EmotionAnalyzer class
│   └── configs.py     # EmotionConfig dataclass
├── retention/         # Affective retention scoring
│   ├── scorer.py      # AffectiveRetentionScorer
│   └── configs.py     # RetentionConfig
├── consolidation/     # Sleep-based memory consolidation
│   ├── engine.py      # SleepConsolidator
│   └── configs.py     # ConsolidationConfig
├── meta_learner/      # Adaptive parameter tuning (top-k weighted-averaging heuristic)
│   ├── optimizer.py   # MetaCognitiveOptimizer
│   └── configs.py     # MetaLearnerConfig
└── utils/             # Helper functions
    └── helpers.py     # Convenience wrappers
```

## Paper Reference

"Mem0-Cognitive: Emotion-Weighted Forgetting and Sleep Consolidation for Adaptive LLM Memory." ACL 2026 submission. Title is kept in sync with the root `README.md`, `paper/README.md`, and `paper/main.tex`.

## License

Apache 2.0
