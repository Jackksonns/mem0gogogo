# Mem0-Cognitive: Core Research Module

**Biologically-Inspired Memory Enhancement for LLM Agents**

This module implements the cognitive psychology mechanisms described in our ACL 2026 submission:
- **Affective Retention Score**: Emotion-weighted forgetting curve
- **Sleep Consolidation Engine**: Offline memory reconsolidation
- **Meta-Cognitive Learner**: Bayesian hyperparameter optimization

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
├── meta_learner/      # Bayesian parameter optimization
│   ├── optimizer.py   # MetaCognitiveOptimizer
│   └── configs.py     # MetaLearnerConfig
└── utils/             # Helper functions
    └── helpers.py     # Convenience wrappers
```

## Paper Reference

Zhou, Hongyi. "Mem0-Cognitive: Biologically-Inspired Memory Management with Affective Retention and Sleep Consolidation for LLM Agents." ACL 2026 Submission.

## License

Apache 2.0
