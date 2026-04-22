"""Minimal demo of the ``mem0_cognitive`` primitives.

This script replaces an older demo that imported several non-existent modules
(``mem0.memory.scoring``, ``mem0.memory.forgetting_manager``,
``mem0.memory.consolidation_engine``). Those modules were never implemented;
the cognitive extensions live in the standalone ``mem0_cognitive`` package.

This demo intentionally does NOT spin up a vector store or call any external
LLM. It only exercises the pure-Python primitives so that it runs on any
developer machine without API keys.
"""

from mem0_cognitive import (
    AffectiveRetentionScorer,
    EmotionAnalyzer,
    MetaCognitiveOptimizer,
)
from mem0_cognitive.emotion.configs import EmotionConfig
from mem0_cognitive.retention.configs import RetentionConfig


def demo_emotion_extraction() -> None:
    print("=" * 60)
    print("1) Emotion extraction (lexicon fallback, no LLM required)")
    print("=" * 60)

    # Disable LLM path explicitly so this demo runs without network/API key.
    analyzer = EmotionAnalyzer(
        EmotionConfig(enable_lexicon_fallback=True, seed=42)
    )
    # We bypass the LLM path by calling the lexicon fallback directly; the
    # public .extract() method would otherwise try the LLM first.
    for utterance in [
        "I absolutely love this feature! It's amazing!",
        "This is terrible and frustrating.",
        "The meeting is at 3pm tomorrow.",
    ]:
        result = analyzer._extract_via_lexicon(utterance, analyzer.config.scale)
        print(f"  utterance: {utterance!r}")
        print(f"    -> intensity={result['intensity']:.2f} "
              f"valence={result['valence']} arousal={result['arousal']}")
    print()


def demo_retention_scoring() -> None:
    print("=" * 60)
    print("2) Affective retention scoring over time")
    print("=" * 60)

    scorer = AffectiveRetentionScorer(RetentionConfig(lambda_value=1.0))
    print(f"  {'turns':>6} | {'E=0.0':>8} | {'E=0.5':>8} | {'E=1.0':>8}")
    print(f"  {'-'*6:>6} + {'-'*8:>8} + {'-'*8:>8} + {'-'*8:>8}")
    for t in [0, 10, 50, 100, 200, 500]:
        row = [scorer.compute(elapsed_turns=t, emotion_intensity=e)
               for e in (0.0, 0.5, 1.0)]
        print(f"  {t:>6} | {row[0]:>8.3f} | {row[1]:>8.3f} | {row[2]:>8.3f}")
    print()


def demo_adaptive_parameter_tuning() -> None:
    print("=" * 60)
    print("3) Adaptive parameter tuning (top-k weighted averaging heuristic)")
    print("=" * 60)

    optimizer = MetaCognitiveOptimizer()
    user_id = "demo_user"
    # Simulate a few interaction episodes with observed performance metrics.
    for step, performance in enumerate([0.55, 0.60, 0.58, 0.71, 0.74, 0.72]):
        params = optimizer.optimize_for_user(
            user_id=user_id,
            dialogue_history=[{"turn": i} for i in range(step + 1)],
            performance_metric=performance,
        )
        print(f"  step={step} perf={performance:.2f} -> params={ {k: round(v, 3) for k, v in params.items()} }")
    profile = optimizer.get_user_profile(user_id)
    if profile:
        print(f"  best_performance={profile['best_performance']:.3f} "
              f"observations={profile['n_observations']}")
    print()


def main() -> None:
    demo_emotion_extraction()
    demo_retention_scoring()
    demo_adaptive_parameter_tuning()


if __name__ == "__main__":
    main()
