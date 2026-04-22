"""
Utility helper functions for Mem0-Cognitive

Convenience functions that wrap the core modules for easy integration.
"""

from typing import Any, Dict


def extract_emotion(utterance: str, 
                   model_name: str = "gpt-4o-mini",
                   temperature: float = 0.0) -> Dict[str, Any]:
    """
    Extract emotional features from text using zero-shot LLM.
    
    Convenience wrapper around EmotionAnalyzer.
    
    Args:
        utterance: Text to analyze
        model_name: LLM model to use
        temperature: Sampling temperature (0 for deterministic)
        
    Returns:
        Dictionary with intensity, valence, arousal, rationale
        
    Example:
        >>> emotion = extract_emotion("I love this feature!")
        >>> print(emotion['intensity'])
        0.92
    """
    from mem0_cognitive.emotion.analyzer import EmotionAnalyzer
    from mem0_cognitive.emotion.configs import EmotionConfig
    
    config = EmotionConfig(
        model_name=model_name,
        temperature=temperature
    )
    analyzer = EmotionAnalyzer(config)
    return analyzer.extract(utterance)


def compute_retention_score(elapsed_turns: int,
                           emotion_intensity: float = 0.0,
                           lambda_value: float = 1.0,
                           tau_base: float = 100.0,
                           tau_salience: float = 50.0) -> float:
    """Convenience wrapper around :class:`AffectiveRetentionScorer`.

    Implements paper Eq. 2::

        R_affective(m, t) = exp(-t / (tau_base * (1 + lambda * E)))

    Args:
        elapsed_turns: Turns since memory creation.
        emotion_intensity: ``E`` in ``[0, 1]``.
        lambda_value: Emotional-inertia coefficient ``lambda`` in
            ``[0, 2]``.
        tau_base: Baseline time constant (turns).
        tau_salience: **Deprecated.** Historically documented as a
            second time constant, but the retention formula never
            actually needed it; the scorer ignores it. Kept here only
            so existing callers do not break. Supplying a non-default
            value emits a ``DeprecationWarning``.

    Returns:
        Retention probability in ``(0, 1]`` \u2014 monotonically
        non-decreasing in ``emotion_intensity`` for ``lambda_value >= 0``.
    """
    from mem0_cognitive.retention.configs import RetentionConfig
    from mem0_cognitive.retention.scorer import AffectiveRetentionScorer

    config = RetentionConfig(
        lambda_value=lambda_value,
        tau_base=tau_base,
        tau_salience=tau_salience,
    )
    scorer = AffectiveRetentionScorer(config)
    return scorer.compute(elapsed_turns, emotion_intensity)


async def run_consolidation_cycle(memory_store,
                                  clustering_threshold: float = 0.85,
                                  generalization_strategy: str = "summarize") -> Dict[str, int]:
    """
    Run one sleep consolidation cycle.
    
    Convenience wrapper around SleepConsolidator.
    
    Args:
        memory_store: Memory storage backend
        clustering_threshold: Similarity threshold for clustering
        generalization_strategy: 'average', 'summarize', or 'keep_best'
        
    Returns:
        Statistics dictionary with counts of retrieved, clustered, consolidated, pruned
        
    Example:
        >>> stats = await run_consolidation_cycle(store)
        >>> print(f"Pruned {stats['pruned']} memories")
    """
    from mem0_cognitive.consolidation.configs import ConsolidationConfig
    from mem0_cognitive.consolidation.engine import SleepConsolidator
    
    config = ConsolidationConfig(
        clustering_threshold=clustering_threshold,
        generalization_strategy=generalization_strategy
    )
    consolidator = SleepConsolidator(memory_store, config)
    return await consolidator.run_consolidation_cycle()


def get_decay_curve_data(max_turns: int = 500,
                        lambda_value: float = 1.0,
                        emotion_levels: list = None) -> Dict[str, list]:
    """
    Generate decay curve data for visualization.
    
    Useful for plotting Figure 2 in the paper showing how emotional
    memories decay slower than neutral ones.
    
    Args:
        max_turns: Maximum turns to plot
        lambda_value: Emotional inertia coefficient
        emotion_levels: List of E values to compare (default: [0, 0.5, 1.0])
        
    Returns:
        Dictionary mapping emotion level strings to score lists
        
    Example:
        >>> curves = get_decay_curve_data(emotion_levels=[0.0, 0.5, 1.0])
        >>> # Plot using matplotlib: plt.plot(curves['E=0.0']), etc.
    """
    from mem0_cognitive.retention.configs import RetentionConfig
    from mem0_cognitive.retention.scorer import AffectiveRetentionScorer
    
    config = RetentionConfig(lambda_value=lambda_value)
    scorer = AffectiveRetentionScorer(config)
    return scorer.get_decay_curve(max_turns, emotion_levels)
