"""
Utility helper functions for Mem0-Cognitive

Convenience functions that wrap the core modules for easy integration.
"""

from typing import Dict, Any, Optional, Tuple
from datetime import datetime


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
    """
    Compute affective retention score for a memory item.
    
    Convenience wrapper around AffectiveRetentionScorer.
    
    Formula (paper Eq. 2):
        S_eff = exp(-t/τ_effective) · (1 + λ·E)
    
    Args:
        elapsed_turns: Turns since memory creation
        emotion_intensity: E ∈ [0, 1] from emotion extraction
        lambda_value: Emotional inertia coefficient λ ∈ [0, 2]
        tau_base: Base time constant τ_base
        tau_salience: Salience-modulated time constant τ_salience
        
    Returns:
        Retention score (higher = more likely to retain)
        
    Example:
        >>> score = compute_retention_score(50, emotion_intensity=0.8)
        >>> print(f"Score: {score:.3f}")
        Score: 0.923
    """
    from mem0_cognitive.retention.scorer import AffectiveRetentionScorer
    from mem0_cognitive.retention.configs import RetentionConfig
    
    config = RetentionConfig(
        lambda_value=lambda_value,
        tau_base=tau_base,
        tau_salience=tau_salience
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
    from mem0_cognitive.consolidation.engine import SleepConsolidator
    from mem0_cognitive.consolidation.configs import ConsolidationConfig
    
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
    from mem0_cognitive.retention.scorer import AffectiveRetentionScorer
    from mem0_cognitive.retention.configs import RetentionConfig
    
    config = RetentionConfig(lambda_value=lambda_value)
    scorer = AffectiveRetentionScorer(config)
    return scorer.get_decay_curve(max_turns, emotion_levels)
