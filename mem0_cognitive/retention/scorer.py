"""
Affective Retention Scorer: Emotion-weighted forgetting curve

Implements the core mathematical formulation from Section 3.2 of our ACL 2026 paper:

    S_eff(t) = S_base(t) · (1 + λ·E)

where:
    - S_base(t) = exp(-t/τ) is the Ebbinghaus exponential decay
    - E ∈ [0, 1] is the emotional intensity extracted by EmotionAnalyzer
    - λ ∈ [0, 2] is the emotional inertia coefficient (hyperparameter)

This formulation couples traditional time-based forgetting with real-time emotional
salience, enabling "salience-aware memory pruning without catastrophic forgetting"
(as stated in paper Contributions).
"""

import logging
import math
from typing import Dict, Any, Optional
from datetime import datetime

from mem0_cognitive.retention.configs import RetentionConfig

logger = logging.getLogger(__name__)


class AffectiveRetentionScorer:
    """
    Computes affective retention scores for memory items.
    
    As described in paper Section 3.2, this scorer implements an emotion-modulated
    forgetting curve that extends the classic Ebbinghaus model:
    
        S_eff(t) = exp(-t/τ_effective) · (1 + λ·E)
    
    where τ_effective is adaptively modulated by emotional salience.
    
    Key insight from our research: Emotional memories decay slower than neutral ones,
    consistent with psychological findings on flashbulb memories.
    
    Example:
        >>> scorer = AffectiveRetentionScorer(lambda_value=1.0)
        >>> score = scorer.compute(
        ...     elapsed_turns=50,
        ...     emotion_intensity=0.85,
        ...     created_at=datetime.now()
        ... )
        >>> print(f"Retention score: {score:.3f}")
        Retention score: 0.923
    """
    
    def __init__(self, config: Optional[RetentionConfig] = None):
        """
        Initialize the Affective Retention Scorer.
        
        Args:
            config: RetentionConfig object with scoring parameters.
                   If None, uses default configuration (λ=1.0, τ_base=100).
        """
        self.config = config or RetentionConfig()
        
        logger.info(f"AffectiveRetentionScorer initialized with λ={self.config.lambda_value}, "
                   f"τ_base={self.config.tau_base}, τ_salience={self.config.tau_salience}")
    
    def compute(self, 
                elapsed_turns: int,
                emotion_intensity: float = 0.0,
                created_at: Optional[datetime] = None,
                current_time: Optional[datetime] = None) -> float:
        """
        Compute the effective retention score for a memory item.
        
        Args:
            elapsed_turns: Number of dialogue turns since memory creation
            emotion_intensity: Emotional intensity E ∈ [0, 1] from EmotionAnalyzer
            created_at: Timestamp of memory creation (alternative to elapsed_turns)
            current_time: Current timestamp (used with created_at)
            
        Returns:
            Retention score S_eff ∈ [0, ∞), typically in [0, 2] range
            
        Mathematical formulation (paper Eq. 2):
            S_base(t) = exp(-t / τ_effective)
            τ_effective = τ_base - (τ_base - τ_salience) · E
            S_eff = S_base · (1 + λ·E)
            
        Note: The formula ensures that high-emotion memories (E≈1) have:
            - Slower decay (larger τ_effective)
            - Higher initial boost (1 + λ·E term)
        """
        # Use elapsed_turns if provided, otherwise compute from timestamps
        if elapsed_turns is None and created_at is not None:
            if current_time is None:
                current_time = datetime.now()
            # Approximate turns from time difference (assume ~1 turn per minute)
            elapsed_minutes = (current_time - created_at).total_seconds() / 60
            elapsed_turns = max(1, int(elapsed_minutes))
        elif elapsed_turns is None:
            elapsed_turns = 1  # Default to 1 turn if no timing info
        
        # Ensure emotion_intensity is in valid range [0, 1]
        E = max(0.0, min(1.0, emotion_intensity))
        
        # Compute effective time constant τ_effective
        # High emotion → larger τ → slower decay
        tau_effective = self.config.tau_base - (
            self.config.tau_base - self.config.tau_salience
        ) * E
        
        # Ensure τ_effective stays positive
        tau_effective = max(1.0, tau_effective)
        
        # Compute base Ebbinghaus decay: S_base(t) = exp(-t/τ)
        S_base = math.exp(-elapsed_turns / tau_effective)
        
        # Apply emotion modulation if enabled
        if self.config.enable_emotion_weighting:
            λ = self.config.lambda_value
            S_eff = S_base * (1 + λ * E)
        else:
            S_eff = S_base
        
        # Log for debugging (sample at debug level)
        logger.debug(
            f"Retention score computed: t={elapsed_turns}, E={E:.2f}, "
            f"τ_eff={tau_effective:.1f}, S_base={S_base:.3f}, S_eff={S_eff:.3f}"
        )
        
        return S_eff
    
    def compute_batch(self, memory_items: list) -> list:
        """
        Compute retention scores for multiple memory items.
        
        Args:
            memory_items: List of dicts with keys:
                - 'elapsed_turns' or 'created_at'
                - 'emotion_intensity' (optional, default 0.0)
                
        Returns:
            List of (memory_item, score) tuples, sorted by score descending
        """
        scored_items = []
        
        for item in memory_items:
            score = self.compute(
                elapsed_turns=item.get('elapsed_turns'),
                emotion_intensity=item.get('emotion_intensity', 0.0),
                created_at=item.get('created_at')
            )
            scored_items.append((item, score))
        
        # Sort by score descending (highest retention first)
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        return scored_items
    
    def should_retain(self, score: float) -> bool:
        """
        Determine if a memory should be retained based on its score.
        
        Args:
            score: Retention score from compute()
            
        Returns:
            True if score >= min_retention_threshold, False otherwise
            
        This implements the memory pruning决策 described in paper Section 3.2:
        Memories with S_eff < threshold are candidates for removal during
        sleep consolidation.
        """
        return score >= self.config.min_retention_threshold
    
    def get_decay_curve(self, max_turns: int = 500, emotion_levels: list = None) -> Dict[str, list]:
        """
        Generate decay curves for visualization (used in paper Figure 2).
        
        Args:
            max_turns: Maximum number of turns to plot
            emotion_levels: List of emotion intensities to compare (default: [0, 0.5, 1.0])
            
        Returns:
            Dictionary with emotion levels as keys and score lists as values
            
        Example usage for paper figure:
            >>> curves = scorer.get_decay_curve(emotion_levels=[0.0, 0.5, 1.0])
            >>> # Plot curves showing how high-emotion memories decay slower
        """
        if emotion_levels is None:
            emotion_levels = [0.0, 0.5, 1.0]
        
        curves = {}
        
        for E in emotion_levels:
            scores = []
            for t in range(max_turns + 1):
                score = self.compute(elapsed_turns=t, emotion_intensity=E)
                scores.append(score)
            curves[f"E={E}"] = scores
        
        return curves
