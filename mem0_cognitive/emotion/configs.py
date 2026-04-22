"""
Configuration for Emotion Analysis
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class EmotionConfig:
    """
    Configuration parameters for emotion extraction.
    
    Attributes:
        model_name: LLM model to use for zero-shot emotion analysis
        temperature: Sampling temperature (0 for deterministic output)
        scale: Tuple of (min, max) for intensity scoring, default (0, 1)
        enable_lexicon_fallback: Whether to use lexicon-based fallback
        seed: Random seed for reproducibility (ACL 2026 experiments use seed=42)
    """
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    scale: Tuple[float, float] = (0.0, 1.0)
    enable_lexicon_fallback: bool = True
    seed: Optional[int] = 42
    
    # Emotional inertia coefficient λ ∈ [0, 2] as per paper Section 3.2
    lambda_range: Tuple[float, float] = (0.0, 2.0)
    default_lambda: float = 1.0
    
    def __post_init__(self):
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if len(self.scale) != 2 or self.scale[0] >= self.scale[1]:
            raise ValueError("Scale must be a tuple of (min, max) with min < max")
