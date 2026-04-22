"""
Configuration for Affective Retention Scoring
"""

from dataclasses import dataclass


@dataclass
class RetentionConfig:
    """
    Configuration parameters for affective retention scoring.
    
    As defined in paper Section 3.2, the effective retention score is computed as:
    
        S_eff(t) = S_base(t) · (1 + λ·E)
    
    where:
        - S_base(t) = exp(-t/τ) is the Ebbinghaus exponential decay
        - E ∈ [0, 1] is the emotional intensity
        - λ ∈ [0, 2] is the emotional inertia coefficient
    
    Attributes:
        lambda_value: Emotional inertia coefficient λ (default 1.0, range [0, 2])
        tau_base: Base time constant τ_base for Ebbinghaus decay (default 100 turns)
        tau_salience: Salience-modulated time constant τ_salience (default 50 turns)
        enable_emotion_weighting: Whether to apply emotion modulation
        min_retention_threshold: Minimum score below which memories are pruned
    """
    lambda_value: float = 1.0
    tau_base: float = 100.0  # turns
    tau_salience: float = 50.0  # turns
    enable_emotion_weighting: bool = True
    min_retention_threshold: float = 0.1
    
    # Time decay parameters
    time_unit: str = "turns"  # or "hours", "days"
    
    def __post_init__(self):
        if self.lambda_value < 0 or self.lambda_value > 2:
            raise ValueError("λ (lambda_value) must be in range [0, 2] as per paper")
        if self.tau_base <= 0 or self.tau_salience <= 0:
            raise ValueError("Time constants τ must be positive")
        if not 0 <= self.min_retention_threshold <= 1:
            raise ValueError("Retention threshold must be in [0, 1]")
