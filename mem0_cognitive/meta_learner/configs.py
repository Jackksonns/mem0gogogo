"""
Configuration for Meta-Cognitive Learner
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class MetaLearnerConfig:
    """
    Configuration parameters for meta-cognitive learning.
    
    As described in paper Section 3.4, the meta-cognitive learner uses Bayesian
    optimization to learn personalized memory parameters (λ, τ_salience, etc.)
    for each user or domain, adapting to individual "memory fingerprints".
    
    Attributes:
        enable_meta_learning: Whether to run meta-cognitive optimization
        optimization_interval_turns: Number of dialogue turns between optimizations
        initial_params: Initial parameter values for new users
            - lambda_value: Emotional inertia coefficient λ ∈ [0, 2]
            - tau_base: Base time constant τ_base
            - tau_salience: Salience-modulated time constant τ_salience
        param_bounds: Search space bounds for Bayesian optimization
        acquisition_function: Acquisition function for Bayesian optimization
            - 'ei': Expected Improvement
            - 'ucb': Upper Confidence Bound
            - 'poi': Probability of Improvement
        n_initial_samples: Number of random samples before Bayesian optimization
        maximize_metric: Metric to optimize ('retention_rate', 'accuracy', 'f1')
    """
    enable_meta_learning: bool = True
    optimization_interval_turns: int = 100
    initial_params: Dict[str, float] = None
    param_bounds: Dict[str, tuple] = None
    acquisition_function: str = 'ei'
    n_initial_samples: int = 5
    maximize_metric: str = 'retention_rate'
    
    # User modeling
    cluster_users: bool = False  # Cluster users by behavior patterns
    min_user_history_for_optimization: int = 50  # Minimum turns before optimizing
    
    def __post_init__(self):
        if self.initial_params is None:
            self.initial_params = {
                'lambda_value': 1.0,
                'tau_base': 100.0,
                'tau_salience': 50.0
            }
        
        if self.param_bounds is None:
            self.param_bounds = {
                'lambda_value': (0.0, 2.0),
                'tau_base': (50.0, 200.0),
                'tau_salience': (25.0, 100.0)
            }
        
        if self.acquisition_function not in ['ei', 'ucb', 'poi']:
            raise ValueError(
                f"Invalid acquisition function: {self.acquisition_function}. "
                "Must be 'ei', 'ucb', or 'poi'"
            )
