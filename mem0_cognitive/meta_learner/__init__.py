"""
Meta-Cognitive Learner Module

Implements Bayesian optimization for personalized memory parameter tuning
as described in Section 3.4 of our ACL 2026 paper.
"""

from mem0_cognitive.meta_learner.optimizer import MetaCognitiveOptimizer
from mem0_cognitive.meta_learner.configs import MetaLearnerConfig

__all__ = ["MetaCognitiveOptimizer", "MetaLearnerConfig"]
