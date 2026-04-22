"""Meta-Cognitive Learner Module.

Implements the adaptive per-user parameter tuner described in
Section 3.4 of the paper. The algorithm is a **top-$k$ reward-weighted
averaging heuristic** over observed ``(params, performance)`` pairs,
not Gaussian-Process Bayesian Optimization; see
``mem0_cognitive/meta_learner/optimizer.py`` for the exact update.
"""

from mem0_cognitive.meta_learner.configs import MetaLearnerConfig
from mem0_cognitive.meta_learner.optimizer import MetaCognitiveOptimizer

__all__ = ["MetaCognitiveOptimizer", "MetaLearnerConfig"]
