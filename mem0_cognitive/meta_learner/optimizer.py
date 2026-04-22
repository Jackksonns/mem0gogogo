"""Meta-Cognitive Optimizer: per-user adaptive parameter tuning.

Implements the adaptive-tuning heuristic described in Section 3.4 of the
paper. The class name is retained for backward compatibility, but the
actual algorithm is **not** Gaussian-Process Bayesian Optimization — it
is a **top-k reward-weighted averaging heuristic** over observed
(parameter, performance) pairs, clipped to per-dimension bounds. In
earlier revisions both the paper and this module were labelled "GP-BO
with Expected Improvement"; that framing was larger than the artifact.
This module and the paper have been aligned to describe the heuristic
actually implemented.

A drop-in replacement using a real Gaussian Process surrogate (e.g.
scikit-optimize, BoTorch) is a clean future extension: it slots in at
``_weighted_topk_step`` without changing the public API.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from mem0_cognitive.meta_learner.configs import MetaLearnerConfig

logger = logging.getLogger(__name__)


class MetaCognitiveOptimizer:
    """Per-user adaptive parameter tuner backed by a top-$k$ heuristic.

    This optimizer treats per-user memory-parameter tuning as a black-box
    search ``phi* = argmax_phi Performance(user, phi)`` over
    ``phi = {lambda_value, tau_base, tau_salience}``. It is **not** a
    Gaussian-Process Bayesian Optimizer. Instead, after an initial random
    exploration phase of ``n_initial_samples`` observations, each update
    step computes the reward-weighted mean of the top-$k$ observations in
    the user's history and clips the result to the per-dimension bounds
    declared in :class:`MetaLearnerConfig`. See paper Section 3.4,
    Equation 6 for the exact update.

    Example:
        >>> optimizer = MetaCognitiveOptimizer()
        >>> for perf in [0.60, 0.62, 0.71, 0.75, 0.74, 0.76]:
        ...     phi = optimizer.optimize_for_user(
        ...         user_id="alice",
        ...         dialogue_history=[],
        ...         performance_metric=perf,
        ...     )
        >>> sorted(phi.keys())
        ['lambda_value', 'tau_base', 'tau_salience']
    """
    
    def __init__(self, config: Optional[MetaLearnerConfig] = None):
        """
        Initialize the Meta-Cognitive Optimizer.
        
        Args:
            config: MetaLearnerConfig object with optimization parameters.
                   If None, uses default configuration.
        """
        self.config = config or MetaLearnerConfig()
        self._user_histories = {}  # user_id -> list of (params, performance) tuples
        self._current_params = {}  # user_id -> current best params
        
        logger.info(
            "MetaCognitiveOptimizer initialized with strategy=%s (top-k weighted averaging; k=%d)",
            self.config.update_strategy,
            self.config.top_k,
        )
    
    def optimize_for_user(self, 
                         user_id: str,
                         dialogue_history: List[Dict[str, Any]],
                         performance_metric: float) -> Dict[str, float]:
        """
        Optimize memory parameters for a specific user.
        
        Args:
            user_id: Unique identifier for the user
            dialogue_history: List of dialogue turns with memory interactions
            performance_metric: Observed performance (e.g., retention rate, accuracy)
            
        Returns:
            Dictionary with optimized parameter values
            
        Algorithm (paper Section 3.4, Eq. 6):
            1. Append the current observation ``(current_params,
               performance_metric)`` to the user's history.
            2. If the user has fewer than ``n_initial_samples``
               observations, sample the next ``phi`` uniformly from the
               per-dimension bounds (random exploration).
            3. Otherwise, take the top-``k`` observations by performance
               and return their reward-weighted mean, clipped to the
               per-dimension bounds.

        This is not a Gaussian Process step; the earlier ``acquisition
        function / Expected Improvement`` framing has been retracted in
        favour of an honest description of the implemented heuristic.
        """
        # Get or initialize user history
        if user_id not in self._user_histories:
            self._user_histories[user_id] = []
            self._current_params[user_id] = self.config.initial_params.copy()
        
        history = self._user_histories[user_id]
        current_params = self._current_params[user_id]
        
        # Record observation
        observation = {
            'params': current_params.copy(),
            'performance': performance_metric,
            'timestamp': datetime.now(),
            'turns_count': len(dialogue_history)
        }
        history.append(observation)
        
        logger.debug(
            f"User {user_id}: Recorded performance={performance_metric:.3f} "
            f"with params={current_params}"
        )
        
        # Warm-up phase: use random exploration until we have enough
        # observations for the top-k weighted-averaging update.
        if len(history) < self.config.n_initial_samples:
            logger.info(
                "User %s: %d samples, need %d before the top-k step",
                user_id,
                len(history),
                self.config.n_initial_samples,
            )
            return self._explore_randomly(user_id)
        
        # Run the top-k weighted-averaging update.
        try:
            optimized_params = self._weighted_topk_step(user_id)
            self._current_params[user_id] = optimized_params
            
            logger.info(
                f"User {user_id}: Optimized params → {optimized_params} "
                f"after {len(history)} observations"
            )
            
            return optimized_params
            
        except Exception as e:
            logger.error(
                "Adaptive-tuning update failed for user %s: %s", user_id, e
            )
            # Fallback to current best
            return self._get_best_params_from_history(user_id)
    
    def _explore_randomly(self, user_id: str) -> Dict[str, float]:
        """Warm-up phase: sample ``phi`` uniformly from ``param_bounds``.

        Used for the first ``n_initial_samples`` observations so that
        the subsequent top-$k$ weighted-averaging update has at least
        some spread in its input.
        """
        import random
        
        params = {}
        for param_name, (lower, upper) in self.config.param_bounds.items():
            params[param_name] = random.uniform(lower, upper)
        
        logger.debug(f"User {user_id}: Random exploration → {params}")
        return params
    
    def _weighted_topk_step(self, user_id: str) -> Dict[str, float]:
        """Perform one top-$k$ reward-weighted-averaging update step.

        Concretely: select the ``k = min(self.config.top_k, |history|)``
        observations with the highest recorded performance for this
        user, and for each parameter dimension return the reward-weighted
        mean of that dimension across the top-$k$, projected onto the
        per-dimension bounds declared in :class:`MetaLearnerConfig`.

        This method is deliberately not a Gaussian-Process Bayesian
        Optimization step. A future revision can swap this body out for
        ``skopt.gp_minimize`` or ``botorch`` without changing the public
        API.
        """
        history = self._user_histories[user_id]

        X = [
            [obs["params"][key] for key in self.config.param_bounds.keys()]
            for obs in history
        ]
        y = [obs["performance"] for obs in history]

        k = min(self.config.top_k, len(X))
        top_indices = sorted(range(len(y)), key=lambda i: y[i], reverse=True)[:k]

        optimized_params: Dict[str, float] = {}
        for param_idx, param_name in enumerate(self.config.param_bounds.keys()):
            weights = [y[i] for i in top_indices]
            total_weight = sum(weights)

            if total_weight > 0:
                value = sum(X[i][param_idx] * y[i] for i in top_indices) / total_weight
            else:
                value = sum(X[i][param_idx] for i in top_indices) / k

            lower, upper = self.config.param_bounds[param_name]
            value = max(lower, min(upper, value))
            optimized_params[param_name] = value

        return optimized_params
    
    def _get_best_params_from_history(self, user_id: str) -> Dict[str, float]:
        """Return parameters that achieved highest performance in history."""
        history = self._user_histories.get(user_id, [])
        
        if not history:
            return self.config.initial_params.copy()
        
        best_obs = max(history, key=lambda obs: obs['performance'])
        return best_obs['params'].copy()
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve learned profile for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with:
                - optimal_params: Best parameters found so far
                - n_observations: Number of optimization iterations
                - best_performance: Highest performance achieved
                - convergence_status: Whether optimization has converged
        """
        if user_id not in self._user_histories:
            return None
        
        history = self._user_histories[user_id]
        current_params = self._current_params.get(user_id, self.config.initial_params)
        
        if not history:
            return None
        
        best_performance = max(obs['performance'] for obs in history)
        
        # Simple convergence check: variance of last 5 performances
        recent_performances = [obs['performance'] for obs in history[-5:]]
        if len(recent_performances) >= 3:
            import numpy as np
            variance = np.var(recent_performances)
            converged = variance < 0.01  # Threshold for convergence
        else:
            converged = False
        
        return {
            'optimal_params': current_params,
            'n_observations': len(history),
            'best_performance': best_performance,
            'convergence_status': converged
        }
    
    def reset_user(self, user_id: str) -> bool:
        """
        Reset optimization history for a user (e.g., if user behavior changes).
        
        Args:
            user_id: User identifier
            
        Returns:
            True if user was found and reset, False otherwise
        """
        if user_id in self._user_histories:
            del self._user_histories[user_id]
        if user_id in self._current_params:
            del self._current_params[user_id]
        
        logger.info(f"Reset optimization history for user {user_id}")
        return user_id not in self._user_histories
