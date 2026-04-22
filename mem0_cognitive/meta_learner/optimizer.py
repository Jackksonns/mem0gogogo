"""
Meta-Cognitive Optimizer: Bayesian hyperparameter tuning for personalized memory

Implements the meta-cognitive learning mechanism from Section 3.4 of our ACL 2026
paper. This optimizer uses Bayesian optimization to learn user-specific memory
parameters, adapting to individual "memory fingerprints" over time.

Key insight: Different users have different memory needs. A one-size-fits-all
approach (fixed λ, τ) is suboptimal. Our meta-learner continuously adapts
parameters based on observed retention performance.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from mem0_cognitive.meta_learner.configs import MetaLearnerConfig

logger = logging.getLogger(__name__)


class MetaCognitiveOptimizer:
    """
    Learns optimal memory parameters per user via Bayesian optimization.
    
    As described in paper Section 3.4, this optimizer treats memory parameter
    tuning as a black-box optimization problem:
    
        θ* = argmax_θ Performance(user, θ)
    
    where θ = {λ, τ_base, τ_salience} and Performance is measured by retention
    accuracy or downstream task metrics.
    
    The optimizer maintains a probabilistic surrogate model (Gaussian Process)
    of the performance landscape and uses an acquisition function to balance
    exploration vs exploitation.
    
    Example usage:
        >>> optimizer = MetaCognitiveOptimizer(config)
        >>> # After collecting user interaction data...
        >>> optimal_params = optimizer.optimize_for_user(
        ...     user_id="alice",
        ...     dialogue_history=conversation_turns,
        ...     performance_metric=0.85
        ... )
        >>> print(optimal_params)
        {'lambda_value': 1.2, 'tau_base': 120.0, 'tau_salience': 60.0}
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
        
        logger.info(f"MetaCognitiveOptimizer initialized with acquisition={self.config.acquisition_function}")
    
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
            
        Algorithm (Bayesian Optimization loop):
            1. Add observation to history: (current_params, performance)
            2. Update Gaussian Process surrogate model
            3. Compute acquisition function over parameter space
            4. Select next parameters to evaluate
            5. Return updated best parameters
            
        Note: In practice, steps 3-4 require external libraries like scikit-optimize
        or BoTorch. This implementation provides the framework.
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
        
        # Check if enough data for Bayesian optimization
        if len(history) < self.config.n_initial_samples:
            logger.info(
                f"User {user_id}: Only {len(history)} samples, "
                f"need {self.config.n_initial_samples} before Bayesian optimization"
            )
            return self._explore_randomly(user_id)
        
        # Run Bayesian optimization step
        try:
            optimized_params = self._bayesian_optimization_step(user_id)
            self._current_params[user_id] = optimized_params
            
            logger.info(
                f"User {user_id}: Optimized params → {optimized_params} "
                f"after {len(history)} observations"
            )
            
            return optimized_params
            
        except Exception as e:
            logger.error(f"Bayesian optimization failed for user {user_id}: {e}")
            # Fallback to current best
            return self._get_best_params_from_history(user_id)
    
    def _explore_randomly(self, user_id: str) -> Dict[str, float]:
        """
        Random exploration phase before Bayesian optimization kicks in.
        
        Samples parameters uniformly within bounds to build initial dataset.
        """
        import random
        
        params = {}
        for param_name, (lower, upper) in self.config.param_bounds.items():
            params[param_name] = random.uniform(lower, upper)
        
        logger.debug(f"User {user_id}: Random exploration → {params}")
        return params
    
    def _bayesian_optimization_step(self, user_id: str) -> Dict[str, float]:
        """
        Perform one step of Bayesian optimization.
        
        This is a simplified implementation. Production version should use
        scikit-optimize or similar library for proper Gaussian Process modeling.
        """
        history = self._user_histories[user_id]
        
        # Extract parameter vectors and performance values
        X = []  # Parameter vectors
        y = []  # Performance values
        
        for obs in history:
            param_vector = [obs['params'][key] for key in self.config.param_bounds.keys()]
            X.append(param_vector)
            y.append(obs['performance'])
        
        # Simple approach: weighted average of top-k performers
        # (Real BO would use GP + acquisition function)
        k = min(5, len(X))
        top_indices = sorted(range(len(y)), key=lambda i: y[i], reverse=True)[:k]
        
        optimized_params = {}
        for param_idx, param_name in enumerate(self.config.param_bounds.keys()):
            # Weighted average of top-k values
            weights = [y[i] for i in top_indices]
            total_weight = sum(weights)
            
            if total_weight > 0:
                value = sum(X[i][param_idx] * y[i] for i in top_indices) / total_weight
            else:
                value = sum(X[i][param_idx] for i in top_indices) / k
            
            # Clip to bounds
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
