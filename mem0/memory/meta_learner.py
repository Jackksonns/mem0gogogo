"""Search-side adaptive re-weighting of memory parameters.

PROBLEM: Existing memory systems use static, one-size-fits-all parameters
(fixed forgetting-curve decay, uniform scoring weights). This fails to
account for per-user differences in what matters.

SOLUTION: A lightweight adaptive re-weighting layer that treats per-user
parameter tuning as a sequential black-box search. Each trial records an
observation ``(params, reward)`` from the user's implicit feedback
signals; the next trial is proposed by (a) random wide exploration
during a warm-up phase and (b) small Gaussian perturbation around the
current best once the warm-up ends.

This module is deliberately **not** a Gaussian Process Bayesian
Optimizer. An earlier revision described it as "GP-BO with Expected
Improvement"; the implemented algorithm has always been
perturb-and-track-best, so that framing is retracted. Swapping this
body for ``skopt.gp_minimize`` or ``botorch`` is a clean future
extension that does not change the public API.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass
class UserMetaState:
    """Stores the meta-cognitive state for a specific user."""
    user_id: str
    # Current best parameters
    best_S: float = 10.0  # Memory strength factor in Ebbinghaus curve
    best_weights: Dict[str, float] = field(default_factory=lambda: {
        "freq": 0.3,
        "recency": 0.25,
        "emotion": 0.25,
        "base": 0.2
    })
    # History of (params, reward) trials for the adaptive re-weighter.
    trials: list = field(default_factory=list)
    confidence: float = 0.0  # How confident we are in the current params (0-1)


class MetaCognitiveLearner:
    """Per-user adaptive parameter re-weighter (perturb-and-track-best).

    Core logic:

    1. Maintain a history of ``(params, reward)`` pairs for each user.
    2. During the first 5 trials, propose parameters by uniformly
       sampling ``S`` from a wide range; this is the warm-up phase.
    3. Afterwards, perturb the current best parameters by multiplicative
       Gaussian noise; widen the noise when recent rewards stagnate.
    4. Update the stored best whenever a trial reports a higher reward.

    This is **not** a Gaussian Process Bayesian Optimizer; the previous
    docstring described it as "GP-BO with Expected Improvement" and
    that framing is retracted.
    """

    def __init__(self):
        self.user_states: Dict[str, UserMetaState] = {}
        self.default_S_range = (1.0, 50.0)  # Reasonable range for decay factor
        self.default_weight_range = (0.0, 1.0)

    def get_user_state(self, user_id: str) -> UserMetaState:
        """Get or initialize user state."""
        if user_id not in self.user_states:
            self.user_states[user_id] = UserMetaState(user_id=user_id)
        return self.user_states[user_id]

    def suggest_parameters(self, user_id: str) -> Tuple[float, Dict[str, float]]:
        """Propose the next parameter set for ``user_id``.

        The implemented policy is deliberately simple: during the
        warm-up phase (fewer than 5 trials) sample ``S`` uniformly
        within the default range to get broad coverage; afterwards,
        perturb the current best parameters with Gaussian noise whose
        scale widens if the most recent rewards are stagnant. No GP,
        no Expected Improvement.
        """
        state = self.get_user_state(user_id)
        
        # If few trials, explore more aggressively around a wider range
        if len(state.trials) < 5:
            # Wide exploration: sample from broader range
            import random
            exploratory_S = random.uniform(self.default_S_range[0], self.default_S_range[1])
            return self._perturb_params(exploratory_S, state.best_weights, scale=0.3)
        
        # Exploit-then-explore step (not a GP Bayesian-optimization
        # step; see module docstring). `_get_best_trial` is retained in
        # case a future refactor reintroduces a surrogate model.
        best_params, best_reward = self._get_best_trial(state)

        # Heuristic: If recent rewards are stagnant, increase exploration
        recent_rewards = [t['reward'] for t in state.trials[-5:]]
        if len(recent_rewards) > 1 and max(recent_rewards) - min(recent_rewards) < 0.05:
            # Stagnant: Explore more aggressively
            return self._perturb_params(state.best_S, state.best_weights, scale=0.3)
        
        # Otherwise: Exploit around best known, with small noise
        return self._perturb_params(state.best_S, state.best_weights, scale=0.1)

    def _perturb_params(self, S: float, weights: Dict[str, float], scale: float = 0.1) -> Tuple[float, Dict[str, float]]:
        """Add Gaussian noise to parameters."""
        import random
        new_S = max(self.default_S_range[0], min(self.default_S_range[1], 
                                                 S * (1 + random.gauss(0, scale))))
        
        new_weights = {}
        total = 0
        for k, v in weights.items():
            w = max(0.0, v * (1 + random.gauss(0, scale)))
            new_weights[k] = w
            total += w
        
        # Normalize weights to sum to 1.0
        if total > 0:
            new_weights = {k: v/total for k, v in new_weights.items()}
            
        return new_S, new_weights

    def record_feedback(self, user_id: str, params: Dict[str, Any], reward: float):
        """
        Record a trial result and update the user's best parameters if this reward is higher.
        
        Args:
            user_id: The user identifier.
            params: The parameters used during the trial {'S': ..., 'weights': ...}.
            reward: The observed reward signal (0.0 - 1.0).
        """
        state = self.get_user_state(user_id)
        
        trial_data = {
            "params": params,
            "reward": reward
        }
        state.trials.append(trial_data)
        
        # Keep only last N trials to adapt to changing user behavior (Concept Drift)
        if len(state.trials) > 50:
            state.trials = state.trials[-50:]
        
        # Update best if this is the new maximum
        _, current_best_reward = self._get_best_trial(state)
        if reward > current_best_reward:
            state.best_S = params['S']
            state.best_weights = params['weights'].copy()
            state.confidence = min(1.0, state.confidence + 0.05)
            logger.info(f"[MetaLearn] User {user_id}: New best params found! S={state.best_S:.2f}, Reward={reward:.4f}")
        else:
            state.confidence = max(0.0, state.confidence - 0.01)

    def _get_best_trial(self, state: UserMetaState) -> Tuple[Dict, float]:
        """Return the parameters and reward of the best trial so far."""
        if not state.trials:
            return {"S": state.best_S, "weights": state.best_weights}, 0.0
        
        best_trial = max(state.trials, key=lambda x: x['reward'])
        return best_trial['params'], best_trial['reward']

    def get_optimized_params(self, user_id: str) -> Tuple[float, Dict[str, float]]:
        """
        Return the current *best known* parameters for inference.
        Unlike `suggest_parameters`, this does not add exploration noise.
        """
        state = self.get_user_state(user_id)
        return state.best_S, state.best_weights
