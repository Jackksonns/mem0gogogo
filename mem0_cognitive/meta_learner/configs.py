"""Configuration dataclass for :class:`MetaCognitiveOptimizer`.

The meta-learner implemented in
:mod:`mem0_cognitive.meta_learner.optimizer` is **not** a Gaussian-Process
Bayesian Optimizer. It is a top-$k$ reward-weighted averaging heuristic
over per-user (parameter, performance) observations. This config
documents the knobs that actually influence the heuristic.

An earlier revision declared an ``acquisition_function`` field and
advertised EI / UCB / POI choices. Those values were never read by the
implementation. The field is retained here for backward compatibility
with existing configuration files but only the top-$k$ strategy is
supported; any other value raises ``ValueError`` at construction time.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

_DEFAULT_INITIAL_PARAMS: Dict[str, float] = {
    "lambda_value": 1.0,
    "tau_base": 100.0,
    "tau_salience": 50.0,
}

_DEFAULT_PARAM_BOUNDS: Dict[str, Any] = {
    "lambda_value": (0.0, 2.0),
    "tau_base": (50.0, 200.0),
    "tau_salience": (25.0, 100.0),
}


@dataclass
class MetaLearnerConfig:
    """Parameters for :class:`MetaCognitiveOptimizer`.

    Attributes:
        enable_meta_learning: Whether to run the adaptive tuner at all.
            When ``False`` callers should skip calling
            ``optimize_for_user``.
        optimization_interval_turns: Number of dialogue turns between
            successive calls to the tuner. Informational; not enforced
            inside the optimizer itself.
        initial_params: Population prior used for new users. Defaults to
            ``lambda_value=1.0, tau_base=100.0, tau_salience=50.0``.
        param_bounds: Per-dimension search bounds used (a) to generate
            random exploration samples during the warm-up period and
            (b) to clip the top-$k$ weighted-averaging update onto a
            legal box.
        update_strategy: Which update strategy the optimizer runs once
            the warm-up phase ends. The only currently supported value
            is ``'topk_weighted_mean'`` (paper Section 3.4, Eq. 6).
        top_k: ``k`` for the top-$k$ reward-weighted-averaging step. The
            paper reports ``k = 5``.
        n_initial_samples: Number of random-exploration observations
            before the top-$k$ step is first invoked.
        maximize_metric: Human-readable tag for the optimized scalar
            (``'retention_rate'``, ``'accuracy'``, etc.); not consumed
            by the update rule.
        acquisition_function: **Deprecated**; only ``'topk_weighted_mean'``
            (equivalently ``'ei'`` for backward compat) is accepted.
            The previous GP-BO / Expected Improvement framing is
            retracted and deferred to future work.
        cluster_users: Informational toggle for a (disabled) user
            clustering step.
        min_user_history_for_optimization: Lower bound on dialogue-turn
            history before the tuner is allowed to run; a policy check,
            not used inside the optimizer.
    """

    enable_meta_learning: bool = True
    optimization_interval_turns: int = 100
    initial_params: Optional[Dict[str, float]] = None
    param_bounds: Optional[Dict[str, Any]] = None
    update_strategy: str = "topk_weighted_mean"
    top_k: int = 5
    n_initial_samples: int = 5
    maximize_metric: str = "retention_rate"

    # Back-compat alias: historically stored in "acquisition_function" and
    # advertised as 'ei'/'ucb'/'poi'. Only 'topk_weighted_mean' and the
    # legacy 'ei' label are accepted so that old configs do not crash.
    acquisition_function: str = "topk_weighted_mean"

    cluster_users: bool = False
    min_user_history_for_optimization: int = 50

    def __post_init__(self):
        if self.initial_params is None:
            self.initial_params = dict(_DEFAULT_INITIAL_PARAMS)
        if self.param_bounds is None:
            self.param_bounds = dict(_DEFAULT_PARAM_BOUNDS)

        # Normalize legacy 'ei' / 'EI' to the new tag so callers that
        # wrote old configs continue to work.
        normalized = self.acquisition_function.lower()
        if normalized in {"ei", "topk_weighted_mean"}:
            self.acquisition_function = "topk_weighted_mean"
        else:
            raise ValueError(
                "MetaLearnerConfig.acquisition_function must be "
                "'topk_weighted_mean' (or the legacy alias 'ei'); "
                "Gaussian-Process Bayesian Optimization with EI / UCB / POI "
                "is not currently implemented and was retracted from the "
                "paper. See mem0_cognitive/meta_learner/optimizer.py."
            )

        if self.update_strategy != "topk_weighted_mean":
            raise ValueError(
                "MetaLearnerConfig.update_strategy must be "
                "'topk_weighted_mean'; no other strategy is currently "
                "implemented."
            )

        if self.top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if self.n_initial_samples < 1:
            raise ValueError("n_initial_samples must be >= 1")
