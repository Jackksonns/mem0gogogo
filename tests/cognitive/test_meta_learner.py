"""Behaviour-contract and convergence-smoke tests for :class:`MetaCognitiveOptimizer`.

The meta-learner's runtime behaviour is described in paper Section 3.4
and in the module docstring of
:mod:`mem0_cognitive.meta_learner.optimizer`. After Stage 3 of the
remediation plan, the optimizer is **not** a Gaussian-Process Bayesian
Optimizer: it runs uniform-random exploration for the first
``n_initial_samples`` observations, then takes a reward-weighted mean of
the top-$k$ observations and clips the result to the per-dimension
bounds declared in :class:`MetaLearnerConfig`.

These tests pin:

1. **Warm-up contract.** The first ``n_initial_samples`` calls return
   uniformly-sampled ``phi`` records inside ``param_bounds`` and do not
   consult past performance.
2. **Top-$k$ transition.** Once the history contains
   ``n_initial_samples`` observations, the optimizer switches to the
   top-$k$ reward-weighted-averaging step.
3. **Weighted-mean correctness.** For a small, fully controllable
   history, the returned ``phi`` equals the reward-weighted mean of the
   top-$k$ parameters, by hand.
4. **Bounds clipping.** When the weighted mean would land outside
   ``param_bounds``, the optimizer clips it onto the legal box.
5. **Retraction sanity.** The deprecated ``acquisition_function`` values
   (``'ucb'``, ``'poi'``) are rejected at config construction; the
   legacy ``'ei'`` label is still accepted and normalised.
6. **Convergence smoke.** On a simple synthetic reward landscape, the
   parameter trajectory moves *towards* the optimum. This is an end-to-
   end sanity check that the heuristic can, in fact, learn; it is not a
   statistical regret claim.
"""

from __future__ import annotations

import random
from typing import Dict, List

import pytest

from mem0_cognitive.meta_learner.configs import MetaLearnerConfig
from mem0_cognitive.meta_learner.optimizer import MetaCognitiveOptimizer

# ---------------------------------------------------------------------------
# Invariant 1: warm-up contract.
# ---------------------------------------------------------------------------


def _run(optimizer: MetaCognitiveOptimizer, user_id: str, rewards: List[float]) -> List[Dict[str, float]]:
    """Helper: feed a sequence of rewards and collect the returned params."""
    outs = []
    for r in rewards:
        phi = optimizer.optimize_for_user(
            user_id=user_id, dialogue_history=[], performance_metric=r
        )
        outs.append(phi)
    return outs


def test_warm_up_returns_samples_inside_param_bounds():
    random.seed(0)
    cfg = MetaLearnerConfig(n_initial_samples=5)
    opt = MetaCognitiveOptimizer(cfg)

    outs = _run(opt, "alice", [0.1, 0.2, 0.3, 0.4])  # still under warm-up

    assert len(outs) == 4
    for phi in outs:
        for name, (lo, hi) in cfg.param_bounds.items():
            assert lo <= phi[name] <= hi


def test_warm_up_length_matches_n_initial_samples():
    """For the first ``n_initial_samples`` calls we should see random samples.

    The optimizer stores the *previous* call's params as ``current_params``
    and uses them in the *next* record. Consequently the history only
    crosses the threshold after the ``n_initial_samples``-th call. We check
    that the ``n_initial_samples``-th call triggers the top-$k$ step.
    """
    random.seed(0)
    cfg = MetaLearnerConfig(n_initial_samples=3)
    opt = MetaCognitiveOptimizer(cfg)

    _run(opt, "bob", [0.1, 0.2])  # two warm-up samples
    assert len(opt._user_histories["bob"]) == 2

    _run(opt, "bob", [0.9])  # third observation triggers top-k
    # After the third call, the history has 3 entries and the current
    # params are the output of the top-k step.
    assert len(opt._user_histories["bob"]) == 3


# ---------------------------------------------------------------------------
# Invariant 2 & 3: top-k weighted-averaging correctness.
# ---------------------------------------------------------------------------


def _seed_history(opt: MetaCognitiveOptimizer, user_id: str, params_and_perf):
    """Inject a pre-baked history directly, skipping the warm-up phase."""
    history = []
    for params, perf in params_and_perf:
        history.append({"params": params.copy(), "performance": perf, "timestamp": None, "turns_count": 0})
    opt._user_histories[user_id] = history
    opt._current_params[user_id] = params_and_perf[-1][0].copy()


def test_topk_weighted_mean_matches_hand_calculation():
    """Pin Eq. 6: returned phi is the reward-weighted mean of the top-k."""
    cfg = MetaLearnerConfig(
        n_initial_samples=1,  # history is seeded below, bypassing warm-up
        top_k=3,
        initial_params={"lambda_value": 1.0, "tau_base": 100.0, "tau_salience": 50.0},
    )
    opt = MetaCognitiveOptimizer(cfg)

    # Five observations, worst to best. top-3 are the last three.
    history = [
        ({"lambda_value": 0.0, "tau_base": 50.0, "tau_salience": 25.0}, 0.10),
        ({"lambda_value": 0.5, "tau_base": 75.0, "tau_salience": 30.0}, 0.20),
        ({"lambda_value": 1.0, "tau_base": 100.0, "tau_salience": 50.0}, 0.60),
        ({"lambda_value": 1.5, "tau_base": 150.0, "tau_salience": 70.0}, 0.80),
        ({"lambda_value": 2.0, "tau_base": 200.0, "tau_salience": 100.0}, 0.90),
    ]
    _seed_history(opt, "carol", history)

    phi = opt.optimize_for_user(
        user_id="carol", dialogue_history=[], performance_metric=0.95
    )

    # After the call, history has 6 entries; top-3 by reward are
    # r=0.95 (just added with current_params = last seeded),
    # r=0.90, r=0.80.
    # Current params at the time of the call were the last seeded row
    # (lambda=2.0, tau_base=200, tau_sal=100).
    expected = {
        "lambda_value": (2.0 * 0.95 + 2.0 * 0.90 + 1.5 * 0.80)
        / (0.95 + 0.90 + 0.80),
        "tau_base": (200.0 * 0.95 + 200.0 * 0.90 + 150.0 * 0.80)
        / (0.95 + 0.90 + 0.80),
        "tau_salience": (100.0 * 0.95 + 100.0 * 0.90 + 70.0 * 0.80)
        / (0.95 + 0.90 + 0.80),
    }
    for k, v in expected.items():
        assert phi[k] == pytest.approx(v, rel=1e-6), f"{k}: got {phi[k]!r}, want {v!r}"


def test_topk_uses_at_most_history_length_observations():
    """``k`` is capped by ``len(history)`` so it does not error on tiny runs."""
    cfg = MetaLearnerConfig(n_initial_samples=1, top_k=10)
    opt = MetaCognitiveOptimizer(cfg)

    history = [
        ({"lambda_value": 0.5, "tau_base": 80.0, "tau_salience": 40.0}, 0.5),
        ({"lambda_value": 1.5, "tau_base": 120.0, "tau_salience": 60.0}, 0.6),
    ]
    _seed_history(opt, "dave", history)

    phi = opt.optimize_for_user(
        user_id="dave", dialogue_history=[], performance_metric=0.7
    )

    for name, (lo, hi) in cfg.param_bounds.items():
        assert lo <= phi[name] <= hi


def test_zero_reward_history_falls_back_to_unweighted_mean():
    """When all rewards are zero the weighted-mean branch divides; guard fires."""
    cfg = MetaLearnerConfig(n_initial_samples=1, top_k=2)
    opt = MetaCognitiveOptimizer(cfg)

    history = [
        ({"lambda_value": 0.0, "tau_base": 60.0, "tau_salience": 30.0}, 0.0),
        ({"lambda_value": 2.0, "tau_base": 160.0, "tau_salience": 80.0}, 0.0),
    ]
    _seed_history(opt, "erin", history)

    phi = opt.optimize_for_user(
        user_id="erin", dialogue_history=[], performance_metric=0.0
    )

    # Unweighted mean of top-2 (pick any 2 since ties; implementation
    # picks the first two by index). All params must still be clipped
    # onto the legal box.
    for name, (lo, hi) in cfg.param_bounds.items():
        assert lo <= phi[name] <= hi


# ---------------------------------------------------------------------------
# Invariant 4: clipping.
# ---------------------------------------------------------------------------


def test_weighted_mean_is_clipped_to_param_bounds():
    """Even if the best history lies at the upper edge, output stays in-bounds."""
    cfg = MetaLearnerConfig(n_initial_samples=1, top_k=3)
    opt = MetaCognitiveOptimizer(cfg)

    # All three top entries sit exactly at the upper bound.
    upper = {"lambda_value": 2.0, "tau_base": 200.0, "tau_salience": 100.0}
    history = [
        (upper, 0.9),
        (upper, 0.9),
        (upper, 0.9),
    ]
    _seed_history(opt, "felix", history)

    phi = opt.optimize_for_user(
        user_id="felix", dialogue_history=[], performance_metric=0.9
    )

    for name, (lo, hi) in cfg.param_bounds.items():
        assert lo <= phi[name] <= hi
        # And it should equal the upper bound, since every top observation
        # already sits there.
        assert phi[name] == pytest.approx(upper[name])


# ---------------------------------------------------------------------------
# Invariant 5: retracted acquisition-function values are rejected at config
# construction.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_af", ["ucb", "poi", "thompson", "ucb1", "something"])
def test_retracted_acquisition_function_values_are_rejected(bad_af):
    """Stage 3 retraction: GP-BO acquisition-function choices raise."""
    with pytest.raises(ValueError):
        MetaLearnerConfig(acquisition_function=bad_af)


@pytest.mark.parametrize("legacy_af", ["ei", "EI", "Ei", "topk_weighted_mean"])
def test_legacy_ei_alias_is_accepted_and_normalised(legacy_af):
    """Old configs with 'ei' must still load and normalise to 'topk_weighted_mean'."""
    cfg = MetaLearnerConfig(acquisition_function=legacy_af)
    assert cfg.acquisition_function == "topk_weighted_mean"


def test_update_strategy_other_than_topk_is_rejected():
    with pytest.raises(ValueError):
        MetaLearnerConfig(update_strategy="gp_bo")


def test_top_k_must_be_positive():
    with pytest.raises(ValueError):
        MetaLearnerConfig(top_k=0)
    with pytest.raises(ValueError):
        MetaLearnerConfig(top_k=-1)


# ---------------------------------------------------------------------------
# Invariant 6: convergence smoke.
# ---------------------------------------------------------------------------


def _synthetic_reward(params: Dict[str, float]) -> float:
    """Quadratic reward peaked at a known, in-bounds optimum.

    The optimum lives at ``lambda=1.0, tau_base=150, tau_salience=60``; the
    reward is bounded in ``(0, 1)`` and strictly decreasing with distance
    from the optimum. This gives the optimizer a clear, smooth, convex
    landscape to move on, which is the least controversial possible smoke
    test of "the heuristic makes progress".
    """
    opt = {"lambda_value": 1.0, "tau_base": 150.0, "tau_salience": 60.0}
    scale = {"lambda_value": 2.0, "tau_base": 150.0, "tau_salience": 75.0}

    sq = 0.0
    for k, target in opt.items():
        sq += ((params[k] - target) / scale[k]) ** 2
    return max(0.0, 1.0 - sq / 3.0)


def _random_params(bounds: Dict[str, tuple]) -> Dict[str, float]:
    return {name: random.uniform(lo, hi) for name, (lo, hi) in bounds.items()}


def test_convergence_smoke_output_beats_initial_params_on_seeded_history():
    """Given an externally-explored history, the top-$k$ step beats the prior.

    This is a mechanism-level smoke test, not a regret bound. We seed a
    diverse observation history by sampling ``phi`` uniformly from the
    param bounds and scoring each against a known convex reward
    landscape. The top-$k$ step must then return a ``phi`` that is
    closer to the true optimum than ``initial_params`` under the same
    reward function.

    This is the cleanest way to exercise the heuristic without being
    confused by the warm-up attribution caveat documented in
    ``_explore_randomly``: during warm-up the optimizer does not
    currently propagate the returned random ``phi`` into
    ``_current_params``, so the smoke test feeds history directly.
    """
    random.seed(7)
    cfg = MetaLearnerConfig(n_initial_samples=1, top_k=5)
    opt = MetaCognitiveOptimizer(cfg)

    optimum = {"lambda_value": 1.0, "tau_base": 150.0, "tau_salience": 60.0}

    # Seed 40 diverse observations.
    history = []
    for _ in range(40):
        phi = _random_params(cfg.param_bounds)
        history.append((phi, _synthetic_reward(phi)))
    _seed_history(opt, "heidi", history)

    # Trigger one top-k step.
    result = opt.optimize_for_user(
        user_id="heidi", dialogue_history=[], performance_metric=0.0
    )

    def dist(p: Dict[str, float]) -> float:
        return sum((p[k] - optimum[k]) ** 2 for k in optimum) ** 0.5

    assert dist(result) < dist(cfg.initial_params), (
        f"top-k output {result} no closer to optimum than initial_params "
        f"{cfg.initial_params}"
    )
    # And the returned reward should beat the warm-up prior's reward.
    assert _synthetic_reward(result) > _synthetic_reward(cfg.initial_params)


def test_convergence_smoke_result_lives_in_convex_hull_of_topk():
    """The returned $\\phi$ is a convex combination of the top-$k$ observations.

    Pins the mechanism: the top-$k$ step does not propose points outside
    the convex hull of its top-$k$ input parameters. In particular,
    every returned coordinate must sit inside ``[min_topk, max_topk]``
    for that dimension (before the final clip, which can only pull it
    inward). This documents the honest limitation that the heuristic
    averages known-good points rather than searching the interior of
    parameter space.
    """
    random.seed(21)
    cfg = MetaLearnerConfig(n_initial_samples=1, top_k=5)
    opt = MetaCognitiveOptimizer(cfg)

    history = []
    for _ in range(40):
        phi = _random_params(cfg.param_bounds)
        history.append((phi, _synthetic_reward(phi)))
    _seed_history(opt, "ida", history)

    # Record the top-k by reward BEFORE the call adds its own observation.
    top_k_records = sorted(history, key=lambda x: x[1], reverse=True)[: cfg.top_k]

    result = opt.optimize_for_user(
        user_id="ida", dialogue_history=[], performance_metric=0.0
    )

    # The call adds a new observation with the *current* params (last
    # seeded row) at reward 0.0, so it does not enter the top-k. The
    # returned phi must therefore live in the coordinate-wise convex
    # hull of the top_k_records above (then clipped to bounds).
    for name, (lo, hi) in cfg.param_bounds.items():
        topk_vals = [p[name] for p, _ in top_k_records]
        # Allow a small numerical tolerance around the hull.
        assert min(topk_vals) - 1e-9 <= result[name] <= max(topk_vals) + 1e-9, (
            f"{name}: {result[name]!r} outside top-k hull "
            f"[{min(topk_vals)!r}, {max(topk_vals)!r}]"
        )
        # And it must still live in the legal box.
        assert lo <= result[name] <= hi


def test_convergence_smoke_is_not_a_gp_claim():
    """Negative-space test: the optimizer must not expose GP-specific API.

    The retraction in Stage 3 removed the Gaussian-Process framing; no GP
    mean / variance accessor should exist on the public surface.
    """
    opt = MetaCognitiveOptimizer()
    for attr in (
        "posterior_mean",
        "posterior_variance",
        "expected_improvement",
        "acquisition",
        "fit_gp",
    ):
        assert not hasattr(opt, attr), f"unexpected GP-BO surface: {attr}"


# ---------------------------------------------------------------------------
# Profile / reset contract (used by SDK integration).
# ---------------------------------------------------------------------------


def test_get_user_profile_returns_none_before_any_observation():
    opt = MetaCognitiveOptimizer()
    assert opt.get_user_profile("unknown") is None


def test_get_user_profile_reports_best_performance():
    random.seed(0)
    opt = MetaCognitiveOptimizer(MetaLearnerConfig(n_initial_samples=3))
    _run(opt, "jane", [0.1, 0.4, 0.3, 0.7])

    profile = opt.get_user_profile("jane")

    assert profile is not None
    assert profile["n_observations"] == 4
    assert profile["best_performance"] == pytest.approx(0.7)


def test_reset_user_clears_history():
    opt = MetaCognitiveOptimizer()
    _run(opt, "kyle", [0.2, 0.3, 0.4])
    assert opt.get_user_profile("kyle") is not None

    assert opt.reset_user("kyle")
    assert opt.get_user_profile("kyle") is None


def test_reset_user_is_idempotent_on_unknown_user():
    opt = MetaCognitiveOptimizer()
    # Should return True (the user is now absent) and not raise.
    assert opt.reset_user("never-existed")
