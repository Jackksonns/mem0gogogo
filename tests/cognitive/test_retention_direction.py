"""Directional / monotonicity tests for ``AffectiveRetentionScorer``.

These tests pin the following two invariants of the affective-retention
formula implemented in :mod:`mem0_cognitive.retention.scorer`:

1. **Emotion monotonicity.** At any fixed ``elapsed_turns > 0``, the
   returned retention score is *monotonically non-decreasing* in
   ``emotion_intensity`` when ``lambda_value >= 0``. In other words,
   emotional memories decay no faster than neutral memories \u2014 which is
   the claim made in Section 3.2 of the paper and in the root README.

2. **Time monotonicity.** At any fixed ``emotion_intensity``, the score
   is *monotonically non-increasing* in ``elapsed_turns``.

An earlier implementation of the formula inverted invariant (1): with the
historical defaults (``tau_base=100``, ``tau_salience=50``), high-emotion
memories decayed *faster* than neutral ones. This test file exists to
make that regression impossible to reintroduce silently.
"""

from __future__ import annotations

import math

import pytest

from mem0_cognitive.retention.configs import RetentionConfig
from mem0_cognitive.retention.scorer import AffectiveRetentionScorer

# ---------------------------------------------------------------------------
# Invariant 1: monotonicity in E (the bug this stage fixes).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("elapsed_turns", [1, 5, 10, 25, 50, 100, 200])
@pytest.mark.parametrize("lambda_value", [0.0, 0.5, 1.0, 2.0])
def test_retention_is_monotone_nondecreasing_in_emotion(elapsed_turns, lambda_value):
    """For every fixed (t, lambda), R(E=0) <= R(E=0.25) <= ... <= R(E=1)."""
    scorer = AffectiveRetentionScorer(
        RetentionConfig(lambda_value=lambda_value, tau_base=100.0)
    )
    emotions = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    scores = [scorer.compute(elapsed_turns=elapsed_turns, emotion_intensity=E) for E in emotions]
    for earlier, later in zip(scores, scores[1:]):
        assert later + 1e-9 >= earlier, (
            f"Retention should be monotone non-decreasing in E at "
            f"t={elapsed_turns}, lambda={lambda_value}; got {scores}"
        )


def test_strict_increase_in_emotion_for_positive_lambda_and_time():
    """With lambda > 0 and t > 0, higher E strictly beats lower E.

    This is the tighter form of Invariant 1 and is the canonical
    regression test for the direction bug.
    """
    scorer = AffectiveRetentionScorer(RetentionConfig(lambda_value=1.0, tau_base=100.0))
    r_neutral = scorer.compute(elapsed_turns=50, emotion_intensity=0.0)
    r_mid = scorer.compute(elapsed_turns=50, emotion_intensity=0.5)
    r_salient = scorer.compute(elapsed_turns=50, emotion_intensity=1.0)
    assert r_neutral < r_mid < r_salient


def test_lambda_zero_removes_emotion_weighting():
    """With lambda=0, retention must not depend on E \u2014 pure Ebbinghaus."""
    scorer = AffectiveRetentionScorer(RetentionConfig(lambda_value=0.0, tau_base=100.0))
    ref = scorer.compute(elapsed_turns=50, emotion_intensity=0.0)
    for E in [0.0, 0.25, 0.5, 1.0]:
        got = scorer.compute(elapsed_turns=50, emotion_intensity=E)
        assert got == pytest.approx(ref), f"lambda=0 should ignore E; got {got} vs {ref} at E={E}"


def test_enable_emotion_weighting_false_ignores_lambda():
    """enable_emotion_weighting=False must collapse to the neutral curve."""
    scorer = AffectiveRetentionScorer(
        RetentionConfig(lambda_value=2.0, tau_base=100.0, enable_emotion_weighting=False)
    )
    r_neutral = scorer.compute(elapsed_turns=50, emotion_intensity=0.0)
    r_salient = scorer.compute(elapsed_turns=50, emotion_intensity=1.0)
    assert r_neutral == pytest.approx(r_salient)


# ---------------------------------------------------------------------------
# Invariant 2: monotonicity in t.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("emotion_intensity", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("lambda_value", [0.0, 1.0, 2.0])
def test_retention_is_monotone_nonincreasing_in_time(emotion_intensity, lambda_value):
    """For fixed (E, lambda), R(t) is monotonically non-increasing in t."""
    scorer = AffectiveRetentionScorer(
        RetentionConfig(lambda_value=lambda_value, tau_base=100.0)
    )
    ts = [0, 1, 5, 10, 50, 100, 500, 1000]
    scores = [scorer.compute(elapsed_turns=t, emotion_intensity=emotion_intensity) for t in ts]
    for earlier, later in zip(scores, scores[1:]):
        assert earlier + 1e-9 >= later, (
            f"Retention should be monotone non-increasing in t at "
            f"E={emotion_intensity}, lambda={lambda_value}; got {scores}"
        )


# ---------------------------------------------------------------------------
# Closed-form check: the code must match paper Eq. 2 exactly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "elapsed_turns,emotion_intensity,lambda_value,tau_base",
    [
        (10, 0.0, 1.0, 100.0),
        (50, 0.5, 1.0, 100.0),
        (100, 1.0, 2.0, 200.0),
        (5, 0.3, 0.5, 50.0),
    ],
)
def test_formula_matches_paper_eq2(
    elapsed_turns, emotion_intensity, lambda_value, tau_base
):
    """R_affective(m, t) = exp(-t / (tau_base * (1 + lambda * E)))."""
    scorer = AffectiveRetentionScorer(
        RetentionConfig(lambda_value=lambda_value, tau_base=tau_base)
    )
    got = scorer.compute(elapsed_turns=elapsed_turns, emotion_intensity=emotion_intensity)
    expected = math.exp(
        -elapsed_turns / (tau_base * (1.0 + lambda_value * emotion_intensity))
    )
    assert got == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# get_decay_curve: the returned curves must preserve invariant 1 pointwise.
# ---------------------------------------------------------------------------


def test_decay_curves_preserve_emotion_monotonicity():
    scorer = AffectiveRetentionScorer(RetentionConfig(lambda_value=1.0, tau_base=100.0))
    curves = scorer.get_decay_curve(max_turns=200, emotion_levels=[0.0, 0.5, 1.0])
    series_low = curves["E=0.0"]
    series_mid = curves["E=0.5"]
    series_high = curves["E=1.0"]
    # At every t (including t=0), higher E must give no-lower retention.
    for t in range(len(series_low)):
        assert series_mid[t] + 1e-12 >= series_low[t], f"E=0.5 < E=0.0 at t={t}"
        assert series_high[t] + 1e-12 >= series_mid[t], f"E=1.0 < E=0.5 at t={t}"


# ---------------------------------------------------------------------------
# Deprecated field: tau_salience should raise DeprecationWarning but not affect
# the returned score (it no longer participates in the formula).
# ---------------------------------------------------------------------------


def test_tau_salience_does_not_change_score():
    """Changing tau_salience must NOT change retention output, per the config docstring."""
    with pytest.warns(DeprecationWarning):
        cfg_small = RetentionConfig(lambda_value=1.0, tau_base=100.0, tau_salience=10.0)
    with pytest.warns(DeprecationWarning):
        cfg_large = RetentionConfig(lambda_value=1.0, tau_base=100.0, tau_salience=500.0)

    s_small = AffectiveRetentionScorer(cfg_small).compute(elapsed_turns=50, emotion_intensity=0.8)
    s_large = AffectiveRetentionScorer(cfg_large).compute(elapsed_turns=50, emotion_intensity=0.8)
    assert s_small == pytest.approx(s_large)
