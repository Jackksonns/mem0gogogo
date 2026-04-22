"""Affective Retention Scorer: emotion-weighted Ebbinghaus decay.

This module implements a single, scalar retention score per memory item. The
formulation matches the paper's Equation 2 of Section 3.2:

.. math::

    R_{\\text{affective}}(m_i, \\Delta t)
        = \\exp\\left(-\\frac{\\Delta t}{S_{\\text{base}} \\cdot (1 + \\lambda E_i)}\\right)

The *effective* time constant is ``S_base * (1 + lambda * E)``. Because
``1 + lambda * E`` is monotonically non-decreasing in ``E`` for
``lambda >= 0`` and ``E in [0, 1]``, the effective time constant is
monotonically non-decreasing in ``E``; the exponential is therefore
monotonically non-decreasing in ``E``. In plain English: **high emotional
intensity slows decay**, as the paper and the README claim.

Historical note
---------------
An earlier implementation used ``tau_effective = tau_base - (tau_base -
tau_salience) * E`` together with a post-hoc multiplicative boost of
``(1 + lambda * E)``. With the historical defaults (``tau_base = 100``,
``tau_salience = 50``) the first term *decreased* ``tau_effective`` as
``E`` grew, giving the formula the **opposite** monotonicity from what the
paper claimed: high-emotion memories were decaying *faster*, not slower.
The multiplicative boost only partly masked the bug and made the overall
retention non-monotone in ``E``. The current file fixes that silent
scientific bug. See ``tests/cognitive/test_retention_direction.py`` for the
monotonicity tests that pin this behavior in place.

The ``RetentionConfig.tau_salience`` field is retained for backward
compatibility with existing ``MetaCognitiveOptimizer`` configurations that
searched over it, but it no longer participates in the retention formula.
A ``DeprecationWarning`` is emitted when a non-default value is supplied.
"""

import logging
import math
from datetime import datetime
from typing import Dict, Optional

from mem0_cognitive.retention.configs import RetentionConfig

logger = logging.getLogger(__name__)


class AffectiveRetentionScorer:
    """Compute an affective retention score in ``[0, 1]`` for a memory item.

    The retention score is the paper's ``R_affective(m, Δt)``: the probability
    that the memory is still active at time ``Δt`` given baseline strength
    ``tau_base`` and emotional intensity ``E`` modulated by ``lambda_value``.

    Example
    -------
    >>> scorer = AffectiveRetentionScorer(RetentionConfig(tau_base=100.0, lambda_value=1.0))
    >>> neutral = scorer.compute(elapsed_turns=50, emotion_intensity=0.0)
    >>> salient = scorer.compute(elapsed_turns=50, emotion_intensity=1.0)
    >>> salient > neutral
    True
    """

    def __init__(self, config: Optional[RetentionConfig] = None):
        self.config = config or RetentionConfig()
        logger.info(
            "AffectiveRetentionScorer initialized with lambda=%s, tau_base=%s",
            self.config.lambda_value,
            self.config.tau_base,
        )

    def compute(
        self,
        elapsed_turns: Optional[int] = None,
        emotion_intensity: float = 0.0,
        created_at: Optional[datetime] = None,
        current_time: Optional[datetime] = None,
    ) -> float:
        """Compute the effective retention score.

        Args:
            elapsed_turns: Number of dialogue turns since memory creation. If
                ``None``, ``created_at`` (optionally with ``current_time``) is
                used; if both are ``None``, defaults to ``1`` turn.
            emotion_intensity: Emotional intensity ``E in [0, 1]``. Values
                outside the range are clamped.
            created_at: Timestamp of memory creation (fallback for
                ``elapsed_turns``).
            current_time: Current timestamp (fallback; defaults to ``now``).

        Returns:
            ``R_affective in (0, 1]`` \u2014 the probability of retention at the
            given elapsed time. Monotonically non-increasing in
            ``elapsed_turns`` and monotonically non-decreasing in
            ``emotion_intensity`` for ``lambda_value >= 0``.
        """
        if elapsed_turns is None and created_at is not None:
            if current_time is None:
                current_time = datetime.now()
            elapsed_minutes = (current_time - created_at).total_seconds() / 60.0
            elapsed_turns = max(1, int(elapsed_minutes))
        elif elapsed_turns is None:
            elapsed_turns = 1

        # Clamp E into the documented domain so the monotonicity guarantee
        # holds even on accidentally out-of-range inputs.
        E = max(0.0, min(1.0, emotion_intensity))

        if self.config.enable_emotion_weighting:
            lam = self.config.lambda_value
        else:
            lam = 0.0

        # Paper Eq. 2: effective time constant is S_base scaled by the
        # emotional-inertia factor. With lam >= 0 and E in [0, 1] this is
        # monotonically non-decreasing in E, so larger E -> larger time
        # constant -> slower decay, matching the paper's claim.
        tau_effective = self.config.tau_base * (1.0 + lam * E)
        tau_effective = max(1e-6, tau_effective)

        retention = math.exp(-elapsed_turns / tau_effective)

        logger.debug(
            "Retention score: t=%s E=%.3f lambda=%.3f tau_eff=%.3f R=%.6f",
            elapsed_turns,
            E,
            lam,
            tau_effective,
            retention,
        )
        return retention

    def compute_batch(self, memory_items: list) -> list:
        """Compute retention scores for multiple memory items.

        Args:
            memory_items: List of dicts with keys ``elapsed_turns`` /
                ``created_at`` and optionally ``emotion_intensity``.

        Returns:
            List of ``(memory_item, score)`` tuples, sorted by score
            descending.
        """
        scored_items = []
        for item in memory_items:
            score = self.compute(
                elapsed_turns=item.get("elapsed_turns"),
                emotion_intensity=item.get("emotion_intensity", 0.0),
                created_at=item.get("created_at"),
            )
            scored_items.append((item, score))

        scored_items.sort(key=lambda x: x[1], reverse=True)
        return scored_items

    def should_retain(self, score: float) -> bool:
        """Return ``True`` iff ``score >= config.min_retention_threshold``.

        Memories below the threshold are candidates for removal during the
        sleep-consolidation pass described in the paper's Section 3.3.
        """
        return score >= self.config.min_retention_threshold

    def get_decay_curve(
        self, max_turns: int = 500, emotion_levels: Optional[list] = None
    ) -> Dict[str, list]:
        """Generate retention-vs-time curves at several emotion levels.

        Useful for the retention-curve figure: at any fixed ``t > 0`` the
        returned curves must satisfy ``curves[E_high][t] >= curves[E_low][t]``
        whenever ``E_high >= E_low`` (for ``lambda_value >= 0``).
        """
        if emotion_levels is None:
            emotion_levels = [0.0, 0.5, 1.0]

        curves: Dict[str, list] = {}
        for E in emotion_levels:
            curves[f"E={E}"] = [
                self.compute(elapsed_turns=t, emotion_intensity=E) for t in range(max_turns + 1)
            ]
        return curves
