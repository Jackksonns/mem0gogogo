"""Configuration dataclass for :class:`AffectiveRetentionScorer`.

The retention formula implemented in
:mod:`mem0_cognitive.retention.scorer` is the paper's Equation 2,

.. math::

    R_{\\text{affective}}(m_i, \\Delta t)
        = \\exp\\left(-\\frac{\\Delta t}{\\tau_{\\text{base}} \\cdot (1 + \\lambda E_i)}\\right),

which has two free parameters: ``tau_base`` (the neutral-memory time
constant, i.e.\\ ``S_base``) and ``lambda_value`` (the emotional-inertia
coefficient ``lambda``). Both are exposed below.

The ``tau_salience`` field is retained for backward compatibility with
pre-existing ``MetaCognitiveOptimizer`` search spaces but is *not* used
by the retention formula: in earlier drafts it was (incorrectly) used as
a second time constant, which flipped the direction of the emotion
weighting. Supplying a non-default value now emits a
``DeprecationWarning``; its only planned future use is as the salience
**threshold** of the Salience Gate (paper Eq. 3), which is a separate
concept from a time constant.
"""

import warnings
from dataclasses import dataclass


@dataclass
class RetentionConfig:
    """Parameters for :class:`AffectiveRetentionScorer`.

    Attributes:
        lambda_value: Emotional-inertia coefficient ``lambda in [0, 2]``.
            Larger values make emotional memories decay more slowly. Must
            be non-negative so that the retention formula is monotone in
            ``E``.
        tau_base: Baseline time constant ``S_base`` (in turns) for
            neutral (``E = 0``) memories. Must be strictly positive.
        tau_salience: **Deprecated, not used by the retention formula.**
            Kept as a declared field only so that legacy meta-optimizer
            configs that search over it do not fail at construction
            time. Will be repurposed as the paper's salience-gate
            threshold in a later stage.
        enable_emotion_weighting: If ``False``, ``lambda_value`` is
            effectively clamped to ``0`` and the score reduces to the
            classic Ebbinghaus curve.
        min_retention_threshold: Score below which
            ``AffectiveRetentionScorer.should_retain`` returns
            ``False``. Used by downstream pruning logic.
        time_unit: Informational only; documents the unit of
            ``elapsed_turns``. No code path branches on it today.
    """

    lambda_value: float = 1.0
    tau_base: float = 100.0  # turns
    tau_salience: float = 50.0  # turns (deprecated; see class docstring)
    enable_emotion_weighting: bool = True
    min_retention_threshold: float = 0.1

    time_unit: str = "turns"

    def __post_init__(self):
        if self.lambda_value < 0 or self.lambda_value > 2:
            raise ValueError("lambda_value must be in range [0, 2] as per the paper")
        if self.tau_base <= 0:
            raise ValueError("tau_base must be strictly positive")
        if self.tau_salience <= 0:
            raise ValueError("tau_salience must be strictly positive when supplied")
        if not 0 <= self.min_retention_threshold <= 1:
            raise ValueError("min_retention_threshold must be in [0, 1]")

        # The retention formula no longer consumes tau_salience. Warn
        # callers who set a non-default value so their expectation does
        # not silently diverge from the implemented math.
        if self.tau_salience != 50.0:
            warnings.warn(
                "RetentionConfig.tau_salience is deprecated and no longer "
                "participates in the affective-retention formula "
                "(see mem0_cognitive/retention/scorer.py). It is retained "
                "only for backward compatibility with legacy meta-optimizer "
                "search spaces and will be repurposed as the salience-gate "
                "threshold in a future stage.",
                DeprecationWarning,
                stacklevel=2,
            )
