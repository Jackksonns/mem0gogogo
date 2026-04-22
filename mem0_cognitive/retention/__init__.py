"""
Affective Retention Score Module

Implements the emotion-weighted forgetting curve described in Section 3.2 of
our ACL 2026 paper: S_eff = S_base · (1 + λ·E)
"""

from mem0_cognitive.retention.scorer import AffectiveRetentionScorer
from mem0_cognitive.retention.configs import RetentionConfig

__all__ = ["AffectiveRetentionScorer", "RetentionConfig"]
