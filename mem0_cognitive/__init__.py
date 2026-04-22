"""
Mem0-Cognitive: Biologically-Inspired Memory Enhancement Module

This module extends the official Mem0 project with cognitive psychology mechanisms
for ACL 2026 submission. It provides emotion-aware memory management, sleep
consolidation, and meta-cognitive learning capabilities.

Author: Hongyi Zhou
Affiliation: [Your University/Institution]
Paper: ACL 2026 Submission
"""

from mem0_cognitive.emotion.analyzer import EmotionAnalyzer
from mem0_cognitive.retention.scorer import AffectiveRetentionScorer
from mem0_cognitive.consolidation.engine import SleepConsolidator
from mem0_cognitive.meta_learner.optimizer import MetaCognitiveOptimizer
from mem0_cognitive.utils.helpers import (
    extract_emotion,
    compute_retention_score,
    run_consolidation_cycle,
)

__version__ = "1.0.0-acl2026"
__author__ = "Hongyi Zhou"
__all__ = [
    "EmotionAnalyzer",
    "AffectiveRetentionScorer",
    "SleepConsolidator",
    "MetaCognitiveOptimizer",
    "extract_emotion",
    "compute_retention_score",
    "run_consolidation_cycle",
]
