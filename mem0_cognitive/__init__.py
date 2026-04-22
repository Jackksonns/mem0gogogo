"""
Mem0-Cognitive: Research extension of Mem0 with cognitive memory mechanisms.

This subpackage extends the Mem0 project (https://github.com/mem0ai/mem0) with
three research prototype modules:

    - Emotion-weighted retention scoring (affective Ebbinghaus decay)
    - Offline sleep-style consolidation of clustered memories
    - Adaptive heuristic tuning of per-user retention parameters

Intended use: academic / research experimentation only. For production memory,
use the underlying ``mem0`` package directly.
"""

from mem0_cognitive.consolidation.engine import SleepConsolidator
from mem0_cognitive.emotion.analyzer import EmotionAnalyzer
from mem0_cognitive.meta_learner.optimizer import MetaCognitiveOptimizer
from mem0_cognitive.retention.scorer import AffectiveRetentionScorer
from mem0_cognitive.utils.helpers import (
    compute_retention_score,
    extract_emotion,
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
