"""
Consolidation Engine Module

Implements sleep-based memory consolidation as described in Section 3.3 of
our ACL 2026 paper, mimicking hippocampus-to-neocortex transfer in human memory.
"""

from mem0_cognitive.consolidation.engine import SleepConsolidator
from mem0_cognitive.consolidation.configs import ConsolidationConfig

__all__ = ["SleepConsolidator", "ConsolidationConfig"]
