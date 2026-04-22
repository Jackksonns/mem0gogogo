"""Integration shims between :mod:`mem0_cognitive` and the host ``mem0`` SDK.

The classes here are deliberately small and side-effect-free on their
own: they expose explicit entry points (``enrich_memory_metadata``,
``apply_retention_reranking``, ``run_sleep_cycle``) that the host
SDK opts into. Importing this module does **not** start any
background work.
"""

from mem0_cognitive.integration.hooks import CognitiveHooks, CognitiveHooksConfig

__all__ = ["CognitiveHooks", "CognitiveHooksConfig"]
