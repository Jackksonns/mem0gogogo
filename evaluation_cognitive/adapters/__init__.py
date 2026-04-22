"""Dataset adapters for the cognitive-memory evaluation.

Status: interface-only. Every public entry point raises
``NotImplementedError``. See ``evaluation_cognitive/README.md``.
"""

from evaluation_cognitive.adapters.locomo import LoCoMoAdapter

__all__ = ["LoCoMoAdapter"]
