"""CognitiveBench generators.

Status: interface-only. Every public entry point in this package currently
raises ``NotImplementedError`` and the seed manifest
(``configs/cognitivebench_seeds.yaml``) still has ``TODO`` values. See
``evaluation_cognitive/README.md`` for the full reproducibility checklist.
"""

from evaluation_cognitive.generators.cognitivebench import (
    CognitiveBenchGenerator,
    GeneratedDialogue,
)

__all__ = ["CognitiveBenchGenerator", "GeneratedDialogue"]
