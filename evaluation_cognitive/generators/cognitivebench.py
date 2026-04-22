"""CognitiveBench generator (interface-only skeleton).

This module documents the public surface that ``scripts/run_ablation.py``
will call into. Every method currently raises ``NotImplementedError`` — this
is *intentional*: Stage 6 of the remediation plan ships the directory
layout, config schema, and adapter interfaces first, and defers data
generation to a follow-up PR. Do not paper over the ``NotImplementedError``
with a silently-broken mock; implement the real generator or file a
follow-up PR.

The end-to-end reproducibility checklist is in
``evaluation_cognitive/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DialogueTurn:
    """One conversational turn produced by the generator.

    Attributes:
        speaker: Either ``"user"`` or ``"assistant"``.
        text: Utterance text.
        emotion_label: Optional categorical emotion label (or ``None``).
        is_noise: ``True`` if this turn is a distractor / noise turn that
            should not be retrieved by any gold query.
    """

    speaker: str
    text: str
    emotion_label: str | None = None
    is_noise: bool = False


@dataclass(frozen=True)
class DialogueQuery:
    """A query attached to a generated dialogue.

    Attributes:
        query_text: Natural-language query issued to memory.
        gold_turn_ids: Indices into the dialogue's ``turns`` list that are
            the ground-truth supporting turns.
        taxonomy_class: One of the classes declared in
            ``configs/cognitivebench_seeds.yaml#query_taxonomy``.
    """

    query_text: str
    gold_turn_ids: tuple[int, ...]
    taxonomy_class: str


@dataclass(frozen=True)
class GeneratedDialogue:
    """A fully generated dialogue plus its attached queries.

    Attributes:
        dialogue_id: Stable, seed-derived identifier.
        turns: Ordered list of turns.
        queries: Queries attached to this dialogue.
        difficulty: One of ``{"easy", "medium", "hard"}``.
        metadata: Free-form generator-specific metadata (seed, version, …).
    """

    dialogue_id: str
    turns: tuple[DialogueTurn, ...]
    queries: tuple[DialogueQuery, ...]
    difficulty: str
    metadata: dict = field(default_factory=dict)


class CognitiveBenchGenerator:
    """Generates CognitiveBench dialogues from a frozen seed manifest.

    This class is the reference public API for the CognitiveBench dataset.
    It is **not** implemented in this PR — every method raises
    ``NotImplementedError``. When implementing, follow the contract in
    ``configs/cognitivebench_seeds.yaml`` and the leak-check rule described
    there.
    """

    def __init__(self, seed_manifest_path: str | Path) -> None:
        self.seed_manifest_path = Path(seed_manifest_path)

    def generate(self) -> Iterable[GeneratedDialogue]:
        """Yield all dialogues declared by the seed manifest.

        Raises:
            NotImplementedError: Always, in this skeleton release.
        """
        raise NotImplementedError(
            "CognitiveBenchGenerator.generate is not implemented yet. "
            "See evaluation_cognitive/README.md for the reproducibility "
            "checklist and the seed-manifest freeze requirement."
        )

    def verify_no_leakage(
        self, dialogues: Iterable[GeneratedDialogue]
    ) -> None:
        """Assert that no gold answer leaks into the distractor stream.

        Raises:
            NotImplementedError: Always, in this skeleton release.
        """
        raise NotImplementedError(
            "CognitiveBenchGenerator.verify_no_leakage is not implemented "
            "yet. The seed manifest declares "
            "leak_check.max_allowed_leaks_per_dialogue = 0; this method is "
            "where that invariant must be enforced."
        )
