"""LoCoMo adapter for the cognitive-memory evaluation (interface-only).

This module is the reference public API that ``scripts/run_ablation.py`` will
use to consume the LoCoMo dataset. Every method raises
``NotImplementedError``; the metric-definition ambiguity flagged by the
Stage-6 reviewer feedback ("answer accuracy vs retrieval faithfulness vs
attribution correctness") must be resolved in the implementing PR and
pinned in this docstring before any LoCoMo number is reported.

The LoCoMo data file itself lives under
``evaluation_mem0_original/dataset/locomo10.json`` (download instructions are
in ``evaluation_mem0_original/README.md``). The implementing PR must
document the exact path it reads from and add a clear ``FileNotFoundError``
message when the dataset has not been downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class LoCoMoSession:
    """One LoCoMo dialogue session adapted for our retention metric.

    Attributes:
        session_id: Upstream LoCoMo session identifier.
        transcript: Ordered list of ``(speaker, utterance)`` pairs.
        queries: Each query is ``(query_text, gold_memory_ids)`` where
            ``gold_memory_ids`` are indices into ``transcript`` that the
            upstream LoCoMo annotation attributes to the gold answer.
    """

    session_id: str
    transcript: tuple[tuple[str, str], ...]
    queries: tuple[tuple[str, tuple[int, ...]], ...]


class LoCoMoAdapter:
    """Adapts LoCoMo's answer-attribution labels to our retention metric."""

    def __init__(self, dataset_path: str | Path) -> None:
        self.dataset_path = Path(dataset_path)

    def load(self) -> Iterable[LoCoMoSession]:
        """Load and adapt all LoCoMo sessions.

        Raises:
            NotImplementedError: Always, in this skeleton release.
        """
        raise NotImplementedError(
            "LoCoMoAdapter.load is not implemented yet. See "
            "evaluation_cognitive/README.md, bullet 'LoCoMo adapter'."
        )

    def describe_metric(self) -> str:
        """Return the exact metric definition used when scoring LoCoMo.

        The implementing PR must return a short string that specifies
        whether the metric is (a) answer-accuracy against the LoCoMo gold
        answer, (b) retrieval faithfulness against the gold turn set, or
        (c) attribution correctness (gold turn ids must appear in the
        top-k retrieved). Mixing the three silently is the concrete bug
        that Stage 6's reviewer flagged.

        Raises:
            NotImplementedError: Always, in this skeleton release.
        """
        raise NotImplementedError(
            "LoCoMoAdapter.describe_metric is not implemented yet. This "
            "method must return a single unambiguous metric definition "
            "(answer-accuracy vs retrieval-faithfulness vs "
            "attribution-correctness) before any LoCoMo number is "
            "reported in the paper."
        )
