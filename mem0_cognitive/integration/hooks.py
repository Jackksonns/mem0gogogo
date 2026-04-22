"""Thin integration surface that wires cognitive modules into host memory.

The :class:`CognitiveHooks` object owns one instance each of the three
cognitive modules (emotion extraction, affective retention scoring,
sleep consolidation) and exposes three entry points that the host SDK
calls at well-defined moments in the memory lifecycle:

- ``enrich_memory_metadata(content, metadata) -> None``
    Called from ``Memory._create_memory`` **before** the record is
    persisted to the vector store. Injects ``emotion_intensity``,
    ``emotion_valence``, and ``emotion_method`` into ``metadata``
    so downstream scoring + consolidation can read them back.

- ``apply_retention_reranking(candidates) -> None``
    Called from ``Memory._search_vector_store`` **after** the semantic
    / keyword / entity signals have populated each candidate. Mutates
    every candidate dict in-place to carry a ``retention_score``
    (the paper's ``R_affective``) and an
    ``affective_composite_score`` that callers can rank on.

- ``run_sleep_cycle(memory_store_adapter) -> stats``
    Called from ``Memory.run_sleep_consolidation``, which is a new
    explicit entry point; the hooks themselves never schedule work.

The integration is **opt-in**. :func:`CognitiveHooks.from_config`
returns ``None`` if no opt-in flag is present, and the host SDK
tolerates ``None`` on every call site. By design this means a user
who has not set ``MEM0_COGNITIVE_ENABLED=1`` sees exactly the
pre-Stage-5 behaviour.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

from mem0_cognitive.consolidation.configs import ConsolidationConfig
from mem0_cognitive.consolidation.engine import SleepConsolidator, Summarizer
from mem0_cognitive.emotion.analyzer import EmotionAnalyzer
from mem0_cognitive.emotion.configs import EmotionConfig
from mem0_cognitive.retention.configs import RetentionConfig
from mem0_cognitive.retention.scorer import AffectiveRetentionScorer

logger = logging.getLogger(__name__)


_TRUTHY = {"1", "true", "yes", "on"}


def _parse_iso_timestamp(value: Any) -> Optional[datetime]:
    """Best-effort parse of a memory's ``created_at`` payload field.

    The host SDK stores ``created_at`` as an ISO-8601 string with a
    ``Z`` or ``+00:00`` suffix; older code paths may also store a
    naive ``datetime``. Return ``None`` on anything else so the
    retention scorer's own default (``elapsed_turns = 1``) kicks in.
    """

    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # Python 3.10 fromisoformat does not accept trailing 'Z'.
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return None
    return None


@dataclass
class CognitiveHooksConfig:
    """Knobs for :class:`CognitiveHooks`.

    All fields default to safe, offline-friendly values so the object
    can be constructed and exercised in unit tests without any network
    access or LLM credentials.
    """

    enable_emotion_enrichment: bool = True
    enable_retention_reranking: bool = True
    enable_sleep_consolidation: bool = True

    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)

    # Weight used to blend retention into the final ranking score
    # without swamping the existing semantic/BM25/entity signals.
    retention_weight: float = 0.25

    # Optional summariser for sleep consolidation. ``None`` -> use the
    # deterministic template fallback in
    # :func:`mem0_cognitive.consolidation.engine._default_summarizer`.
    summarizer: Optional[Summarizer] = None


class CognitiveHooks:
    """Owns the three cognitive modules and wires them into host memory.

    Parameters
    ----------
    config:
        :class:`CognitiveHooksConfig` controlling which hooks run.

    Notes
    -----
    The class does **not** import anything from :mod:`mem0`; it only
    operates on plain dicts (candidate records, metadata) and on a
    duck-typed ``memory_store`` object for consolidation. This keeps
    the dependency direction one-way: ``mem0`` imports
    ``mem0_cognitive``, never the reverse.
    """

    def __init__(self, config: Optional[CognitiveHooksConfig] = None):
        self.config = config or CognitiveHooksConfig()

        self.emotion_analyzer: Optional[EmotionAnalyzer] = (
            EmotionAnalyzer(self.config.emotion)
            if self.config.enable_emotion_enrichment
            else None
        )
        self.retention_scorer: Optional[AffectiveRetentionScorer] = (
            AffectiveRetentionScorer(self.config.retention)
            if self.config.enable_retention_reranking
            else None
        )
        # Consolidator is instantiated on-demand in run_sleep_cycle
        # because it requires a memory_store adapter.
        self._summarizer = self.config.summarizer

        logger.info(
            "CognitiveHooks initialized: emotion=%s retention=%s sleep=%s",
            bool(self.emotion_analyzer),
            bool(self.retention_scorer),
            self.config.enable_sleep_consolidation,
        )

    # ---------------------------------------------------------- factory

    @classmethod
    def from_config(
        cls,
        memory_config: Any = None,
        *,
        env: Optional[Dict[str, str]] = None,
    ) -> Optional["CognitiveHooks"]:
        """Build a hooks instance iff cognitive features are opted into.

        The enable flag is sourced, in order, from:

        1. ``memory_config.cognitive`` if the host ``MemoryConfig``
           declares such an attribute (None-safe).
        2. The ``MEM0_COGNITIVE_ENABLED`` environment variable
           (``"1"``, ``"true"``, ``"yes"``, ``"on"`` all accepted).

        Returns ``None`` when neither opt-in is present; callers use
        the return value as a truthy guard.
        """

        cognitive_attr = getattr(memory_config, "cognitive", None) if memory_config else None
        if cognitive_attr is not None:
            if isinstance(cognitive_attr, CognitiveHooksConfig):
                return cls(cognitive_attr)
            if isinstance(cognitive_attr, dict):
                return cls(CognitiveHooksConfig(**cognitive_attr))
            if cognitive_attr is True:
                return cls()

        env = env if env is not None else os.environ
        flag = env.get("MEM0_COGNITIVE_ENABLED", "").lower()
        if flag in _TRUTHY:
            return cls()

        return None

    # ---------------------------------------------------------- hooks

    def enrich_memory_metadata(
        self, content: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Inject emotion fields into ``metadata`` before a write.

        Mutates ``metadata`` in place (for the caller's convenience)
        and also returns it. If emotion enrichment is disabled, or
        the analyser raises, the metadata is returned unchanged and
        ``emotion_method`` is set to ``'none'``.
        """

        if not self.emotion_analyzer or not isinstance(content, str) or not content:
            metadata.setdefault("emotion_intensity", 0.0)
            metadata.setdefault("emotion_valence", "neutral")
            metadata.setdefault("emotion_method", "none")
            return metadata

        try:
            result = self.emotion_analyzer.extract(content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("EmotionAnalyzer.extract failed: %s", exc)
            metadata.setdefault("emotion_intensity", 0.0)
            metadata.setdefault("emotion_valence", "neutral")
            metadata.setdefault("emotion_method", "none")
            return metadata

        intensity = float(result.get("intensity", 0.0))
        metadata["emotion_intensity"] = max(0.0, min(1.0, intensity))
        metadata["emotion_valence"] = str(result.get("valence", "neutral"))
        metadata["emotion_method"] = str(result.get("method", "unknown"))
        return metadata

    def apply_retention_reranking(
        self,
        candidates: Iterable[Dict[str, Any]],
        *,
        current_time: Optional[datetime] = None,
    ) -> None:
        """Populate ``retention_score`` + ``affective_composite_score``.

        Each candidate is expected to carry at least a ``score`` (the
        semantic/BM25/entity composite already computed by the host
        ranking pipeline) and a ``payload`` dict possibly containing
        ``emotion_intensity`` and ``created_at``. Missing fields are
        handled safely.
        """

        if self.retention_scorer is None:
            return

        weight = float(self.config.retention_weight)
        weight = max(0.0, min(1.0, weight))

        for cand in candidates:
            payload = cand.get("payload") or {}
            emotion_intensity = float(payload.get("emotion_intensity", 0.0) or 0.0)
            created_at = _parse_iso_timestamp(payload.get("created_at"))

            # The retention scorer subtracts timestamps directly, so we
            # must hand it two values of the same tz-awareness. If the
            # payload timestamp is tz-aware we use an aware ``now``; if
            # it is naive we strip tzinfo from any caller-supplied
            # ``current_time`` so the subtraction still works.
            effective_now = current_time
            if created_at is not None and created_at.tzinfo is not None:
                if effective_now is None:
                    effective_now = datetime.now(created_at.tzinfo)
                elif effective_now.tzinfo is None:
                    effective_now = effective_now.replace(tzinfo=created_at.tzinfo)
            elif created_at is not None and created_at.tzinfo is None:
                if effective_now is not None and effective_now.tzinfo is not None:
                    effective_now = effective_now.replace(tzinfo=None)

            try:
                retention = self.retention_scorer.compute(
                    emotion_intensity=emotion_intensity,
                    created_at=created_at,
                    current_time=effective_now,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("retention_scorer.compute failed: %s", exc)
                retention = 0.0

            cand["retention_score"] = retention
            base = float(cand.get("score", 0.0) or 0.0)
            cand["affective_composite_score"] = (
                (1.0 - weight) * base + weight * retention
            )

    def run_sleep_cycle(
        self,
        memory_store_adapter: Any,
        *,
        config_override: Optional[ConsolidationConfig] = None,
        summarizer: Optional[Summarizer] = None,
    ) -> Dict[str, Any]:
        """Run one offline consolidation cycle against the given store.

        ``memory_store_adapter`` must implement the memory-store
        protocol documented in
        :mod:`mem0_cognitive.consolidation.engine`: ``get_all()``,
        ``add(content, metadata) -> id``, and ``delete(id)``.
        """

        if not self.config.enable_sleep_consolidation:
            return {
                "retrieved": 0,
                "clusters_formed": 0,
                "consolidated": 0,
                "pruned": 0,
                "duration_seconds": 0.0,
                "skipped_reason": "enable_sleep_consolidation=False",
            }

        cfg = config_override or self.config.consolidation
        consolidator = SleepConsolidator(
            memory_store_adapter,
            cfg,
            summarizer=summarizer or self._summarizer,
        )

        # SleepConsolidator.run_consolidation_cycle is an async method
        # but its body has no actual awaitables; we execute it via a
        # temporary event loop so the wire-in callers can stay
        # synchronous. If the caller is already on a running loop we
        # return the coroutine for them to ``await``.
        import asyncio

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is not None:
            return consolidator.run_consolidation_cycle()  # type: ignore[return-value]

        return asyncio.run(consolidator.run_consolidation_cycle())

    # ---------------------------------------------------------- utils

    @staticmethod
    def guard_sort_key(retention_weight: float) -> Callable[[Dict[str, Any]], float]:
        """Return a ranking key that prefers ``affective_composite_score``.

        Falls back to ``score`` when the composite field is absent.
        Exposed so host code can sort candidates without having to
        reproduce the blending formula.
        """

        def _key(c: Dict[str, Any]) -> float:
            return float(
                c.get("affective_composite_score", c.get("score", 0.0)) or 0.0
            )

        del retention_weight  # currently unused, kept for forward-compat
        return _key


def rerank_candidates_by_affective_composite(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Stable-sort candidates by ``affective_composite_score`` descending.

    Convenience wrapper for callers that want an explicit return value
    rather than mutating in place. Sort is stable so candidates with
    identical composite scores preserve their input order.
    """

    return sorted(
        candidates,
        key=lambda c: float(
            c.get("affective_composite_score", c.get("score", 0.0)) or 0.0
        ),
        reverse=True,
    )
