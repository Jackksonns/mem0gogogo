"""Sleep Consolidation Engine: offline memory reconsolidation.

Implements the sleep-based consolidation mechanism from Section 3.3 of the
companion paper. The engine performs offline memory processing inspired by
hippocampus-to-neocortex transfer during human sleep, clustering similar
memories and generalizing specifics into abstract knowledge.

Phases:

1. **Reactivation** (``_retrieve_consolidation_candidates``): retrieve
   memories that are (a) old enough to be outside the current working
   window and (b) below the retention-score cutoff.
2. **Clustering** (``_cluster_memories``): group candidates by cosine
   similarity on stored embeddings with a single-pass first-fit rule.
3. **Generalization + Writeback** (``_generalize_cluster``): for each
   non-singleton cluster, materialise a consolidated record
   (strategy-dependent), write it to the memory store via ``add``, and
   delete the source records. Singleton candidates are pruned.

Every writeback and delete is recorded in an audit log
(``self.audit_log``) so that experiments can defend the behaviour of
this phase by reading the log rather than inspecting the store after
the fact.

Memory-store protocol
---------------------

The consolidator consumes a duck-typed ``memory_store`` with the
following signature:

``get_all() -> List[Dict]``
    Return all memories as dicts with at least ``id``, ``content``,
    ``embedding``, ``created_at`` (``datetime``) and
    ``retention_score`` (``float``) fields. Additional fields are
    forwarded into consolidated records as ``source_metadata``.

``add(content: str, metadata: Dict) -> str``
    Insert a new consolidated memory and return its id.

``delete(memory_id: str) -> None``
    Remove a memory by id.

The in-process :class:`InMemoryStore` below is a fully-functional
reference implementation used by the unit tests in
``tests/cognitive/test_sleep_consolidator.py``.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from mem0_cognitive.consolidation.configs import ConsolidationConfig

logger = logging.getLogger(__name__)


Summarizer = Callable[[List[str]], str]
"""Deterministic or LLM-backed summariser interface.

A summariser takes the list of source-memory contents for a cluster
and returns a single consolidated string.
"""


@dataclass
class ConsolidationAuditEntry:
    """One record of a consolidation event (materialise + delete)."""

    timestamp: datetime
    strategy: str
    cluster_size: int
    source_ids: List[str]
    consolidated_id: Optional[str]
    summary_preview: str = ""
    fallback_used: Optional[str] = None
    """If ``summarize`` fell back to another strategy, the name of the
    fallback strategy used. ``None`` means the configured strategy
    completed without falling back."""


@dataclass
class InMemoryStore:
    """Minimal in-process store implementing the protocol above.

    Provided for tests and for the bundled demo so that the
    consolidation cycle can run end-to-end without a vector database.
    Not intended for production use.
    """

    _records: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def get_all(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in self._records.values()]

    def add(self, content: str, metadata: Dict[str, Any]) -> str:
        mid = metadata.get("id") or str(uuid.uuid4())
        record = dict(metadata)
        record["id"] = mid
        record["content"] = content
        record.setdefault("created_at", datetime.now())
        record.setdefault("retention_score", 1.0)
        self._records[mid] = record
        return mid

    def delete(self, memory_id: str) -> None:
        self._records.pop(memory_id, None)

    def __len__(self) -> int:
        return len(self._records)


def _default_summarizer(contents: List[str]) -> str:
    """Deterministic, dependency-free fallback summariser.

    Used when no LLM-backed summariser is provided. The output is
    prefixed so it is unambiguously identifiable as a consolidation
    artifact rather than a real generated summary. Truncates to
    keep the result compact.
    """

    preview = " | ".join(contents)
    if len(preview) > 480:
        preview = preview[:477] + "..."
    return f"[consolidated x{len(contents)}] {preview}"


class SleepConsolidator:
    """Offline, idempotent consolidation of clustered memories.

    As described in paper Section 3.3, this consolidator mimics
    biological sleep consolidation in three phases: reactivation,
    clustering, and generalization (schema induction).

    Parameters
    ----------
    memory_store:
        A duck-typed store implementing ``get_all``, ``add``, and
        ``delete`` (see module docstring for the protocol).
    config:
        Consolidation configuration. Defaults to
        :class:`ConsolidationConfig`.
    summarizer:
        Optional callable mapping a list of source contents to a
        consolidated string. If ``None`` a deterministic template
        summariser is used. A real LLM-backed summariser can be
        supplied by passing e.g.::

            def _llm_summary(contents):
                resp = client.chat.completions.create(...)
                return resp.choices[0].message.content

            SleepConsolidator(store, summarizer=_llm_summary)

    Notes
    -----
    Every cluster materialisation is recorded in ``self.audit_log``
    as a :class:`ConsolidationAuditEntry`, so callers (and tests)
    can verify exactly what was written and which source ids were
    deleted.
    """

    def __init__(
        self,
        memory_store: Any,
        config: Optional[ConsolidationConfig] = None,
        summarizer: Optional[Summarizer] = None,
    ) -> None:
        self.memory_store = memory_store
        self.config = config or ConsolidationConfig()
        self.summarizer: Summarizer = summarizer or _default_summarizer
        self._last_consolidation_time: Optional[datetime] = None
        self.audit_log: List[ConsolidationAuditEntry] = []

        logger.info(
            "SleepConsolidator initialized with strategy=%s, interval=%dh, "
            "summarizer=%s",
            self.config.generalization_strategy,
            self.config.consolidation_interval_hours,
            "custom" if summarizer is not None else "default_template",
        )

    async def run_consolidation_cycle(self) -> Dict[str, Any]:
        """Execute a full consolidation cycle.

        Returns
        -------
        dict
            Statistics:

            - ``retrieved``: number of candidate memories.
            - ``clusters_formed``: number of non-singleton clusters.
            - ``consolidated``: number of source memories absorbed
              into consolidated records.
            - ``pruned``: number of memories deleted (singletons
              below the cutoff plus any fallback deletions).
            - ``duration_seconds``: wall-clock time for the cycle.
        """

        start_time = datetime.now()
        stats: Dict[str, Any] = {
            "retrieved": 0,
            "clusters_formed": 0,
            "consolidated": 0,
            "pruned": 0,
            "duration_seconds": 0.0,
        }

        if not self.config.enable_consolidation:
            logger.info("Consolidation disabled in config")
            return stats

        logger.info("Phase 1: Retrieving memories for consolidation...")
        memories = self._retrieve_consolidation_candidates()
        stats["retrieved"] = len(memories)

        if len(memories) < self.config.min_memories_for_consolidation:
            logger.info(
                "Only %d candidate memories found; skipping consolidation",
                len(memories),
            )
            stats["duration_seconds"] = (datetime.now() - start_time).total_seconds()
            return stats

        logger.info("Phase 2: Clustering memories by similarity...")
        clusters, unclustered = self._cluster_memories(memories)
        stats["clusters_formed"] = len(clusters)

        logger.info(
            "Phase 3: Generalising %d clusters and pruning %d singletons...",
            len(clusters),
            len(unclustered),
        )
        for cluster_members in clusters.values():
            result = await self._generalize_cluster(cluster_members)
            stats["consolidated"] += result.get("merged", 0)
            stats["pruned"] += result.get("pruned", 0)

        # Singleton candidates could not be consolidated with anything
        # else. They are still below the retention cutoff, so prune them
        # directly. This is a separate accounting path from generalise.
        stats["pruned"] += self._prune_low_retention(unclustered)

        self._last_consolidation_time = datetime.now()
        stats["duration_seconds"] = (datetime.now() - start_time).total_seconds()

        logger.info(
            "Consolidation cycle complete: retrieved=%d clusters=%d "
            "consolidated=%d pruned=%d in %.1fs",
            stats["retrieved"],
            stats["clusters_formed"],
            stats["consolidated"],
            stats["pruned"],
            stats["duration_seconds"],
        )

        return stats

    # ------------------------------------------------------------------ phase 1

    def _retrieve_consolidation_candidates(self) -> List[Dict[str, Any]]:
        """Select memories eligible for consolidation.

        A memory is a candidate iff its ``created_at`` is older than
        ``consolidation_interval_hours`` and its ``retention_score`` is
        below ``retention_score_cutoff``. Memories missing either field
        are assigned safe defaults (``retention_score`` = 0.5).
        """

        all_memories = self.memory_store.get_all()
        cutoff_time = datetime.now() - timedelta(
            hours=self.config.consolidation_interval_hours
        )

        candidates: List[Dict[str, Any]] = []
        for memory in all_memories:
            created_at = memory.get("created_at")
            if created_at is None or created_at >= cutoff_time:
                continue
            retention_score = memory.get("retention_score", 0.5)
            if retention_score < self.config.retention_score_cutoff:
                candidates.append(memory)

        logger.debug(
            "Found %d consolidation candidates out of %d total",
            len(candidates),
            len(all_memories),
        )
        return candidates

    # ------------------------------------------------------------------ phase 2

    def _cluster_memories(
        self, memories: List[Dict[str, Any]]
    ) -> "tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]":
        """First-fit single-pass clustering by cosine similarity.

        Returns a pair ``(non_singleton_clusters, unclustered_singletons)``
        so the caller can route each set to the right phase-3 sink
        without re-inspecting the cluster sizes.
        """

        if not memories:
            return {}, []

        clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for i, memory in enumerate(memories):
            assigned = False
            for cluster_id, cluster_members in clusters.items():
                similarities = [
                    self._compute_similarity(memory, member)
                    for member in cluster_members
                ]
                if not similarities:
                    continue
                avg_similarity = sum(similarities) / len(similarities)
                if avg_similarity >= self.config.clustering_threshold:
                    if len(cluster_members) >= self.config.max_cluster_size:
                        # A full cluster cannot absorb more without
                        # forcing a summarisation split; leave the
                        # memory to form its own cluster and let the
                        # next cycle pick it up.
                        continue
                    clusters[cluster_id].append(memory)
                    assigned = True
                    break

            if not assigned:
                clusters[f"cluster_{i}"].append(memory)

        non_singleton: Dict[str, List[Dict[str, Any]]] = {}
        unclustered: List[Dict[str, Any]] = []
        for cid, members in clusters.items():
            if len(members) >= 2:
                non_singleton[cid] = members
            else:
                unclustered.extend(members)

        logger.debug(
            "Formed %d non-singleton clusters; %d singleton candidates",
            len(non_singleton),
            len(unclustered),
        )
        return non_singleton, unclustered

    def _compute_similarity(
        self, mem1: Dict[str, Any], mem2: Dict[str, Any]
    ) -> float:
        """Cosine similarity between two memory embeddings.

        Returns 0.0 if either memory is missing an embedding, both to
        keep the single-pass clusterer total and to avoid silently
        misclassifying memories that were never embedded.
        """

        emb1 = mem1.get("embedding")
        emb2 = mem2.get("embedding")
        if emb1 is None or emb2 is None:
            return 0.0

        arr1 = np.asarray(emb1, dtype=float)
        arr2 = np.asarray(emb2, dtype=float)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        return float(np.dot(arr1, arr2) / (norm1 * norm2))

    # ------------------------------------------------------------------ phase 3

    async def _generalize_cluster(
        self, cluster_memories: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Materialise one cluster into a consolidated record.

        Dispatches on ``config.generalization_strategy``:

        - ``keep_best``: retain the highest-retention source memory as
          the consolidated record; delete the rest.
        - ``average``: concatenate source contents with a header,
          compute the mean embedding, write a new record, delete all
          sources.
        - ``summarize``: call ``self.summarizer(contents)`` and write
          a new record whose content is the summary; delete all
          sources. On failure falls back to ``keep_best`` in a single
          non-recursive step (the ``fallback_used`` field of the
          audit entry records this).
        """

        if len(cluster_memories) < 2:
            return {"merged": 0, "pruned": 0}

        strategy = self.config.generalization_strategy

        if strategy == "keep_best":
            return self._apply_keep_best(cluster_memories, fallback_used=None)

        if strategy == "average":
            return self._apply_average(cluster_memories, fallback_used=None)

        if strategy == "summarize":
            try:
                return self._apply_summarize(cluster_memories)
            except Exception as exc:  # noqa: BLE001 - operator-visible fallback
                logger.error(
                    "LLM summarisation failed (%s); falling back to keep_best",
                    exc,
                )
                # Non-recursive fallback. If keep_best itself raises we
                # let the exception propagate so the caller sees it.
                return self._apply_keep_best(
                    cluster_memories, fallback_used="summarize_failed"
                )

        # Guarded at config construction, but keep a defensive path.
        return {"merged": 0, "pruned": 0}

    def _apply_keep_best(
        self,
        cluster_memories: List[Dict[str, Any]],
        fallback_used: Optional[str],
    ) -> Dict[str, int]:
        best = max(cluster_memories, key=lambda m: m.get("retention_score", 0.0))
        deleted_ids: List[str] = []
        for mem in cluster_memories:
            if mem["id"] == best["id"]:
                continue
            self.memory_store.delete(mem["id"])
            deleted_ids.append(mem["id"])

        self._record_audit(
            strategy="keep_best",
            cluster=cluster_memories,
            consolidated_id=best["id"],
            summary_preview=str(best.get("content", ""))[:100],
            fallback_used=fallback_used,
        )
        return {"merged": len(deleted_ids), "pruned": 0}

    def _apply_average(
        self,
        cluster_memories: List[Dict[str, Any]],
        fallback_used: Optional[str],
    ) -> Dict[str, int]:
        contents = [m.get("content", "") for m in cluster_memories]
        consolidated_content = (
            f"[consolidated x{len(contents)}] " + " | ".join(contents)
        )

        embeddings = [
            np.asarray(m["embedding"], dtype=float)
            for m in cluster_memories
            if m.get("embedding") is not None
        ]
        mean_embedding = (
            np.mean(np.stack(embeddings, axis=0), axis=0).tolist()
            if embeddings
            else None
        )

        consolidated_id = self.memory_store.add(
            consolidated_content,
            {
                "consolidation_strategy": "average",
                "source_ids": [m["id"] for m in cluster_memories],
                "embedding": mean_embedding,
                "retention_score": 1.0,
                "created_at": datetime.now(),
            },
        )

        for mem in cluster_memories:
            self.memory_store.delete(mem["id"])

        self._record_audit(
            strategy="average",
            cluster=cluster_memories,
            consolidated_id=consolidated_id,
            summary_preview=consolidated_content[:100],
            fallback_used=fallback_used,
        )
        return {"merged": len(cluster_memories), "pruned": 0}

    def _apply_summarize(
        self, cluster_memories: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        summary = self._llm_summarize(cluster_memories)

        embeddings = [
            np.asarray(m["embedding"], dtype=float)
            for m in cluster_memories
            if m.get("embedding") is not None
        ]
        mean_embedding = (
            np.mean(np.stack(embeddings, axis=0), axis=0).tolist()
            if embeddings
            else None
        )

        consolidated_id = self.memory_store.add(
            summary,
            {
                "consolidation_strategy": "summarize",
                "source_ids": [m["id"] for m in cluster_memories],
                "embedding": mean_embedding,
                "retention_score": 1.0,
                "created_at": datetime.now(),
            },
        )

        for mem in cluster_memories:
            self.memory_store.delete(mem["id"])

        self._record_audit(
            strategy="summarize",
            cluster=cluster_memories,
            consolidated_id=consolidated_id,
            summary_preview=summary[:100],
            fallback_used=None,
        )
        return {"merged": len(cluster_memories), "pruned": 0}

    def _llm_summarize(self, cluster_memories: List[Dict[str, Any]]) -> str:
        """Invoke the configured summariser on a cluster's contents.

        Default summariser is the deterministic template
        :func:`_default_summarizer`; a real LLM-backed summariser can
        be passed to the constructor (see the class docstring).
        """

        contents = [str(m.get("content", "")) for m in cluster_memories]
        return self.summarizer(contents)

    def _prune_low_retention(self, memories: List[Dict[str, Any]]) -> int:
        pruned_count = 0
        for memory in memories:
            score = memory.get("retention_score", 0.5)
            if score < self.config.retention_score_cutoff:
                self.memory_store.delete(memory["id"])
                self._record_audit(
                    strategy="prune",
                    cluster=[memory],
                    consolidated_id=None,
                    summary_preview="",
                    fallback_used=None,
                )
                pruned_count += 1
                logger.debug("Pruned memory %s with score %.3f", memory["id"], score)
        return pruned_count

    # ------------------------------------------------------------------ utils

    def _record_audit(
        self,
        *,
        strategy: str,
        cluster: List[Dict[str, Any]],
        consolidated_id: Optional[str],
        summary_preview: str,
        fallback_used: Optional[str],
    ) -> None:
        self.audit_log.append(
            ConsolidationAuditEntry(
                timestamp=datetime.now(),
                strategy=strategy,
                cluster_size=len(cluster),
                source_ids=[m["id"] for m in cluster],
                consolidated_id=consolidated_id,
                summary_preview=summary_preview,
                fallback_used=fallback_used,
            )
        )

    def should_run_consolidation(self) -> bool:
        """Return True iff enough time has elapsed since the last cycle."""

        if not self.config.enable_consolidation:
            return False
        if self._last_consolidation_time is None:
            return True

        elapsed = datetime.now() - self._last_consolidation_time
        return elapsed.total_seconds() >= self.config.consolidation_interval_hours * 3600
