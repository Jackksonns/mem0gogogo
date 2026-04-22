"""Unit tests for :mod:`mem0_cognitive.consolidation.engine`.

These tests pin the contract that was absent in earlier revisions:

- The ``average`` and ``summarize`` strategies actually **write
  consolidated records** into the memory store and **delete** their
  source records.
- ``summarize`` failure falls back to ``keep_best`` **non-recursively**
  (no risk of unbounded recursion).
- Every materialise / prune is recorded in the audit log with the
  correct strategy, source ids, and fallback annotation.
- The config's ``enable_consolidation = False`` toggle short-circuits
  the cycle without touching the store.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pytest

from mem0_cognitive.consolidation.configs import ConsolidationConfig
from mem0_cognitive.consolidation.engine import (
    ConsolidationAuditEntry,
    InMemoryStore,
    SleepConsolidator,
    _default_summarizer,
)


def _seed_store(
    store: InMemoryStore,
    *,
    cluster_a_size: int = 3,
    cluster_b_size: int = 3,
    singleton_count: int = 2,
    retention_score: float = 0.05,
) -> Dict[str, List[str]]:
    """Populate ``store`` with two low-retention clusters + a few singletons.

    Returns ``{"cluster_a": [...ids], "cluster_b": [...ids], "singletons": [...ids]}``.
    The two clusters use near-orthogonal embedding directions so that
    cosine similarity within a cluster is ~1.0 and between clusters is
    ~0.0, matching the default ``clustering_threshold = 0.85``.

    Every seeded memory is created 48 hours in the past so that the
    default ``consolidation_interval_hours = 6`` considers them eligible.
    """

    created_at = datetime.now() - timedelta(hours=48)
    ids: Dict[str, List[str]] = {"cluster_a": [], "cluster_b": [], "singletons": []}

    for i in range(cluster_a_size):
        mid = store.add(
            f"cluster A fact #{i}",
            {
                "embedding": [1.0, 0.0, 0.0],
                "created_at": created_at,
                "retention_score": retention_score + i * 0.001,
            },
        )
        ids["cluster_a"].append(mid)

    for i in range(cluster_b_size):
        mid = store.add(
            f"cluster B fact #{i}",
            {
                "embedding": [0.0, 1.0, 0.0],
                "created_at": created_at,
                "retention_score": retention_score + i * 0.001,
            },
        )
        ids["cluster_b"].append(mid)

    # Singleton directions - each one is ~orthogonal to everything else
    # so they cannot join any cluster.
    for i in range(singleton_count):
        mid = store.add(
            f"singleton #{i}",
            {
                "embedding": [0.0, 0.0, 1.0 if i == 0 else -1.0],
                "created_at": created_at,
                "retention_score": retention_score,
            },
        )
        ids["singletons"].append(mid)

    return ids


def _run(coro):
    return asyncio.run(coro)


def test_keep_best_writes_nothing_new_but_deletes_losers():
    """``keep_best`` retains the highest-retention source in place."""

    store = InMemoryStore()
    seeded = _seed_store(store, cluster_b_size=0, singleton_count=0)
    cfg = ConsolidationConfig(
        generalization_strategy="keep_best", min_memories_for_consolidation=2
    )
    consolidator = SleepConsolidator(store, cfg)

    stats = _run(consolidator.run_consolidation_cycle())

    assert stats["clusters_formed"] == 1
    assert stats["consolidated"] == len(seeded["cluster_a"]) - 1

    # Exactly one member of the original cluster survives and it is an
    # original source id (no new record was written).
    survivors = [
        r for r in store.get_all() if r["id"] in seeded["cluster_a"]
    ]
    assert len(survivors) == 1
    assert survivors[0]["id"] in seeded["cluster_a"]

    audit = consolidator.audit_log
    assert len(audit) == 1
    assert audit[0].strategy == "keep_best"
    assert audit[0].consolidated_id in seeded["cluster_a"]
    assert audit[0].fallback_used is None


def test_average_writes_consolidated_and_deletes_all_sources():
    """``average`` materialises a new record and removes every source."""

    store = InMemoryStore()
    seeded = _seed_store(store, cluster_b_size=0, singleton_count=0)
    cfg = ConsolidationConfig(
        generalization_strategy="average", min_memories_for_consolidation=2
    )
    consolidator = SleepConsolidator(store, cfg)

    stats = _run(consolidator.run_consolidation_cycle())
    assert stats["consolidated"] == len(seeded["cluster_a"])

    # All sources are gone.
    remaining_ids = {r["id"] for r in store.get_all()}
    for sid in seeded["cluster_a"]:
        assert sid not in remaining_ids

    # Exactly one new record is present with the correct metadata.
    new_records = [
        r for r in store.get_all()
        if r.get("consolidation_strategy") == "average"
    ]
    assert len(new_records) == 1
    rec = new_records[0]
    assert rec["content"].startswith("[consolidated x")
    assert set(rec["source_ids"]) == set(seeded["cluster_a"])
    assert rec["embedding"] is not None
    assert rec["retention_score"] == pytest.approx(1.0)

    audit = [e for e in consolidator.audit_log if e.strategy == "average"]
    assert len(audit) == 1
    assert audit[0].consolidated_id == rec["id"]
    assert audit[0].fallback_used is None


def test_summarize_uses_injected_summarizer_and_deletes_sources():
    """``summarize`` routes cluster contents through the supplied callable."""

    store = InMemoryStore()
    seeded = _seed_store(store, cluster_b_size=0, singleton_count=0)

    captured_calls: List[List[str]] = []

    def fake_llm(contents: List[str]) -> str:
        captured_calls.append(list(contents))
        return f"GENERATED_SUMMARY:: {len(contents)} items"

    cfg = ConsolidationConfig(
        generalization_strategy="summarize", min_memories_for_consolidation=2
    )
    consolidator = SleepConsolidator(store, cfg, summarizer=fake_llm)

    stats = _run(consolidator.run_consolidation_cycle())

    assert stats["consolidated"] == len(seeded["cluster_a"])
    assert len(captured_calls) == 1
    assert len(captured_calls[0]) == len(seeded["cluster_a"])

    new_records = [
        r for r in store.get_all()
        if r.get("consolidation_strategy") == "summarize"
    ]
    assert len(new_records) == 1
    assert new_records[0]["content"].startswith("GENERATED_SUMMARY::")

    audit = [e for e in consolidator.audit_log if e.strategy == "summarize"]
    assert len(audit) == 1
    assert audit[0].fallback_used is None


def test_summarize_failure_falls_back_to_keep_best_without_recursion():
    """Fallback is non-recursive and recorded in the audit log."""

    store = InMemoryStore()
    seeded = _seed_store(store, cluster_b_size=0, singleton_count=0)

    call_counter = {"n": 0}

    def exploding_summarizer(_contents: List[str]) -> str:
        call_counter["n"] += 1
        raise RuntimeError("simulated LLM outage")

    cfg = ConsolidationConfig(
        generalization_strategy="summarize", min_memories_for_consolidation=2
    )
    consolidator = SleepConsolidator(
        store, cfg, summarizer=exploding_summarizer
    )

    stats = _run(consolidator.run_consolidation_cycle())

    # Exactly one attempted summarisation call - fallback must not re-call.
    assert call_counter["n"] == 1

    # keep_best survivor remains, others are deleted, no new record.
    survivors = [
        r for r in store.get_all() if r["id"] in seeded["cluster_a"]
    ]
    assert len(survivors) == 1
    assert stats["consolidated"] == len(seeded["cluster_a"]) - 1

    audit = consolidator.audit_log
    assert any(
        e.strategy == "keep_best" and e.fallback_used == "summarize_failed"
        for e in audit
    )


def test_singleton_candidates_are_pruned_not_consolidated():
    """Singletons below the cutoff are deleted and audited as 'prune'."""

    store = InMemoryStore()
    seeded = _seed_store(store, cluster_b_size=0, singleton_count=2)
    cfg = ConsolidationConfig(
        generalization_strategy="keep_best", min_memories_for_consolidation=2
    )
    consolidator = SleepConsolidator(store, cfg)

    stats = _run(consolidator.run_consolidation_cycle())

    remaining = {r["id"] for r in store.get_all()}
    for sid in seeded["singletons"]:
        assert sid not in remaining
    assert stats["pruned"] >= len(seeded["singletons"])

    prune_entries = [e for e in consolidator.audit_log if e.strategy == "prune"]
    assert len(prune_entries) == len(seeded["singletons"])


def test_two_clusters_are_handled_independently():
    store = InMemoryStore()
    seeded = _seed_store(store, cluster_a_size=3, cluster_b_size=3, singleton_count=0)
    cfg = ConsolidationConfig(
        generalization_strategy="average", min_memories_for_consolidation=2
    )
    consolidator = SleepConsolidator(store, cfg)

    stats = _run(consolidator.run_consolidation_cycle())
    assert stats["clusters_formed"] == 2
    assert stats["consolidated"] == len(seeded["cluster_a"]) + len(seeded["cluster_b"])

    consolidated_records = [
        r for r in store.get_all()
        if r.get("consolidation_strategy") == "average"
    ]
    assert len(consolidated_records) == 2

    all_source_ids = {
        sid for r in consolidated_records for sid in r["source_ids"]
    }
    assert all_source_ids == set(seeded["cluster_a"] + seeded["cluster_b"])


def test_disabled_config_short_circuits_without_side_effects():
    store = InMemoryStore()
    seeded = _seed_store(store)
    before_count = len(store)
    cfg = ConsolidationConfig(enable_consolidation=False)
    consolidator = SleepConsolidator(store, cfg)

    stats = _run(consolidator.run_consolidation_cycle())
    assert stats["retrieved"] == 0
    assert stats["consolidated"] == 0
    assert stats["pruned"] == 0
    assert len(store) == before_count
    assert consolidator.audit_log == []
    # touch `seeded` so linters don't flag it as unused
    assert all(isinstance(v, list) for v in seeded.values())


def test_default_summarizer_shape():
    s = _default_summarizer(["a", "b", "c"])
    assert s.startswith("[consolidated x3]")
    assert "a" in s and "b" in s and "c" in s


def test_audit_entries_have_expected_fields():
    store = InMemoryStore()
    _seed_store(store, cluster_b_size=0, singleton_count=0)
    cfg = ConsolidationConfig(
        generalization_strategy="average", min_memories_for_consolidation=2
    )
    consolidator = SleepConsolidator(store, cfg)

    _run(consolidator.run_consolidation_cycle())

    entry: ConsolidationAuditEntry = consolidator.audit_log[0]
    assert isinstance(entry.timestamp, datetime)
    assert entry.strategy == "average"
    assert entry.cluster_size >= 2
    assert len(entry.source_ids) == entry.cluster_size
    assert entry.consolidated_id is not None


def test_in_memory_store_roundtrip():
    store = InMemoryStore()
    mid = store.add("hello", {"retention_score": 0.3})
    assert any(r["id"] == mid for r in store.get_all())
    store.delete(mid)
    assert all(r["id"] != mid for r in store.get_all())


def test_below_minimum_memories_short_circuits():
    store = InMemoryStore()
    _seed_store(store, cluster_a_size=1, cluster_b_size=1, singleton_count=0)
    cfg = ConsolidationConfig(
        generalization_strategy="keep_best", min_memories_for_consolidation=10
    )
    consolidator = SleepConsolidator(store, cfg)

    stats = _run(consolidator.run_consolidation_cycle())
    assert stats["clusters_formed"] == 0
    assert stats["consolidated"] == 0
    assert stats["pruned"] == 0


def test_run_marks_last_consolidation_time_and_gate():
    store = InMemoryStore()
    _seed_store(store)
    cfg = ConsolidationConfig(min_memories_for_consolidation=2)
    consolidator = SleepConsolidator(store, cfg)

    assert consolidator.should_run_consolidation() is True
    _run(consolidator.run_consolidation_cycle())
    # Immediately after running, the gate is closed.
    assert consolidator.should_run_consolidation() is False


def test_audit_preview_is_length_bounded():
    store = InMemoryStore()
    huge_content = "x" * 2000
    created_at = datetime.now() - timedelta(hours=48)
    ids = [
        store.add(
            huge_content,
            {"embedding": [1.0, 0.0, 0.0], "created_at": created_at,
             "retention_score": 0.05},
        )
        for _ in range(3)
    ]
    cfg = ConsolidationConfig(
        generalization_strategy="average", min_memories_for_consolidation=2
    )
    consolidator = SleepConsolidator(store, cfg)
    _run(consolidator.run_consolidation_cycle())

    entry = consolidator.audit_log[0]
    assert len(entry.summary_preview) <= 100
    # sanity: ids are referenced
    assert all(sid in entry.source_ids for sid in ids)


def test_compute_similarity_returns_zero_for_missing_embeddings():
    store = InMemoryStore()
    consolidator = SleepConsolidator(store)
    a: Dict[str, Any] = {"embedding": None}
    b: Dict[str, Any] = {"embedding": [1.0, 0.0]}
    assert consolidator._compute_similarity(a, b) == 0.0
    assert consolidator._compute_similarity({"embedding": [0.0, 0.0]}, b) == 0.0
