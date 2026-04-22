"""Unit tests for :mod:`mem0_cognitive.integration.hooks`.

Pins the Stage-5 contract:

- ``CognitiveHooks.from_config`` is opt-in (returns ``None`` unless the
  host config carries a ``cognitive`` attribute or
  ``MEM0_COGNITIVE_ENABLED`` is truthy).
- ``enrich_memory_metadata`` always populates the three emotion fields
  on the metadata dict, even when the analyser raises.
- ``apply_retention_reranking`` adds both ``retention_score`` and
  ``affective_composite_score`` and leaves the original ``score``
  untouched.
- ``run_sleep_cycle`` wires through to SleepConsolidator and returns
  the same stats shape.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mem0_cognitive.consolidation.configs import ConsolidationConfig
from mem0_cognitive.consolidation.engine import InMemoryStore
from mem0_cognitive.emotion.analyzer import EmotionAnalyzer
from mem0_cognitive.integration.hooks import (
    CognitiveHooks,
    CognitiveHooksConfig,
    _parse_iso_timestamp,
    rerank_candidates_by_affective_composite,
)
from mem0_cognitive.retention.configs import RetentionConfig


def test_from_config_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MEM0_COGNITIVE_ENABLED", raising=False)
    assert CognitiveHooks.from_config(None) is None


def test_from_config_env_truthy(monkeypatch):
    monkeypatch.setenv("MEM0_COGNITIVE_ENABLED", "1")
    hooks = CognitiveHooks.from_config(None)
    assert hooks is not None
    assert hooks.emotion_analyzer is not None
    assert hooks.retention_scorer is not None


@pytest.mark.parametrize("flag", ["true", "YES", "On", "yes", "true"])
def test_from_config_env_aliases(monkeypatch, flag):
    monkeypatch.setenv("MEM0_COGNITIVE_ENABLED", flag)
    assert CognitiveHooks.from_config(None) is not None


def test_from_config_host_attribute_takes_precedence(monkeypatch):
    monkeypatch.delenv("MEM0_COGNITIVE_ENABLED", raising=False)

    class _Cfg:
        cognitive = True

    assert CognitiveHooks.from_config(_Cfg()) is not None


def test_from_config_host_dict_maps_to_dataclass(monkeypatch):
    monkeypatch.delenv("MEM0_COGNITIVE_ENABLED", raising=False)

    class _Cfg:
        cognitive = {"enable_retention_reranking": False}

    hooks = CognitiveHooks.from_config(_Cfg())
    assert hooks is not None
    assert hooks.retention_scorer is None
    assert hooks.emotion_analyzer is not None


def test_enrich_sets_three_emotion_fields():
    hooks = CognitiveHooks()
    metadata = {"data": "hi"}
    hooks.enrich_memory_metadata("hi", metadata)
    assert "emotion_intensity" in metadata
    assert "emotion_valence" in metadata
    assert "emotion_method" in metadata
    assert 0.0 <= metadata["emotion_intensity"] <= 1.0


def test_enrich_is_defensive_against_analyser_failure(monkeypatch):
    hooks = CognitiveHooks()

    def _boom(_text):
        raise RuntimeError("llm down")

    monkeypatch.setattr(hooks.emotion_analyzer, "extract", _boom)
    metadata = {}
    hooks.enrich_memory_metadata("hello", metadata)
    assert metadata["emotion_intensity"] == 0.0
    assert metadata["emotion_method"] == "none"


def test_enrich_respects_empty_content():
    hooks = CognitiveHooks()
    metadata = {}
    hooks.enrich_memory_metadata("", metadata)
    assert metadata["emotion_method"] == "none"


def test_enrich_forced_disabled_sets_neutral_defaults():
    hooks = CognitiveHooks(CognitiveHooksConfig(enable_emotion_enrichment=False))
    metadata = {}
    hooks.enrich_memory_metadata("some content", metadata)
    assert metadata["emotion_intensity"] == 0.0
    assert metadata["emotion_method"] == "none"


def test_retention_rerank_populates_scores():
    hooks = CognitiveHooks(
        CognitiveHooksConfig(
            retention=RetentionConfig(tau_base=100.0, lambda_value=1.0),
            retention_weight=0.5,
        )
    )
    candidates = [
        {
            "id": "a",
            "score": 0.4,
            "payload": {
                "emotion_intensity": 0.9,
                "created_at": (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
            },
        },
        {
            "id": "b",
            "score": 0.4,
            "payload": {
                "emotion_intensity": 0.0,
                "created_at": (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
            },
        },
    ]
    hooks.apply_retention_reranking(candidates)

    for cand in candidates:
        assert "retention_score" in cand
        assert "affective_composite_score" in cand
        assert 0.0 <= cand["retention_score"] <= 1.0
        assert cand["score"] == 0.4  # original untouched

    # High-emotion memory must outrank low-emotion at equal elapsed time.
    assert (
        candidates[0]["retention_score"] > candidates[1]["retention_score"]
    )
    assert (
        candidates[0]["affective_composite_score"]
        > candidates[1]["affective_composite_score"]
    )


def test_retention_rerank_no_op_when_disabled():
    hooks = CognitiveHooks(
        CognitiveHooksConfig(enable_retention_reranking=False)
    )
    candidates = [{"id": "a", "score": 0.5, "payload": {"emotion_intensity": 0.9}}]
    hooks.apply_retention_reranking(candidates)
    assert "retention_score" not in candidates[0]


def test_retention_rerank_handles_missing_payload():
    hooks = CognitiveHooks()
    candidates = [{"id": "a", "score": 0.5}]
    hooks.apply_retention_reranking(candidates)
    assert 0.0 <= candidates[0]["retention_score"] <= 1.0


def test_retention_weight_is_clamped_to_unit_interval():
    hooks = CognitiveHooks(CognitiveHooksConfig(retention_weight=5.0))
    candidates = [{"id": "a", "score": 0.3, "payload": {"emotion_intensity": 1.0}}]
    hooks.apply_retention_reranking(candidates)
    assert 0.0 <= candidates[0]["affective_composite_score"] <= 1.0


def test_parse_iso_timestamp_handles_z_suffix():
    iso = "2024-01-01T00:00:00Z"
    parsed = _parse_iso_timestamp(iso)
    assert parsed is not None
    assert parsed.year == 2024


def test_parse_iso_timestamp_handles_naive_datetime():
    now = datetime.now()
    assert _parse_iso_timestamp(now) is now


def test_parse_iso_timestamp_returns_none_on_garbage():
    assert _parse_iso_timestamp("not a date") is None
    assert _parse_iso_timestamp(None) is None
    assert _parse_iso_timestamp(object()) is None


def test_rerank_helper_is_stable_sort():
    ranked = rerank_candidates_by_affective_composite(
        [
            {"id": "a", "affective_composite_score": 0.2},
            {"id": "b", "affective_composite_score": 0.5},
            {"id": "c", "score": 0.5},  # ties with b
        ]
    )
    assert [r["id"] for r in ranked] == ["b", "c", "a"]


def test_run_sleep_cycle_invokes_consolidator():
    hooks = CognitiveHooks(
        CognitiveHooksConfig(
            consolidation=ConsolidationConfig(
                generalization_strategy="keep_best",
                min_memories_for_consolidation=2,
            )
        )
    )
    store = InMemoryStore()
    now = datetime.now()
    stale = now - timedelta(hours=48)
    for i in range(3):
        store.add(
            f"cluster fact #{i}",
            {
                "embedding": [1.0, 0.0, 0.0],
                "created_at": stale,
                "retention_score": 0.05 + i * 0.001,
            },
        )
    stats = hooks.run_sleep_cycle(store)
    assert stats["retrieved"] == 3
    assert stats["consolidated"] == 2
    assert len(store) == 1


def test_run_sleep_cycle_disabled_short_circuits():
    hooks = CognitiveHooks(
        CognitiveHooksConfig(enable_sleep_consolidation=False)
    )
    stats = hooks.run_sleep_cycle(InMemoryStore())
    assert stats["consolidated"] == 0
    assert stats["skipped_reason"].startswith("enable_sleep_consolidation")


def test_emotion_analyzer_is_reused_across_calls():
    hooks = CognitiveHooks()
    assert isinstance(hooks.emotion_analyzer, EmotionAnalyzer)
    m1, m2 = {}, {}
    hooks.enrich_memory_metadata("one", m1)
    hooks.enrich_memory_metadata("two", m2)
    assert m1["emotion_method"] == m2["emotion_method"]
