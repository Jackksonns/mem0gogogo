"""Interface-contract tests for :class:`EmotionAnalyzer`.

These tests pin the contract that the paper's Section 3.4 ("Emotion
Extractor: Implementation and Reliability Caveats", narrowed in Stage 7)
promises about the extractor:

1. **Fail-open collapse.** When both the LLM path and the lexicon
   fallback fail (or the fallback is disabled), the extractor must not
   raise; it must return a neutral-intensity record so the retention
   formula collapses to the emotion-agnostic Ebbinghaus curve.

2. **Method field is always populated.** Each returned record carries
   a ``method`` field in ``{'llm', 'lexicon', 'none'}`` so downstream
   logging can tell the three paths apart without peeking at state.

3. **Lexicon output is bounded in [0, 1].** The fallback must not
   produce values outside the documented domain of the retention law,
   otherwise :class:`AffectiveRetentionScorer` silently clamps and
   logs.

4. **Retention composes with the extractor as claimed.** A
   fail-open record (intensity == 0.0) paired with the retention law
   reproduces the Ebbinghaus curve exactly; a high-intensity record
   produces strictly greater retention at the same elapsed time.

The tests deliberately do *not* make any real LLM calls. The LLM path
is exercised via a stub client patched into ``_llm_client``; this keeps
the tests deterministic and runnable offline on CI.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from mem0_cognitive.emotion.analyzer import EmotionAnalyzer
from mem0_cognitive.emotion.configs import EmotionConfig
from mem0_cognitive.retention.configs import RetentionConfig
from mem0_cognitive.retention.scorer import AffectiveRetentionScorer

# ---------------------------------------------------------------------------
# Stub LLM client used to exercise the LLM branch without network IO.
# ---------------------------------------------------------------------------


class _StubLLMClient:
    """Minimal stand-in for the OpenAI client used by the analyzer.

    Only the single call path exercised in ``_extract_via_llm`` is
    implemented. The ``payload`` is returned verbatim as the JSON
    content of the assistant message; a ``raises`` argument forces the
    call to raise so the fail-open path can be exercised.
    """

    def __init__(self, payload: str = "", raises: Exception | None = None):
        self._payload = payload
        self._raises = raises
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs: Any):
        if self._raises is not None:
            raise self._raises
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self._payload),
                )
            ]
        )


# ---------------------------------------------------------------------------
# Invariant 1: fail-open.
# ---------------------------------------------------------------------------


def test_fail_open_returns_neutral_when_llm_fails_and_fallback_disabled():
    """Paper Section 3.4: if both paths fail, $E_i$ defaults to neutral."""
    cfg = EmotionConfig(enable_lexicon_fallback=False)
    analyzer = EmotionAnalyzer(cfg)
    analyzer._llm_client = _StubLLMClient(raises=RuntimeError("LLM unreachable"))

    result = analyzer.extract("Some utterance.")

    assert result["intensity"] == 0.0
    assert result["valence"] == "neutral"
    assert result["arousal"] == "low"
    assert result["method"] == "none"


def test_fail_open_when_llm_raises_and_fallback_is_enabled():
    """LLM failure must not raise when the lexicon fallback is enabled."""
    cfg = EmotionConfig(enable_lexicon_fallback=True)
    analyzer = EmotionAnalyzer(cfg)
    analyzer._llm_client = _StubLLMClient(raises=RuntimeError("LLM unreachable"))

    result = analyzer.extract("I feel very happy today.")

    assert result["method"] == "lexicon"
    assert 0.0 <= result["intensity"] <= 1.0


def test_extract_never_raises_even_on_malformed_llm_json():
    """Malformed LLM output must drop to the fallback, not bubble up."""
    cfg = EmotionConfig(enable_lexicon_fallback=True)
    analyzer = EmotionAnalyzer(cfg)
    analyzer._llm_client = _StubLLMClient(payload="this is not json")

    result = analyzer.extract("I love it.")

    assert result["method"] == "lexicon"
    assert 0.0 <= result["intensity"] <= 1.0


# ---------------------------------------------------------------------------
# Invariant 2: method field is always populated and correctly tagged.
# ---------------------------------------------------------------------------


def test_method_field_llm_path():
    cfg = EmotionConfig(enable_lexicon_fallback=True)
    analyzer = EmotionAnalyzer(cfg)
    analyzer._llm_client = _StubLLMClient(
        payload='{"intensity": 0.7, "valence": "positive", '
        '"arousal": "high", "rationale": "test"}'
    )

    result = analyzer.extract("I am thrilled.")

    assert result["method"] == "llm"
    assert math.isclose(result["intensity"], 0.7)
    assert result["valence"] == "positive"


def test_method_field_lexicon_path():
    cfg = EmotionConfig(enable_lexicon_fallback=True)
    analyzer = EmotionAnalyzer(cfg)
    analyzer._llm_client = _StubLLMClient(raises=RuntimeError("no model"))

    result = analyzer.extract("This is terrible.")

    assert result["method"] == "lexicon"


def test_method_field_none_path_when_fallback_disabled():
    cfg = EmotionConfig(enable_lexicon_fallback=False)
    analyzer = EmotionAnalyzer(cfg)
    analyzer._llm_client = _StubLLMClient(raises=RuntimeError("no model"))

    result = analyzer.extract("whatever")

    assert result["method"] == "none"


@pytest.mark.parametrize(
    "utterance",
    [
        "I love you.",
        "This is awful.",
        "Just some neutral text.",
        "",
    ],
)
def test_method_field_is_in_the_documented_set(utterance):
    cfg = EmotionConfig(enable_lexicon_fallback=True)
    analyzer = EmotionAnalyzer(cfg)
    analyzer._llm_client = _StubLLMClient(raises=RuntimeError("no model"))

    result = analyzer.extract(utterance)

    assert result["method"] in {"llm", "lexicon", "none"}


# ---------------------------------------------------------------------------
# Invariant 3: lexicon intensity is bounded in [0, 1].
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "utterance",
    [
        "",
        "a",
        "love love love love love love love love love love",
        "I absolutely totally really very extremely love love love this!",
        "hate hate hate terrible awful bad worst sad angry frustrated",
        "love love hate hate love hate love hate love hate",
    ],
)
def test_lexicon_intensity_bounded(utterance):
    analyzer = EmotionAnalyzer(EmotionConfig(enable_lexicon_fallback=True))
    result = analyzer._extract_via_lexicon(utterance, (0.0, 1.0))

    assert 0.0 <= result["intensity"] <= 1.0


def test_lexicon_neutral_on_empty_string():
    analyzer = EmotionAnalyzer(EmotionConfig(enable_lexicon_fallback=True))
    result = analyzer._extract_via_lexicon("", (0.0, 1.0))

    assert result["intensity"] == 0.0
    assert result["valence"] == "neutral"


def test_lexicon_positive_for_clearly_positive_utterance():
    analyzer = EmotionAnalyzer(EmotionConfig(enable_lexicon_fallback=True))
    result = analyzer._extract_via_lexicon(
        "I love this amazing wonderful fantastic gift!", (0.0, 1.0)
    )

    assert result["valence"] == "positive"
    assert result["intensity"] > 0.0


def test_lexicon_negative_for_clearly_negative_utterance():
    analyzer = EmotionAnalyzer(EmotionConfig(enable_lexicon_fallback=True))
    result = analyzer._extract_via_lexicon(
        "I hate this terrible awful horrible experience.", (0.0, 1.0)
    )

    assert result["valence"] == "negative"
    assert result["intensity"] > 0.0


# ---------------------------------------------------------------------------
# Invariant 4: the analyzer's output composes correctly with the retention
# law. The paper's Section 3.4 promise is that when the extractor fails
# open, retention reduces to the unmodulated Ebbinghaus curve. When the
# extractor fires a high-intensity record, retention at the same elapsed
# time must be strictly greater than the neutral-intensity case (for
# lambda > 0).
# ---------------------------------------------------------------------------


def _emotion_to_retention(
    analyzer: EmotionAnalyzer,
    scorer: AffectiveRetentionScorer,
    utterance: str,
    elapsed: int,
) -> Dict[str, Any]:
    record = analyzer.extract(utterance)
    r = scorer.compute(
        elapsed_turns=elapsed,
        emotion_intensity=record["intensity"],
    )
    return {"record": record, "retention": r}


def test_fail_open_collapses_retention_to_ebbinghaus_curve():
    """Fail-open (E=0) must reproduce the emotion-agnostic curve exactly."""
    cfg = EmotionConfig(enable_lexicon_fallback=False)
    analyzer = EmotionAnalyzer(cfg)
    analyzer._llm_client = _StubLLMClient(raises=RuntimeError("no model"))

    scorer = AffectiveRetentionScorer(RetentionConfig(lambda_value=1.0, tau_base=100.0))

    out = _emotion_to_retention(analyzer, scorer, "anything at all", elapsed=50)
    ebbinghaus = math.exp(-50 / 100.0)

    assert math.isclose(out["retention"], ebbinghaus, rel_tol=1e-9)
    assert out["record"]["method"] == "none"
    assert out["record"]["intensity"] == 0.0


def test_high_intensity_record_increases_retention_over_neutral():
    """High-$E$ record must slow decay relative to a neutral record."""
    cfg = EmotionConfig(enable_lexicon_fallback=True)
    analyzer_high = EmotionAnalyzer(cfg)
    analyzer_high._llm_client = _StubLLMClient(
        payload='{"intensity": 0.9, "valence": "positive", '
        '"arousal": "high", "rationale": "salient"}'
    )
    analyzer_low = EmotionAnalyzer(cfg)
    analyzer_low._llm_client = _StubLLMClient(
        payload='{"intensity": 0.0, "valence": "neutral", '
        '"arousal": "low", "rationale": "flat"}'
    )

    scorer = AffectiveRetentionScorer(RetentionConfig(lambda_value=1.0, tau_base=100.0))
    high = _emotion_to_retention(analyzer_high, scorer, "x", elapsed=50)
    low = _emotion_to_retention(analyzer_low, scorer, "x", elapsed=50)

    assert high["retention"] > low["retention"]


def test_llm_intensity_is_normalised_into_unit_interval():
    """Even if the LLM returns out-of-scale values, the record is clamped."""
    cfg = EmotionConfig(scale=(0.0, 10.0))
    analyzer = EmotionAnalyzer(cfg)
    analyzer._llm_client = _StubLLMClient(
        payload='{"intensity": 42.0, "valence": "positive", '
        '"arousal": "high", "rationale": "out of range"}'
    )

    result = analyzer.extract("huge!")

    assert 0.0 <= result["intensity"] <= 1.0
    assert result["method"] == "llm"


def test_llm_intensity_negative_is_clamped_to_zero():
    cfg = EmotionConfig(scale=(0.0, 1.0))
    analyzer = EmotionAnalyzer(cfg)
    analyzer._llm_client = _StubLLMClient(
        payload='{"intensity": -5.0, "valence": "neutral", '
        '"arousal": "low", "rationale": "negative"}'
    )

    result = analyzer.extract("whatever")

    assert result["intensity"] == 0.0
    assert result["method"] == "llm"


# ---------------------------------------------------------------------------
# Miscellaneous contract checks.
# ---------------------------------------------------------------------------


def test_extract_returns_all_documented_keys():
    cfg = EmotionConfig(enable_lexicon_fallback=True)
    analyzer = EmotionAnalyzer(cfg)
    analyzer._llm_client = _StubLLMClient(raises=RuntimeError("force fallback"))

    result = analyzer.extract("some text")

    for key in ("intensity", "valence", "arousal", "rationale", "method"):
        assert key in result, f"missing key: {key}"


def test_config_rejects_invalid_temperature():
    with pytest.raises(ValueError):
        EmotionConfig(temperature=-0.1)
    with pytest.raises(ValueError):
        EmotionConfig(temperature=5.0)


def test_config_rejects_invalid_scale():
    with pytest.raises(ValueError):
        EmotionConfig(scale=(1.0, 1.0))
    with pytest.raises(ValueError):
        EmotionConfig(scale=(2.0, 1.0))
