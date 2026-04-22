# dreamfeed

> A cognition-inspired fork of [mem0](https://github.com/mem0ai/mem0). Memories fade, dreams consolidate, and the retriever slowly learns your taste.

`dreamfeed` is a personal hack on top of the open-source [mem0](https://github.com/mem0ai/mem0) memory layer. Upstream `mem0` treats a memory as a vector + metadata row that either exists or doesn't. `dreamfeed` adds three small, independently-toggleable mechanisms that let the memory store behave a bit more like a noisy, forgetful, sleep-dependent mind — and a bit less like a database.

Everything interesting lives in a separate package, `mem0_cognitive/`. The upstream `mem0/` runtime is untouched except for a handful of opt-in hook points, so you can `pip install -e .` this repo and still use plain `mem0` if you don't set the cognitive flag.

---

## What this fork adds on top of `mem0`

| Module | What it does | Code path |
| --- | --- | --- |
| **Affective retention** | Each memory has a time-decaying retention score shaped by the emotional intensity it was stored with. High-emotion memories decay slower. Retrieval re-ranks using `affective_composite = α·similarity + (1-α)·retention`. | `mem0_cognitive/retention/`, `mem0_cognitive/emotion/` |
| **Sleep consolidation** | A background pass clusters near-duplicate episodic memories, asks an LLM to abstract each cluster into a single consolidated memory, writes the new memory back to the vector store, and leaves an audit trail on the originals. Think of it as a scheduled "dream" that garbage-collects redundant traces into summaries. | `mem0_cognitive/consolidation/` |
| **Adaptive reweighting** | A lightweight per-user top-$k$ reward-weighted-averaging heuristic that nudges the retention-law parameters (`λ`, `τ_base`) toward whatever setting is empirically giving the best retrieval outcomes for that user. Not a real Bayesian optimizer — more like a slow-moving EWMA on a reward signal. | `mem0_cognitive/meta_learner/` |

All three modules are opt-in via `MemoryConfig.cognitive = True | dict` or the env var `MEM0_COGNITIVE_ENABLED=1`. When off, the `Memory` class behaves identically to upstream `mem0`.

---

## Why this fork exists

I wanted to see how far you can push a plain vector-memory layer toward something that *feels* like a memory and not a log. A few things bugged me about the default semantics:

- **No forgetting.** A year-old passing remark about the user's coffee order is retrieved with the same confidence as something they repeated last week. The vector store has no intrinsic notion of time.
- **No salience.** If the user cried while telling you something, that should probably stick. If they muttered it while distracted, it probably shouldn't. Nothing in the retrieval pipeline notices.
- **No background work.** Memory accumulates forever. There is no equivalent of the "sleep pass" that biological memory gets every night, where redundant episodes get collapsed into abstractions.
- **No personalization of the retriever itself.** Every user gets the same retention curve and the same retrieval weights. But "important" is a very personal word.

`dreamfeed` is a minimal, ablatable attempt at adding those four things. It is **not** a production system and it is **not** a benchmark-chasing project — the cognitive modules are a toy I wrote to think about the problem, and they are disabled by default. If you want real, tested, scalable memory, use [upstream mem0](https://github.com/mem0ai/mem0). If you want to play with emotion-weighted decay curves and scheduled consolidation, stick around.

---

## Quickstart

```bash
git clone https://github.com/Jackksonns/mem0gogogo.git
cd mem0gogogo
pip install -e .
```

This installs two packages into your environment:

- `mem0` — the upstream mem0 runtime, unchanged.
- `mem0_cognitive` — the dreamfeed extensions.

### Use plain mem0 (default, cognitive off)

```python
from mem0 import Memory

m = Memory()
m.add("I'm allergic to shellfish", user_id="alice")
print(m.search("what should I avoid for dinner?", user_id="alice"))
```

### Use dreamfeed (cognitive on)

```python
from mem0 import Memory
from mem0.configs.base import MemoryConfig

cfg = MemoryConfig(cognitive=True)
m = Memory(config=cfg)

m.add("I'm terrified of flying — last flight was turbulent for two hours",
      user_id="alice")

results = m.search("upcoming trip planning", user_id="alice")
for r in results:
    print(r["memory"],
          "retention=", r.get("retention_score"),
          "composite=", r.get("affective_composite_score"))

# Run a sleep-consolidation pass offline
m.run_sleep_consolidation()
```

There's also a runnable no-API-key demo at <ref_file file="/home/ubuntu/repos/mem0gogogo/examples/cognitive_memory_demo.py" /> that walks through emotion extraction, retention scoring, and sleep consolidation end-to-end against a stub LLM.

---

## Repository layout

```
mem0gogogo/
├── mem0/                        # Upstream mem0 runtime (unchanged except opt-in hooks)
├── mem0_cognitive/              # dreamfeed extensions (this fork's code)
│   ├── emotion/                 #   zero-shot LLM + lexicon-fallback emotion extractor
│   ├── retention/               #   emotion-weighted Ebbinghaus decay scorer
│   ├── consolidation/           #   async sleep-consolidation engine (cluster → abstract → writeback)
│   ├── meta_learner/            #   top-k reward-weighted-averaging parameter tuner
│   └── integration/             #   hooks that wire the above into mem0.Memory
├── tests/cognitive/             # 137 offline unit tests for the above
├── examples/cognitive_memory_demo.py
├── evaluation_mem0_original/    # Upstream mem0 evaluation harness (not modified by this fork)
├── paper/                       # LaTeX write-up describing the mechanisms and math
└── ...                          # Everything else is inherited from upstream mem0 (server, cli,
                                 # mem0-ts, openmemory, docs, cookbooks, embedchain, plugins, ...)
```

`paper/` is kept in the repo as a long-form writeup of the math behind the three modules — the retention law, the consolidation audit, the averaging heuristic — but `dreamfeed` is **not** a paper-submission project. Treat `paper/` as documentation, not a claim of results.

---

## Running the tests

```bash
pip install -e ".[test]"
pytest tests/cognitive -q
```

All 137 tests are offline and deterministic: no network, no `OPENAI_API_KEY`, no Docker. They cover the retention law (49 tests pinning monotonicity and Ebbinghaus fallback), the sleep consolidator (14 tests covering writeback + audit + non-recursive fallback), the SDK integration hooks (24 tests), the emotion analyzer (26 tests for fail-open / `method` field / lexicon bounds), and the meta-learner (24 tests for warm-up, top-$k$ transition, convergence smoke, GP-negative-space).

---

## Honest limitations

These are the things I know are currently hand-wavy in this fork. None of them are secretly fine:

- The emotion extractor is a zero-shot LLM prompt with a lexicon fallback. It has **not** been calibrated against a gold-annotated dataset; treat the intensity value as a coarse signal, not a measurement.
- The meta-learner is reward-weighted averaging over a small history, not a Gaussian-process Bayesian optimizer. It does not model uncertainty and cannot genuinely explore.
- The consolidation clusterer is greedy "assign to first compatible cluster", not DBSCAN.
- There is no end-to-end benchmark. The tests pin algorithmic properties; they do not claim the fork retrieves "better" than upstream on any public benchmark.

If any of these bother you, they're all documented as future work in `paper/sections/conclusion.tex`.

---

## Credits & license

Built on top of the excellent [mem0](https://github.com/mem0ai/mem0) by the Mem0 team. All original mem0 code remains under its upstream license; this fork is distributed under the same license.
