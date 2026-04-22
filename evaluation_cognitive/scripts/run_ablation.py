"""Run one ablation configuration end-to-end (interface-only).

Planned CLI::

    python -m evaluation_cognitive.scripts.run_ablation \\
        --config evaluation_cognitive/configs/ablation_full.yaml

The script will:

1. Load the given config YAML.
2. Instantiate ``mem0.Memory`` with the matching ``CognitiveHooksConfig``.
3. Load the dataset (CognitiveBench or LoCoMo) via the appropriate
   generator / adapter.
4. Stream dialogues through ``Memory.add`` + ``Memory.search`` + optional
   ``Memory.run_sleep_consolidation``.
5. Persist per-turn outputs under ``expected_outputs/<run_id>/``.

Currently every entry point raises ``NotImplementedError``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def run(config_path: Path) -> None:
    """Execute one ablation run described by ``config_path``.

    Raises:
        NotImplementedError: Always, in this skeleton release.
    """
    raise NotImplementedError(
        f"run_ablation is not implemented yet (config={config_path}). "
        "See evaluation_cognitive/README.md for the reproducibility "
        "checklist; this script depends on the CognitiveBench generator "
        "and the LoCoMo adapter, both of which are currently "
        "interface-only."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to an ablation config YAML under configs/.",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
