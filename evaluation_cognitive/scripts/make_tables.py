"""Emit the LaTeX tables consumed by ``paper/main.tex`` (interface-only).

Planned CLI::

    python -m evaluation_cognitive.scripts.make_tables \\
        --runs-dir evaluation_cognitive/expected_outputs/ \\
        --out paper/generated/tables.tex

Reads the per-run outputs produced by ``run_ablation.py`` and emits the
LaTeX fragments that the experiments section references. Currently every
entry point raises ``NotImplementedError``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def make_tables(runs_dir: Path, out: Path) -> None:
    """Emit LaTeX table fragments from a directory of ablation runs.

    Raises:
        NotImplementedError: Always, in this skeleton release.
    """
    raise NotImplementedError(
        f"make_tables is not implemented yet (runs_dir={runs_dir}, "
        f"out={out}). This script depends on run_ablation.py having "
        "produced stable per-run outputs, which is itself gated on the "
        "CognitiveBench generator (see evaluation_cognitive/README.md)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    make_tables(args.runs_dir, args.out)


if __name__ == "__main__":
    main()
