# make_subset_figures.py
# -*- coding: utf-8 -*-
"""
Re-draw the Q2 per-dataset figures for a SUBSET of conditions, into a separate
output tree, leaving the full-condition figures untouched.

Motivation: the E3/ISO/BDM experiment is plotted with all three groups, but the
main figure only contrasts E3 with ISO. This produces a matching E3-vs-ISO-only
set so the supplementary and main figures show the same comparison, without
deleting or overwriting the three-group version.

How it works
------------
Nothing in the pipeline is modified. The script:
  1. copies the dataset's built tables into a sibling output tree
     (``_py_out_<W>min_<TAG>``; underscore-prefixed, so ``discover_csvs``
     ignores it as it does every other output directory),
  2. redirects ``cfg.out_root()`` at that tree, which moves the tables the
     plots read, the figures, AND the stats workbook together - so the
     full-condition workbook is never overwritten,
  3. restricts ``Q2_CELLS_CONDITIONS_ORDER`` and ``STATS["planned_pairs"]`` to
     the requested subset (the latter matters: the fold-change panel reads the
     planned-pairs superset directly, so without it BDM would still be drawn),
  4. runs the real ``plot_Q2_cells_events.main()``, so the output is produced
     by the same code as everything else.

Statistics follow automatically. With only E3 and ISO present, one planned pair
matches, that pair is a control family of one, and Holm on a single p-value is
the identity - so the figure shows a plain two-tailed Welch t-test, recorded in
the workbook as ``welch_none``.

Run:  python make_subset_figures.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcium_config as cfg
from paths_local import DATASET_ROOTS


# =============================================================================
# Config - edit here
# =============================================================================
# (dataset key, conditions to keep in order, planned pairs, output tag)
JOBS = [
    dict(dataset="ISO",
         conditions=["E3", "ISO"],
         planned_pairs=[("E3", "ISO")],
         tag="E3_ISO"),
]


def run_job(dataset: str, conditions: list, planned_pairs: list, tag: str) -> Path:
    root = DATASET_ROOTS[dataset]
    wsuf = cfg.window_suffix()
    src = root / f"_py_out_{wsuf}min"
    dst = root / f"_py_out_{wsuf}min_{tag}"

    src_tables = src / "tables"
    if not src_tables.exists():
        print(f"[ERROR] no built tables for {dataset}: {src_tables}")
        print("        run the normal pipeline for this dataset first")
        sys.exit(1)

    # fresh tree each run, so nothing stale can survive
    shutil.rmtree(dst, ignore_errors=True)
    (dst / "tables").mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src_tables.glob("*"):
        if f.is_file():
            shutil.copy2(f, dst / "tables" / f.name)
            n += 1
    print(f"[{dataset} -> {tag}] copied {n} tables into {dst.name}/")

    # redirect every output path at once (tables, plots, stats workbook)
    cfg.out_root = lambda _d=dst: _d

    # restrict the conditions AND the fold-change planned pairs
    cfg.Q2_CELLS_CONDITIONS_ORDER = list(conditions)
    cfg.STATS["planned_pairs"] = [tuple(p) for p in planned_pairs]
    # The full three-group run is reported as a one-way ANOVA with all pairwise
    # post-hocs; this subset is the two-group t-test figure, so it must not
    # inherit that fallback.
    cfg.STATS_FALLBACK_PAIR_IDS = set()
    print(f"    conditions   = {conditions}")
    print(f"    planned_pairs= {planned_pairs}")

    import plot_Q2_cells_events as q2
    q2.main()
    return dst


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    for job in JOBS:
        out = run_job(**job)
        print(f"[OK] {job['dataset']} subset figures -> {out}\n")


if __name__ == "__main__":
    main()
