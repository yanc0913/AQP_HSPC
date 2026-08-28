# collect_supp_figures.py
# -*- coding: utf-8 -*-
"""
Gather the Ca2+ intensity boxplots into one supplementary-figure folder.

The per-dataset pipeline writes its figures deep inside each dataset's output
tree (`_py_out_<W>min/plots_svg/Q2_cells_events/boxplot_amplitude/<cell class>/
<pair_id>/...`), which is fine for browsing one dataset but awkward when
assembling a supplementary figure that spans four experiments. This script
copies just the intensity panels into `_SuppFigures/Ca_intensity/`, one
subfolder per dataset, laid out like `_MainFigures`.

It only COPIES - nothing is re-plotted and no source file is touched, so the
panels here are byte-identical to the ones in the pipeline output.

Note on the metric name: the figures are labelled "Ca2+ Intensity", but the
files and columns still use the original `amplitude` key so tables, figures and
the stats workbook keep cross-referencing. That is why the sources are read
from `boxplot_amplitude/`.

Which groups each folder shows:
  30hpf_Yoda            DMSO / Yoda1 / E3 / GsMTx4      (all groups)
  48hpf_Yoda            DMSO / Yoda1 / E3 / GsMTx4      (all groups)
  48hpf_ISO_E3_ISO_BDM  E3 / ISO / BDM, three groups    (one-way ANOVA)
                        - deliberately NOT the E3-vs-ISO two-group subset
  48hpf_MIC_Piezo       MIC / Piezo x E3 / ISO          (all groups, Tukey)

Run:  python collect_supp_figures.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcium_config as cfg
from paths_local import DATASET_ROOTS, SUPP_FIG_ROOT


# =============================================================================
# Config - edit here
# =============================================================================
WSUF = "20"
SUBDIR = "Ca_intensity"           # folder under SUPP_FIG_ROOT
METRIC_DIR = "boxplot_amplitude"  # internal key is still `amplitude`
EXTS = ("svg", "png")

# (output folder, dataset key, source subtree, note for the README)
JOBS = [
    dict(folder="30hpf_Yoda", dataset="30hpf", subtree=f"_py_out_{WSUF}min",
         note="30 hpf Yoda experiment, all four groups (DMSO / Yoda1 / E3 / "
              "GsMTx4).\nPlanned drug-vs-vehicle Welch t-tests; DMSO-Yoda1 and "
              "E3-GsMTx4 use\ndifferent vehicles, so each is its own family and "
              "no correction applies."),
    dict(folder="48hpf_Yoda", dataset="48hpf", subtree=f"_py_out_{WSUF}min",
         note="48 hpf Yoda experiment, all four groups (DMSO / Yoda1 / E3 / "
              "GsMTx4).\nSame statistics as the 30 hpf folder."),
    dict(folder="48hpf_ISO_E3_ISO_BDM", dataset="ISO", subtree=f"_py_out_{WSUF}min",
         note="48 hpf E3 / ISO / BDM experiment, ALL THREE groups.\n"
              "One-way ANOVA (omnibus p in the corner) with all three pairwise\n"
              "post-hocs, Holm-corrected across the family of three.\n\n"
              "This is deliberately NOT the E3-vs-ISO two-group version. That "
              "subset\nexists (a plain Welch t-test) and feeds the WT bar of the "
              "main figure,\nbut the supplementary figure reports all three "
              "groups."),
    dict(folder="48hpf_MIC_Piezo", dataset="Piezo", subtree=f"_py_out_{WSUF}min",
         note="48 hpf MIC / piezo-crispant x E3 / ISO, all groups.\n"
              "Type-II two-way ANOVA (label above the axes) with Tukey HSD "
              "post-hocs."),
]

HEADER = (
    "Ca2+ intensity boxplots, copied from the per-dataset pipeline output.\n"
    "Source: {src}\n"
    "These files are copies - re-running the pipeline and this script "
    "regenerates them.\n\n"
)


def run_job(folder: str, dataset: str, subtree: str, note: str) -> int:
    src_root = DATASET_ROOTS[dataset] / subtree
    dst = SUPP_FIG_ROOT / SUBDIR / folder
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)

    n = 0
    for ext in EXTS:
        base = src_root / f"plots_{ext}" / cfg.Q2_CELLS_PLOT_SUBDIR / METRIC_DIR
        if not base.exists():
            print(f"[WARN] missing source: {base}")
            continue
        for f in sorted(base.rglob(f"*.{ext}")):
            # .../boxplot_amplitude/<cell class>/<pair_id>/<file>
            cclass = f.parent.parent.name
            out = dst / f"Ca_intensity_{cfg.cell_class_display(cclass).replace(' ', '')}.{ext}"
            shutil.copy2(f, out)
            n += 1

    (dst / "READ_ME.txt").write_text(
        HEADER.format(src=src_root) + note + "\n", encoding="utf-8")
    print(f"  {folder:24s} {n} files  <- {src_root.parent.name}/{subtree}")
    return n


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    total = 0
    print(f"Collecting into {SUPP_FIG_ROOT / SUBDIR}")
    for job in JOBS:
        total += run_job(**job)
    print(f"\n[OK] {total} files in {len(JOBS)} folders")


if __name__ == "__main__":
    main()
