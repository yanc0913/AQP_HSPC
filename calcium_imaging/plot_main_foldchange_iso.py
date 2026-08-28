# plot_main_foldchange_iso.py
# -*- coding: utf-8 -*-
"""
Main-figure: ISO / E3 fold change across WT, MIC, and piezo-crispant.

Three bars per panel, all log2( ISO / mean(own-E3) ):

    WT             — from the E3/ISO/BDM experiment (Analysis_48hpf_ISO), WT
    WT+MIC         — mock-injected control from the Piezo 2x2 (MIC genotype)
    Piezo crispant — piezo-guide group from the Piezo 2x2 (Piezo genotype)

Each group is normalised to ITS OWN E3, so this asks: is the ISO response the
same in uninjected WT and in the mock-injection control, and blunted in the
crispant? The within-experiment comparison (MIC vs Piezo, same 2x2) is the
clean one; WT is an extra reference from a separate experiment (dotted line
marks the experiment boundary).

Same conventions as plot_main_foldchange.py: 2x2 grid (flat/round x
amplitude/events), box + embryo dots, reference at 0.

The p above each bar is NOT recomputed here. It is read back from the source
experiment's built stats workbook, so the bar prints the value produced by the
method that design actually warrants, and matches the per-dataset figure for
the same comparison exactly:
  * WT              -> Welch + Holm within the E3 control family (ISO and BDM
                       are both tested against E3 in that experiment)
  * MIC / Piezo     -> Tukey HSD from the genotype x drug two-way ANOVA
The method used for every bar is recorded in the `p_test` column. All
between-group comparisons go to the workbook.

Outputs to OUT_DIR:
  * combined 2x2:  main_iso_foldchange_2x2.{png,svg}
  * per panel:     panel_<E-Amp|E-Events|R-Amp|R-Events>.{svg,pdf,png}
                   (svg/pdf keep text editable for Illustrator)
  * main_iso_foldchange.csv  (per-bar folds + all between-group p)

Run:  python plot_main_foldchange_iso.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
# keep text editable in vector exports (Illustrator)
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcium_config as cfg
from plot_Q2_cells_events import draw_prism_box, style_axes, _fmt_p, apply_dense_yticks
from scipy.stats import ttest_ind


# =============================================================================
# Config — edit here
# =============================================================================
WSUF = "20"

# Data paths are machine-specific and live in paths_local.py (git-ignored).
from paths_local import DATASET_ROOTS, MAIN_FIG_DIRS

_SUMMARY = Path("_py_out_%smin" % WSUF) / "tables" / "Q2_embryo_summary_cells.csv"

DATASETS = {
    "ISO":   DATASET_ROOTS["ISO"]   / _SUMMARY,
    "Piezo": DATASET_ROOTS["Piezo"] / _SUMMARY,
}

# (label, colour, dataset_key, genotype, drug, vehicle)
# Colours come from the shared genotype palette in calcium_config.
BARS = [
    ("WT",             cfg.genotype_color("WT"),    "ISO",   "WT",    "ISO", "E3"),
    ("WT+MIC",         cfg.genotype_color("MIC"),   "Piezo", "MIC",   "ISO", "E3"),
    ("Piezo crispant", cfg.genotype_color("Piezo"), "Piezo", "Piezo", "ISO", "E3"),
]
# dotted separator AFTER these bar indices (WT is a separate experiment)
SEP_AFTER = {0}

CELLCLASSES = [("E", "flat"), ("R", "round")]        # (label, token)
METRICS = [
    ("Amp",    "amplitude", f"embryo_mean_amplitude_post_{WSUF}min",   False),
    ("Events", "events",    f"mean_events_per_cell_per_{WSUF}min_post", False),
]

# Per-metric effect transform (see plot_main_foldchange.py for rationale):
#   amplitude -> log2(ISO / mean(own-E3))   events -> ISO - mean(own-E3)
FC_MODE = {"amplitude": "log2", "events": "diff", "duration": "log2"}


def fc_ylab(mkey: str) -> str:
    if FC_MODE.get(mkey, "log2") == "diff":
        return r"$\Delta$ events / cell / %s min (ISO $-$ E3)" % WSUF
    return "log2 fold change (ISO / E3)"


# ---------- Main-figure row layout (1 x 4) ----------
# Mirrors plot_main_foldchange.py so both main figures share one visual system.
# Drawn at final placement size (~75% for a 180 mm column); do not scale below
# that or the type drops under 6 pt.
ROW_FIGSIZE   = (10.0, 3.7)     # taller than the Yoda row: "Piezo crispant" is
                                # a long rotated x-label and needs bottom room
ROW_FONT      = dict(title=10.5, ylabel=9.5, tick=9, pval=8.5, title_pad=6)
ROW_BOX_WIDTH = 0.45    # data units (per-dataset panels use STYLE 0.20).
                        # ~40% of the slot: readable without crowding.
ROW_X_STEP    = 1.15    # spacing between adjacent bars
ROW_X_GAP     = 0.55    # extra gap at the WT / crispant-experiment separator
ROW_XMARGIN   = (0.24, 0.10)   # (left, right); wide left pad keeps the longest
                               # p-string clear of the y-tick labels
ROW_JITTER    = 0.22    # dot jitter spread; keep < ROW_BOX_WIDTH
ROW_DOT_SIZE  = 20
ROW_LW_SCALE  = 1.6
ROW_YTICK_NBINS = 11
ROW_XTICK_ROT = 30
ROW_YPAD_TOP  = 0.21    # headroom for the p-value text
ROW_YPAD_BOT  = 0.05


_NL = chr(10)   # literal newline for two-line axis labels

def fc_ylab_short(mkey: str) -> str:
    """Row-figure y-label, stacked on two lines.

    The normalisation belongs on the axis: a bare delta reads as an absolute
    change, not a change relative to the control. Two lines keep each line
    short enough for the panel height, which one long string was not.
    """
    if FC_MODE.get(mkey, "log2") == "diff":
        return (r"$\Delta$ events / cell / %s min" % WSUF) + _NL + "(ISO $-$ E3)"
    return "log2 fold change" + _NL + "(ISO / E3)"

LOG2 = True
OUT_DIR = MAIN_FIG_DIRS["ISO_MIC_Piezo"]
CELL_CLASS_COL, COND_COL, GENO_COL = "cell_class", "condition", "genotype"

# Reader-facing panel titles (line 1 = metric, line 2 = cell class).
METRIC_PUB = {"amplitude": r"Ca$^{2+}$ Intensity",
              "events":    r"Ca$^{2+}$ Event Frequency",
              "duration":  r"Ca$^{2+}$ Event Duration"}


def panel_title(mkey: str, cclass: str) -> str:
    return f"{METRIC_PUB.get(mkey, mkey)}\n{cfg.cell_class_display(cclass)}"


def _vals(df, geno, cond, cclass, col):
    s = df[(df[GENO_COL].astype(str) == str(geno)) &
           (df[COND_COL].astype(str) == str(cond)) &
           (df[CELL_CLASS_COL].astype(str) == str(cclass))][col]
    a = pd.to_numeric(s, errors="coerce").to_numpy(float)
    return a[np.isfinite(a)]


_STATS = Path("_py_out_%smin" % WSUF) / "tables" / "Q2_stats_pvalues.xlsx"
_STATS_CACHE = {}

# Which output tree each bar's p-value is read from.
#
# WT comes from the E3/ISO/BDM experiment. In the full three-group tree, ISO and
# BDM share the E3 control, so that p is Holm-corrected within a family of two.
# This figure displays ONLY the ISO contrast, so it reads from the E3-vs-ISO
# subset tree (make_subset_figures.py), where the same comparison is a family of
# one and the p is a plain Welch t-test.
#
# NOTE: this is a choice about what the multiple-comparison family is, not a
# formatting detail. If the three-group figure reporting the BDM comparison also
# appears in the paper, the family arguably includes BDM and this should be set
# back to {} so the corrected value is used everywhere.
STATS_SUBTREE = {}   # set per variant by run_variant() below


def _load_pairwise(dkey: str):
    """Cached `pairwise` sheet of a dataset's built stats workbook."""
    if dkey not in _STATS_CACHE:
        sub = STATS_SUBTREE.get(dkey)
        path = (DATASET_ROOTS[dkey] / sub / "tables" / "Q2_stats_pvalues.xlsx"
                if sub else DATASET_ROOTS[dkey] / _STATS)
        try:
            _STATS_CACHE[dkey] = pd.read_excel(path, sheet_name="pairwise")
        except Exception as e:
            print(f"[WARN] cannot read {path}: {e!r}")
            _STATS_CACHE[dkey] = pd.DataFrame()
    return _STATS_CACHE[dkey]


def source_p(dkey, geno, drug, veh, cclass, mkey):
    """The drug-vs-vehicle p AS COMPUTED BY THE SOURCE EXPERIMENT.

    Each bar is a comparison that the per-dataset pipeline already tests with
    the design-appropriate method, so we read that value back instead of
    recomputing a bare Welch here:

      * MIC / Piezo come from the genotype x drug 2x2 -> Tukey HSD
        (labels look like "MIC|E3" vs "MIC|ISO").
      * WT comes from the single-genotype E3/ISO/BDM experiment, where ISO and
        BDM share the E3 control -> Welch with Holm within that control family
        (labels are the bare condition names).

    Reading it back guarantees the main figure prints the same number as the
    per-dataset figure for the same comparison. Returns (p, test_name); falls
    back to a raw Welch computed by the caller if no record is found.
    """
    df = _load_pairwise(dkey)
    if df.empty:
        return float("nan"), ""
    sub = df[(df["cell_class"].astype(str) == str(cclass)) &
             (df["metric"].astype(str) == str(mkey))]
    if sub.empty:
        return float("nan"), ""
    for a, b in ((f"{geno}|{veh}", f"{geno}|{drug}"), (str(veh), str(drug))):
        m = sub[((sub["group1"].astype(str) == a) & (sub["group2"].astype(str) == b)) |
                ((sub["group1"].astype(str) == b) & (sub["group2"].astype(str) == a))]
        m = m[~m["test"].astype(str).str.startswith("foldchange")]
        if len(m):
            return float(m["p_value"].iloc[0]), str(m["test"].iloc[0])
    return float("nan"), ""


def build_panel(data, cclass, mcol, eps, mode="log2",
                x_step=1.0, x_gap=0.4, mkey=None):
    """Return dict with box_data, colours, xpos, labels, vs_p, seps + fc_rows.

    mode = "log2": each embryo -> log2( value / mean(own-E3) ).  (amplitude)
    mode = "diff": each embryo -> value - mean(own-E3).  (events; zero-safe,
                   no pseudocount). The ISO-vs-E3 Welch p is on RAW values."""
    box_data, colors, xpos, labels, vs_p, fc = [], [], [], [], [], []
    seps = []
    x = 1.0
    for k, (label, col, dkey, geno, drug, veh) in enumerate(BARS):
        df = data[dkey]
        dv = _vals(df, geno, drug, cclass, mcol)
        vv = _vals(df, geno, veh, cclass, mcol)
        if len(dv) == 0 or len(vv) == 0:
            x += x_step
            if k in SEP_AFTER:
                seps.append(x - x_step / 2.0); x += x_gap
            continue
        vmean = float(np.mean(vv))
        if mode == "diff":
            val = dv - vmean
        else:
            fold = (dv + eps) / (vmean + eps)
            val = np.log2(fold) if LOG2 else fold
        val = val[np.isfinite(val)]
        p, p_test = source_p(dkey, geno, drug, veh, cclass, mkey)
        if not np.isfinite(p):
            # no record in the source workbook -> fall back to a raw Welch here
            try:
                p = float(ttest_ind(dv, vv, equal_var=False,
                                    nan_policy="omit").pvalue)
            except Exception:
                p = float("nan")
            p_test = "welch_none_fallback"
        box_data.append(val); colors.append(col); xpos.append(x)
        labels.append(label); vs_p.append(p)
        fc.append(dict(cell_class=cclass, metric=mcol, group=label,
                       genotype=geno, dataset=dkey, drug=drug, vehicle=veh,
                       n_drug=int(len(dv)), vehicle_mean=vmean,
                       median=float(np.median(val)), mean=float(np.mean(val)),
                       p_iso_vs_e3=p, p_test=p_test, transform=mode,
                       log2=bool(LOG2 and mode == "log2"),
                       pseudocount=(eps if mode == "log2" else 0.0),
                       _fold=val))   # _fold (=diff for events) for between-group tests
        x += x_step
        if k in SEP_AFTER:
            seps.append(x - x_step / 2.0); x += x_gap
    return dict(box_data=box_data, colors=colors, xpos=xpos, labels=labels,
                vs_p=vs_p, seps=seps, fc=fc)


def draw_panel(ax, P, ylim, ref, ylab, title, fonts=None, box_width=None,
               xmargin=0.12, jitter=None, dot_size=None, lw_scale=1.0, nbins=18,
               xtick_rot=20.0):
    """Draw one ISO/E3 fold-change panel.

    The keyword arguments override typography/geometry for the main-figure row
    layout; with all of them left at their defaults this renders exactly as the
    2x2 grid and the individual panel exports always have."""
    F = fonts or {}
    f_title = F.get("title", cfg.FONT["title"])
    f_ylab = F.get("ylabel", cfg.FONT["label"])
    f_tick = F.get("tick", cfg.FONT["tick"])
    f_pval = F.get("pval", cfg.STATS.get("pval_fontsize", cfg.FONT["legend"]))
    t_pad = F.get("title_pad", cfg.FONT["title_pad"])

    if not P["box_data"]:
        ax.set_title(f"{title} (no data)", fontsize=f_title)
        style_axes(ax)
        return
    draw_prism_box(ax, P["box_data"], P["xpos"], P["colors"],
                   width=box_width, jitter=jitter, dot_size=dot_size,
                   lw_scale=lw_scale)
    ax.axhline(ref, color="0.4", ls="--", lw=0.8 * lw_scale, zorder=0)
    for sx in P["seps"]:
        ax.axvline(sx, color="0.5", ls=":", lw=1.0 * lw_scale, zorder=0)
    ax.set_xticks(P["xpos"])
    ax.set_xticklabels(P["labels"], rotation=xtick_rot, ha="right",
                       rotation_mode="anchor", fontsize=f_tick)
    ax.set_ylabel(ylab, fontsize=f_ylab)
    ax.set_title(title, fontsize=f_title, pad=t_pad)
    ax.set_ylim(*ylim)
    style_axes(ax)
    ax.tick_params(axis="y", labelsize=f_tick)
    apply_dense_yticks(ax, nbins=nbins)
    yr = ylim[1] - ylim[0]
    dec = cfg.STATS.get("pval_decimals", 3)
    for xp, val, p in zip(P["xpos"], P["box_data"], P["vs_p"]):
        yt = float(np.nanmax(val)) + 0.03 * yr
        ax.text(xp, yt, f"p{_fmt_p(p, dec)}", ha="center", va="bottom",
                fontsize=f_pval)
    # xmargin may be a scalar (symmetric) or (left, right); see the sibling
    # script -- the wide left pad keeps the longest p-string off the y-ticks.
    if isinstance(xmargin, (tuple, list)):
        xl, xr = float(xmargin[0]), float(xmargin[1])
        x0, x1 = float(min(P["xpos"])), float(max(P["xpos"]))
        sp = (x1 - x0) or 1.0
        ax.set_xlim(x0 - xl * sp, x1 + xr * sp)
    else:
        ax.margins(x=xmargin)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    for k, path in DATASETS.items():
        if not path.exists():
            print(f"[ERROR] missing table for {k}: {path}")
            sys.exit(1)
        data[k] = pd.read_csv(path)
    print(f"Loaded datasets: {list(data)}")

    # ref = 0 works for both transforms: log2(1) = 0 and (E3 - E3) = 0.
    ref = 0.0

    # uniform per-metric pseudocount (pooled across bars & cell classes)
    eps_by_metric = {}
    for (mlab, mkey, mcol, need_eps) in METRICS:
        if not need_eps:
            eps_by_metric[mkey] = 0.0
            continue
        pooled = np.concatenate([
            _vals(data[dkey], geno, c, cc, mcol)
            for (_l, _col, dkey, geno, drug, veh) in BARS
            for c in (drug, veh) for _, cc in CELLCLASSES])
        pos = pooled[pooled > 0]
        eps_by_metric[mkey] = 0.5 * float(pos.min()) if pos.size else 0.0
    print("pseudocount per metric:", eps_by_metric)

    # build all panels, then per-column shared ylim (with p-text headroom)
    panels = {}   # (clab, mlab) -> P
    for clab, cclass in CELLCLASSES:
        for mlab, mkey, mcol, _ in METRICS:
            panels[(clab, mlab)] = build_panel(data, cclass, mcol,
                                               eps_by_metric[mkey],
                                               mode=FC_MODE.get(mkey, "log2"),
                                               mkey=mkey)
    col_ylim = {}
    for mlab, mkey, mcol, _ in METRICS:
        allv = np.concatenate([v for clab, _ in CELLCLASSES
                               for v in panels[(clab, mlab)]["box_data"]]
                              or [np.array([0.0])])
        lo, hi = float(np.min(allv)), float(np.max(allv))
        rng = (hi - lo) or 1.0
        col_ylim[mlab] = (lo - 0.06 * rng, hi + 0.18 * rng)

    # ---- combined 2x2 ----
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 8.6))
    for i, (clab, cclass) in enumerate(CELLCLASSES):
        for j, (mlab, mkey, mcol, _) in enumerate(METRICS):
            draw_panel(axes[i][j], panels[(clab, mlab)], col_ylim[mlab], ref,
                       fc_ylab(mkey), panel_title(mkey, cclass))
    fig.suptitle("ISO vs E3 effect: WT | MIC vs piezo-crispant.  Intensity: "
                 "log2 fold change;  events: difference (ISO - E3)",
                 fontsize=cfg.FONT["title"] + 1)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT_DIR / "main_iso_foldchange_2x2.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "main_iso_foldchange_2x2.svg", bbox_inches="tight")
    plt.close(fig)

    # ---- individual panels (svg + pdf editable text, + png) ----
    for clab, cclass in CELLCLASSES:
        for mlab, mkey, mcol, _ in METRICS:
            f1, a1 = plt.subplots(figsize=(3.4, 4.2))
            draw_panel(a1, panels[(clab, mlab)], col_ylim[mlab], ref,
                       fc_ylab(mkey), panel_title(mkey, cclass))
            f1.tight_layout()
            stem = OUT_DIR / f"panel_{clab}-{mlab}"
            for ext in ("svg", "pdf", "png"):
                f1.savefig(stem.with_suffix("." + ext),
                           dpi=200 if ext == "png" else None, bbox_inches="tight")
            plt.close(f1)

    # ---- workbook rows: per-bar folds + all between-group comparisons ----
    # ---- MAIN FIGURE: 1x4 row (all four panels in one figure) ----
    # Rebuilt with tighter x-geometry so the bars fill the panel instead of
    # sitting as thin slivers in a wide, mostly-empty axis.
    row_panels = {}
    for clab, cclass in CELLCLASSES:
        for mlab, mkey, mcol, _ in METRICS:
            row_panels[(clab, mlab)] = build_panel(
                data, cclass, mcol, eps_by_metric[mkey],
                mode=FC_MODE.get(mkey, "log2"),
                x_step=ROW_X_STEP, x_gap=ROW_X_GAP, mkey=mkey)

    row_ylim = {}
    for mlab, mkey, mcol, _ in METRICS:
        allv = np.concatenate([v for clab, _ in CELLCLASSES
                               for v in row_panels[(clab, mlab)]["box_data"]]
                              or [np.array([0.0])])
        lo, hi = float(np.min(allv)), float(np.max(allv))
        rng_ = (hi - lo) or 1.0
        row_ylim[mlab] = (lo - ROW_YPAD_BOT * rng_, hi + ROW_YPAD_TOP * rng_)

    # left-to-right: E-Amp, E-Events, R-Amp, R-Events
    row_order = [(clab, cclass, mlab, mkey)
                 for clab, cclass in CELLCLASSES
                 for mlab, mkey, mcol, _ in METRICS]

    figr, axesr = plt.subplots(1, len(row_order), figsize=ROW_FIGSIZE)
    for axr, (clab, cclass, mlab, mkey) in zip(np.atleast_1d(axesr), row_order):
        draw_panel(axr, row_panels[(clab, mlab)], row_ylim[mlab], ref,
                   fc_ylab_short(mkey), panel_title(mkey, cclass),
                   fonts=ROW_FONT, box_width=ROW_BOX_WIDTH,
                   xmargin=ROW_XMARGIN, jitter=ROW_JITTER,
                   dot_size=ROW_DOT_SIZE, lw_scale=ROW_LW_SCALE,
                   nbins=ROW_YTICK_NBINS, xtick_rot=ROW_XTICK_ROT)
    figr.tight_layout(pad=0.5, w_pad=2.0)
    for ext in ("svg", "pdf", "png"):
        figr.savefig((OUT_DIR / "main_iso_foldchange_row").with_suffix("." + ext),
                     dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(figr)

    fc_rows, cmp_rows = [], []
    for (clab, mlab), P in panels.items():
        folds = {}
        for row in P["fc"]:
            r = {k: v for k, v in row.items() if k != "_fold"}
            fc_rows.append(r)
            folds[row["group"]] = row["_fold"]
        labs = list(folds)
        for a in range(len(labs)):
            for b in range(a + 1, len(labs)):
                gA, gB = labs[a], labs[b]
                try:
                    p = float(ttest_ind(folds[gA], folds[gB], equal_var=False,
                                        nan_policy="omit").pvalue)
                except Exception:
                    p = float("nan")
                cmp_rows.append(dict(cell_class=clab, metric=mlab,
                                     group1=gA, group2=gB, p_value=p,
                                     significant=bool(np.isfinite(p) and p < 0.05)))
    with pd.ExcelWriter(OUT_DIR / "main_iso_foldchange.xlsx", engine="openpyxl") as xw:
        pd.DataFrame(fc_rows).to_excel(xw, sheet_name="per_group", index=False)
        pd.DataFrame(cmp_rows).to_excel(xw, sheet_name="between_group", index=False)
    pd.DataFrame(fc_rows).to_csv(OUT_DIR / "main_iso_foldchange.csv", index=False)

    print(f"[OK] wrote 1x4 row + 2x2 + 4 individual panels + workbook to {OUT_DIR}")


# =============================================================================
# Variants: the same figure, differing only in what the WT bar's p-value
# treats as its multiple-comparison family. Both are produced, into separate
# folders, so the choice can be made when the paper is assembled rather than
# baked in here.
#
#   holm   - WT p read from the full E3/ISO/BDM run, where ISO and BDM share the
#            E3 control and are Holm-corrected as a family of two.
#   e3iso  - WT p read from the E3-vs-ISO-only subset run, a family of one, so a
#            plain Welch t-test.
#
# MIC and Piezo are identical in both: they come from the genotype x drug 2x2
# and always use its Tukey HSD.
# =============================================================================
VARIANTS = [
    dict(label="holm  (WT p Holm-corrected within E3 family)",
         out_key="ISO_MIC_Piezo",       stats_subtree={}),
    dict(label="e3iso (WT p from the E3-vs-ISO-only subset)",
         out_key="ISO_MIC_Piezo_E3ISO",
         stats_subtree={"ISO": "_py_out_%smin_E3_ISO" % WSUF}),
]


def run_variant(out_key: str, stats_subtree: dict, label: str) -> None:
    g = globals()
    g["OUT_DIR"] = MAIN_FIG_DIRS[out_key]
    g["STATS_SUBTREE"] = dict(stats_subtree)
    _STATS_CACHE.clear()          # p-values are cached per dataset key
    print(chr(10) + "=== variant: " + str(label) + " ===")
    main()


if __name__ == "__main__":
    for _v in VARIANTS:
        run_variant(_v["out_key"], _v["stats_subtree"], _v["label"])
