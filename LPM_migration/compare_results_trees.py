"""Panel-by-panel comparison of two results trees.

Overlays the Figure 2 panels produced by two runs and reports how far each
curve moved and how each printed p-value changed. Useful when a parameter
changes and the effect on every panel has to be seen at once rather than
inferred.

Panels c-h map onto compare_groups.py outputs:
  c  speed_vs_time        d  boxplot_speed             e  area_vs_time
  f  bursts_vs_time       g  boxplot_burst_track_ratio h  n_tracked_vs_time
"""
import io, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import ttest_ind

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

def _results_root(cfg_path):
    """The results tree a config points at.

    Never hardcode it: these scripts are run against whichever tree is of
    interest, and result folders get renamed."""
    import yaml
    with io.open(cfg_path, encoding="utf-8") as f:
        return Path(yaml.safe_load(f)["project"]["results_root"])


def _args(*extra):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_midline_v3.yaml",
                    help="config whose results_root holds the run to analyse")
    for name, default, helptext in extra:
        ap.add_argument(name, default=default, help=helptext)
    return ap.parse_args()

_ARGS = _args(("--baseline_config", "config.yaml",
               "config whose results_root holds the run to compare against"))
OLD = _results_root(_ARGS.baseline_config)
NEW = _results_root(_ARGS.config)
OUT = NEW / "_group_compare" / "_tree_comparison"
OUT.mkdir(parents=True, exist_ok=True)
C_WT, C_MUT = "#5B6B7B", "#D6216B"

ao = pd.read_csv(OLD / "_group_compare" / "aligned_time_series_per_movie.csv")
an = pd.read_csv(NEW / "_group_compare" / "aligned_time_series_per_movie.csv")
so = pd.read_csv(OLD / "_group_compare" / "all_movies_summary.csv")
sn = pd.read_csv(NEW / "_group_compare" / "all_movies_summary.csv")


def ms(df, col):
    g = df.groupby(["group", "hpf"])[col].agg(["mean", "std", "count"])
    return g.reset_index()


PANELS = [
    ("c", "median_speed_um_per_min", "Median speed (um/min)"),
    ("e", "median_area_um2",         "Median nuclear area (um2)"),
    ("f", "n_bursts",                "# nuclear fragmentation"),
    ("h", "n_tracked",               "# nuclei"),
]
BOX = [
    ("d", "mean_median_speed_um_per_min", "Av. median speed per embryo"),
    ("g", "burst_per_track_ratio",        "fragmentation / tracked nuclei"),
]

print("=== time-series panels: how much do the group-mean curves move? ===")
for tag, col, lab in PANELS:
    o, n = ms(ao, col), ms(an, col)
    m = o.merge(n, on=["group", "hpf"], suffixes=("_o", "_n"))
    print("  panel %s  %s" % (tag, lab))
    for g in ("WT", "MUT"):
        d = m[m.group == g]
        diff = d["mean_n"] - d["mean_o"]
        rel = 100 * diff / d["mean_o"].replace(0, np.nan)
        print("    %-4s mean shift %+.3f (median %+.3f, |max| %.3f) | %+.1f%% median"
              % (g, diff.mean(), diff.median(), diff.abs().max(),
                 rel.median()))

print("\n=== boxplot panels: the printed p-values ===")
for tag, col, lab in BOX:
    print("  panel %s  %s" % (tag, lab))
    for name, s in (("baseline", so), ("current", sn)):
        w = s.loc[s.group == "WT", col]
        u = s.loc[s.group == "MUT", col]
        print("    %-9s WT %.4g  MUT %.4g   Welch p = %.4g"
              % (name, w.mean(), u.mean(),
                 ttest_ind(w, u, equal_var=False).pvalue))

print("\n=== per-embryo values behind panel g (the one that carries the claim) ===")
t = (so[["group", "movie", "burst_per_track_ratio"]]
     .merge(sn[["group", "movie", "burst_per_track_ratio"]],
            on=["group", "movie"], suffixes=("_baseline", "_current")))
print(t.to_string(index=False, float_format=lambda v: "%.4f" % v))

# ---------------- overlay figure ----------------
fig, axs = plt.subplots(2, 3, figsize=(15, 7.5))
order = [("c", "median_speed_um_per_min", "Median speed (um/min)"),
         ("e", "median_area_um2", "Median nuclear area (um$^2$)"),
         ("f", "n_bursts", "# nuclear fragmentation"),
         ("h", "n_tracked", "# nuclei")]
for ax, (tag, col, lab) in zip(axs.ravel()[:4], order):
    for df, style, tagname in ((ao, dict(ls="--", lw=1.3, alpha=.85), "baseline"),
                               (an, dict(ls="-", lw=2.0), "current")):
        g = ms(df, col)
        for grp, c in (("WT", C_WT), ("MUT", C_MUT)):
            d = g[g.group == grp]
            ax.plot(d.hpf, d["mean"], color=c, **style,
                    label="%s %s" % (grp, tagname))
    ax.set_title("panel %s  -  %s" % (tag, lab), fontsize=10)
    ax.set_xlabel("Time (hpf)")
    ax.legend(fontsize=7, frameon=False, ncol=2)

for ax, (tag, col, lab) in zip(axs.ravel()[4:], BOX):
    pos, ticks = [], []
    for i, (name, s) in enumerate((("baseline", so), ("current", sn))):
        for j, (grp, c) in enumerate((("WT", C_WT), ("MUT", C_MUT))):
            v = s.loc[s.group == grp, col].to_numpy(float)
            x = i * 2.6 + j
            bp = ax.boxplot([v], positions=[x], widths=.6, patch_artist=True,
                            showfliers=False)
            bp["boxes"][0].set(facecolor=c, alpha=.35, edgecolor=c)
            for k in ("whiskers", "caps", "medians"):
                for a_ in bp[k]:
                    a_.set(color=c)
            ax.plot(np.full(len(v), x) + np.random.uniform(-.12, .12, len(v)),
                    v, "o", ms=4, color=c)
            pos.append(x)
            ticks.append("%s\n%s" % (grp, name))
        w = s.loc[s.group == "WT", col]
        u = s.loc[s.group == "MUT", col]
        p = ttest_ind(w, u, equal_var=False).pvalue
        ax.text(i * 2.6 + .5, ax.get_ylim()[1], "p = %.4g" % p,
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(pos)
    ax.set_xticklabels(ticks, fontsize=7)
    ax.set_title("panel %s  -  %s" % (tag, lab), fontsize=10)

for ax in axs.ravel():
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("baseline tree (dashed / left boxes) vs current tree "
             "(solid / right boxes)", fontsize=11)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(OUT / ("tree_comparison.%s" % ext), dpi=200, bbox_inches="tight")
plt.close(fig)
t.to_csv(OUT / "panel_g_per_embryo.csv", index=False)
print("\nwrote %s" % OUT)
