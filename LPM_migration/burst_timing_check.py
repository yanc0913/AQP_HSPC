"""Timing of nuclear fragmentation, and the per-movie traces behind it.

Three views, written as one figure plus the tables behind it:
  - every movie's nuclear-area trace on its own, which the group mean+-SD
    plot necessarily hides
  - how many movies contribute at each hpf, since movies cover different
    developmental windows
  - when fragmentation events occur, as a distribution over hpf per group

Also reports, per movie, the change in nuclear area across a chosen hpf
boundary, so a claim about when the change happens can be checked against
individual embryos rather than the group curve.
"""
import io, sys, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, ttest_ind

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

_ARGS = _args()
NEW = _results_root(_ARGS.config)
OUT = NEW / "_group_compare" / "_burst_timing_check"
OUT.mkdir(parents=True, exist_ok=True)
C_WT, C_MUT = "#5B6B7B", "#D6216B"

a = pd.read_csv(NEW / "_group_compare" / "aligned_time_series_per_movie.csv")
piv = a.pivot_table(index="hpf", columns=["group", "movie"], values="median_area_um2")

# ---- pre/post change per movie -------------------------------------------
rows = []
for (g, m) in piv.columns:
    s = piv[(g, m)]
    pre = s[(s.index >= 13.8) & (s.index <= 14.4)].dropna()
    post = s[(s.index >= 14.6) & (s.index <= 15.2)].dropna()
    if len(pre) and len(post):
        rows.append(dict(group=g, movie=m, delta=post.mean() - pre.mean()))
d = pd.DataFrame(rows)
mut_dn = int((d.loc[d.group == "MUT", "delta"] < 0).sum())
wt_dn = int((d.loc[d.group == "WT", "delta"] < 0).sum())
n_mut = int((d.group == "MUT").sum())
n_wt = int((d.group == "WT").sum())
odds, p_fish = fisher_exact([[mut_dn, n_mut - mut_dn], [wt_dn, n_wt - wt_dn]])
p_mw = mannwhitneyu(d.loc[d.group == "MUT", "delta"],
                    d.loc[d.group == "WT", "delta"]).pvalue
print("area change across 14.5 hpf (post 14.6-15.2 minus pre 13.8-14.4)")
print("  MUT %d/%d shrink, WT %d/%d shrink" % (mut_dn, n_mut, wt_dn, n_wt))
print("  Fisher exact p = %.4g ; Mann-Whitney on the change p = %.4g"
      % (p_fish, p_mw))
print("  mean change  MUT %+.1f um2   WT %+.1f um2"
      % (d.loc[d.group == "MUT", "delta"].mean(),
         d.loc[d.group == "WT", "delta"].mean()))

# ---- burst timing ---------------------------------------------------------
pool, per = [], []
for grp in ("WT", "MUT"):
    for mdir in sorted((NEW / grp).iterdir()):
        if not mdir.is_dir():
            continue
        be, ms = mdir / "trackpy" / "burst_events.csv", mdir / "trackpy" / "movie_summary.csv"
        if not (be.exists() and ms.exists()):
            continue
        nf = int(pd.read_csv(ms).iloc[0]["n_frames"])
        s0, s1 = [float(v) for v in
                  re.search(r"(\d+(?:\.\d+)?)[-_](\d+(?:\.\d+)?)\s*hpf",
                            mdir.name, re.I).groups()]
        b = pd.read_csv(be)
        hp = (s0 + b["break_frame"].to_numpy(float) * (s1 - s0) / max(1, nf - 1)
              if len(b) else np.array([]))
        pool += [(grp, float(v)) for v in hp]
        per.append(dict(group=grp, movie=mdir.name, n=len(hp), n_frames=nf,
                        per_frame=len(hp) / nf))
ab = pd.DataFrame(pool, columns=["group", "hpf"])
pf = pd.DataFrame(per)
print("\nburst events")
for g in ("WT", "MUT"):
    s = ab.loc[ab.group == g, "hpf"]
    print("  %-4s n=%3d  median %.2f hpf  IQR %.2f-%.2f  | per frame per movie %.2f"
          % (g, len(s), s.median(), s.quantile(.25), s.quantile(.75),
             pf.loc[pf.group == g, "per_frame"].mean()))
print("  timing WT vs MUT, Mann-Whitney p = %.3g"
      % mannwhitneyu(ab.loc[ab.group == "WT", "hpf"],
                     ab.loc[ab.group == "MUT", "hpf"]).pvalue)
print("  rate (events/frame/movie) Welch p = %.4g"
      % ttest_ind(pf.loc[pf.group == "MUT", "per_frame"],
                  pf.loc[pf.group == "WT", "per_frame"], equal_var=False).pvalue)

# ---- figure ---------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

for (g, m) in piv.columns:
    s = piv[(g, m)].dropna()
    ax[0].plot(s.index, s.values, lw=1.1, alpha=.75,
               color=C_MUT if g == "MUT" else C_WT)
ax[0].axvline(14.5, color="k", ls="--", lw=1)
ax[0].set_xlabel("Time (hpf)")
ax[0].set_ylabel(u"Median tracked nuclear area (\u00b5m\u00b2)")
ax[0].set_title("every movie separately\n(the mean\u00b1SD plot hides this)", fontsize=10)
ax[0].plot([], [], color=C_WT, label="wild type (n=7)")
ax[0].plot([], [], color=C_MUT, label="aqp1a1-/- (n=6)")
ax[0].legend(fontsize=8, frameon=False)

nn = (a.dropna(subset=["median_area_um2"]).groupby(["group", "hpf"])
      .movie.nunique().unstack(0).fillna(0))
for g, c in (("WT", C_WT), ("MUT", C_MUT)):
    ax[1].step(nn.index, nn[g], where="mid", color=c, lw=2, label=g)
ax[1].axvline(14.5, color="k", ls="--", lw=1)
ax[1].set_ylim(0, 8)
ax[1].set_xlabel("Time (hpf)")
ax[1].set_ylabel("movies contributing")
ax[1].set_title("how many movies are averaged\nat each time point", fontsize=10)
ax[1].legend(fontsize=8, frameon=False)

bins = np.arange(13, 17.25, .25)
for g, c in (("WT", C_WT), ("MUT", C_MUT)):
    s = ab.loc[ab.group == g, "hpf"]
    ax[2].hist(s, bins=bins, density=True, histtype="step", lw=1.8, color=c,
               label="%s (n=%d events)" % (g, len(s)))
ax[2].axvline(14.5, color="k", ls="--", lw=1)
ax[2].set_xlabel("Time (hpf)")
ax[2].set_ylabel("fraction of burst events")
ax[2].set_title("burst TIMING is the same in both;\nonly the RATE differs", fontsize=10)
ax[2].legend(fontsize=8, frameon=False)

for x in ax:
    x.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(OUT / ("burst_timing_check.%s" % ext), dpi=200, bbox_inches="tight")
plt.close(fig)

d.to_csv(OUT / "per_movie_area_change.csv", index=False)
ab.to_csv(OUT / "burst_events_hpf.csv", index=False)
pf.to_csv(OUT / "burst_rate_per_movie.csv", index=False)
print("\nwrote %s" % OUT)
