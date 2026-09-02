"""Recompute the quantities a Results paragraph states about Fig 2c-h.

For each statement - group difference in nuclear area over a window, the
direction of the wild-type trend, the change across an hpf boundary, the
difference in nuclei count - this reports n, the group means and the test
result, computed directly from the per-movie tables. Run it against two
configs to see the same quantities from two results trees.
"""
import io, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu

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


def load(root):
    return pd.read_csv(root / "_group_compare" / "aligned_time_series_per_movie.csv")


def emb(a, col, lo, hi):
    """One value per embryo: its mean over [lo, hi] hpf."""
    w = a[(a.hpf >= lo) & (a.hpf <= hi)]
    return w.groupby(["group", "movie"])[col].mean().reset_index()


for tag, root in (("SUBMITTED (results/)", OLD), ("NEW (results_midline_v3/)", NEW)):
    a = load(root)
    print("=" * 70)
    print(tag)

    print('\nCLAIM: "nuclear area is significantly larger than wild type '
          'between 13 and 14 hpf"')
    e = emb(a, "median_area_um2", 13.0, 14.0)
    w = e.loc[e.group == "WT", "median_area_um2"].dropna()
    u = e.loc[e.group == "MUT", "median_area_um2"].dropna()
    print("   WT  n=%d  mean %.1f um2   %s" % (len(w), w.mean(), np.round(w.values, 1)))
    print("   MUT n=%d  mean %.1f um2   %s" % (len(u), u.mean(), np.round(u.values, 1)))
    print("   Welch p = %.4g | Mann-Whitney p = %.4g"
          % (ttest_ind(u, w, equal_var=False).pvalue,
             mannwhitneyu(u, w).pvalue))

    print('\nCLAIM: "in wild-type there is a CONTINUOUS increase in nuclear area"')
    wt = a[a.group == "WT"]
    piv = wt.pivot_table(index="hpf", columns="movie", values="median_area_um2")
    full = piv.columns[piv.notna().all()]        # movies covering the whole axis
    print("   movies covering all of 13-17 hpf: %d of %d" % (len(full), piv.shape[1]))
    m_all = piv.mean(axis=1)
    print("   group mean 13.0 -> 17.0 : %.1f -> %.1f um2 (all movies, n varies)"
          % (m_all.iloc[0], m_all.iloc[-1]))
    if len(full):
        m_fix = piv[full].mean(axis=1)
        print("   same, FIXED subset      : %.1f -> %.1f um2 (n=%d throughout)"
              % (m_fix.iloc[0], m_fix.iloc[-1], len(full)))
    d = m_all.diff().dropna()
    print("   monotonic? %d of %d steps go down (largest drop %.2f um2)"
          % (int((d < 0).sum()), len(d), d.min()))
    per = {}
    for mv in piv.columns:
        s = piv[mv].dropna()
        per[mv] = s.iloc[-1] - s.iloc[0]
    print("   per-movie net change: %d of %d increase"
          % (sum(v > 0 for v in per.values()), len(per)))

    print('\nCLAIM: "significant reduction in fli1+ ECs between 14 and 17 hpf"')
    for lo, hi in ((13.0, 14.0), (14.0, 15.0), (15.0, 16.0), (16.0, 17.0),
                   (14.0, 17.0)):
        e = emb(a, "n_tracked", lo, hi)
        w = e.loc[e.group == "WT", "n_tracked"].dropna()
        u = e.loc[e.group == "MUT", "n_tracked"].dropna()
        print("   %.0f-%.0f hpf : WT %5.1f  MUT %5.1f  Welch p = %.4g  (n=%d/%d)"
              % (lo, hi, w.mean(), u.mean(),
                 ttest_ind(w, u, equal_var=False).pvalue, len(w), len(u)))

    print('\nCLAIM: "decreasing sharply between 14 and 15 hpf" (mutant area)')
    pre = emb(a, "median_area_um2", 13.8, 14.2)
    post = emb(a, "median_area_um2", 14.8, 15.2)
    m = pre.merge(post, on=["group", "movie"], suffixes=("_pre", "_post"))
    m["d"] = m.median_area_um2_post - m.median_area_um2_pre
    for g in ("WT", "MUT"):
        s = m.loc[m.group == g, "d"]
        print("   %-4s change 14.0 -> 15.0 : %+.1f um2 mean, %d/%d decrease"
              % (g, s.mean(), int((s < 0).sum()), len(s)))
    print("   MUT vs WT change, Welch p = %.4g"
          % ttest_ind(m.loc[m.group == "MUT", "d"],
                      m.loc[m.group == "WT", "d"], equal_var=False).pvalue)
    print()
