# tests/test_stats.py
# -*- coding: utf-8 -*-
"""
Regression tests for the load-bearing pure-function statistics/math.

Run either way:
    python -m pytest tests/test_stats.py        # if pytest is installed
    python tests/test_stats.py                  # plain runner (no pytest needed)

Uses the calcium_gcamp env (numpy/pandas/scipy/statsmodels). Tests touching the
two-way ANOVA / Tukey skip cleanly when statsmodels is absent.
"""
from __future__ import annotations

import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path

try:
    import pytest
except Exception:  # plain-runner mode
    pytest = None

from calcium_build_tables import (
    robust_sigma, safe_div, baseline_from_pre,
    detect_events_single_frame, per_peak_duration_local_valley,
)
from plot_Q2_cells_events import (
    holm, twoway_anova_label, tukey_significant_pairs, _HAVE_SM,
)

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
except Exception:
    pairwise_tukeyhsd = None


def _skip(msg: str) -> bool:
    if pytest is not None:
        pytest.skip(msg)
    print(f"   SKIP: {msg}")
    return True


# --------------------------------------------------------------------------- #
# build_tables primitives (hand-computed expectations)
# --------------------------------------------------------------------------- #

def test_robust_sigma():
    # median=3, |dev|=[2,1,0,1,97] -> MAD=1 -> sigma=1.4826
    assert np.isclose(robust_sigma(np.array([1, 2, 3, 4, 100.0])), 1.4826)
    # MAD==0 (constant) -> falls back to std (0)
    assert np.isclose(robust_sigma(np.array([5, 5, 5, 5.0])), 0.0)


def test_safe_div():
    assert np.all(np.isnan(safe_div(np.array([1.0, 2]), -3.0)))   # negative denom
    assert np.all(np.isnan(safe_div(np.array([1.0, 2]), 0.0)))    # zero denom
    assert np.allclose(safe_div(np.array([2.0, 4]), 2.0), [1, 2])
    r = safe_div(np.array([2.0, 4.0, 6.0]), np.array([2.0, -1.0, 0.0]))
    assert np.isclose(r[0], 1.0) and np.isnan(r[1]) and np.isnan(r[2])


def test_baseline_from_pre():
    assert np.isclose(baseline_from_pre(np.array([1, 2, np.nan, 4.0])), 2.0)


def test_detect_events():
    sig = np.ones(30)
    for i in (5, 15, 25):
        sig[i] = 5.0
    ev = detect_events_single_frame(sig, external_threshold=2.0)
    assert list(np.where(ev)[0]) == [5, 15, 25]
    # sub-threshold bump rejected
    s2 = np.ones(20); s2[10] = 1.5
    assert detect_events_single_frame(s2, external_threshold=2.0).sum() == 0


def test_per_peak_duration_fwhm():
    # triangle peak idx3, valleys 0 -> half=2; strict >half gives width 1 frame
    tr = np.array([0, 1, 2, 4, 2, 1, 0.0])
    d = per_peak_duration_local_valley(tr, np.array([3]), dt_min=0.5)
    assert np.isclose(d[0], 0.5)
    # wider peak above half -> 3 frames
    tr2 = np.array([0, 1, 3, 4, 3, 1, 0.0])
    d2 = per_peak_duration_local_valley(tr2, np.array([3]), dt_min=0.5)
    assert np.isclose(d2[0], 1.5)


# --------------------------------------------------------------------------- #
# holm (term-by-term)
# --------------------------------------------------------------------------- #

def test_holm_stepdown_monotonic():
    # p=[0.01,0.04,0.03], m=3 -> [0.03, 0.06, 0.06] (cummax enforces monotonicity)
    assert np.allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])
    assert np.allclose(holm(np.array([0.7])), [0.7])          # single p, capped at 1
    assert np.allclose(holm(np.array([0.5, 0.9])), [1.0, 1.0])  # cap at 1


# --------------------------------------------------------------------------- #
# two-way ANOVA label: guards + plumbing
# --------------------------------------------------------------------------- #

def _twoway_df(means: dict, n=5, sd=0.15, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for (g, c), mu in means.items():
        for _ in range(n):
            rows.append({"value": float(mu + rng.normal(0, sd)), "genotype": g, "condition": c})
    return pd.DataFrame(rows)


def test_twoway_guards():
    base = _twoway_df({("MIC", "E3"): 1, ("MIC", "ISO"): 2,
                       ("Piezo", "E3"): 1, ("Piezo", "ISO"): 1.2})
    # single genotype -> not applicable (label, stats, ok)
    _, stats, ok = twoway_anova_label(base[base.genotype == "MIC"], "value")
    assert ok is False and stats is None
    # single condition -> not applicable
    _, stats, ok = twoway_anova_label(base[base.condition == "E3"], "value")
    assert ok is False and stats is None
    # empty cell -> explicit "empty" reason, not estimable
    lab, stats, ok = twoway_anova_label(
        base[~((base.genotype == "MIC") & (base.condition == "ISO"))], "value")
    assert ok is False and stats is None and "empty" in lab.lower()


def test_twoway_label_wellformed_and_eta2_bounds():
    if not _HAVE_SM and _skip("statsmodels unavailable"):
        return
    df = _twoway_df({("MIC", "E3"): 1, ("MIC", "ISO"): 2.2,
                     ("Piezo", "E3"): 1, ("Piezo", "ISO"): 1.2})
    lab, stats, ok = twoway_anova_label(df, "value")
    # figure label is now just the short header; numbers live in `stats`
    assert ok and lab == "2-way ANOVA (II)"
    etas = [stats["geno_eta2p"], stats["drug_eta2p"], stats["gxd_eta2p"]]
    assert all(0.0 <= e <= 1.0 for e in etas)
    assert all(0.0 <= stats[k] <= 1.0 for k in ("geno_p", "drug_p", "gxd_p"))


def test_twoway_interaction_detected():
    """Partial eta^2 for the interaction should be far larger when an
    interaction is planted than when effects are purely additive."""
    if not _HAVE_SM and _skip("statsmodels unavailable"):
        return
    def inter_eta(means):
        _lab, stats, ok = twoway_anova_label(_twoway_df(means), "value")
        assert ok
        return float(stats["gxd_eta2p"])
    strong = inter_eta({("MIC", "E3"): 1.0, ("MIC", "ISO"): 2.2,
                        ("Piezo", "E3"): 1.0, ("Piezo", "ISO"): 1.2})   # blunted -> interaction
    additive = inter_eta({("MIC", "E3"): 1.0, ("MIC", "ISO"): 2.0,
                          ("Piezo", "E3"): 1.5, ("Piezo", "ISO"): 2.5})  # parallel -> no interaction
    assert strong > additive
    assert strong > 0.3 and additive < 0.2


# --------------------------------------------------------------------------- #
# Tukey: attribute-based extraction matches summary(); separation is flagged
# --------------------------------------------------------------------------- #

def test_tukey_matches_summary_and_flags_difference():
    if not _HAVE_SM and _skip("statsmodels unavailable"):
        return
    rng = np.random.default_rng(1)
    rows = []
    for lbl, mu in {"A": 1.0, "B": 1.05, "C": 5.0}.items():   # C clearly separated
        for _ in range(6):
            rows.append({"value": float(mu + rng.normal(0, 0.1)), "cell": lbl})
    df = pd.DataFrame(rows)

    got = tukey_significant_pairs(df, "value", "cell")

    # independent reference: parse summary() by HEADER NAME (not position)
    res = pairwise_tukeyhsd(df["value"].to_numpy(float), df["cell"].to_numpy())
    tbl = res.summary().data
    hdr = [str(h) for h in tbl[0]]
    ig1, ig2 = hdr.index("group1"), hdr.index("group2")
    ip, irej = hdr.index("p-adj"), hdr.index("reject")
    ref = {(str(r[ig1]), str(r[ig2])): (float(r[ip]), bool(r[irej])) for r in tbl[1:]}

    got_map = {(a, b): (round(p, 6), rej) for (a, b, p, rej) in got}
    assert len(got_map) == len(ref)
    for key, (p, rej) in ref.items():
        # our function may list a pair in the opposite group order; accept either
        cand = got_map.get(key) or got_map.get((key[1], key[0]))
        assert cand is not None, f"pair {key} missing from output"
        assert np.isclose(cand[0], round(p, 6)) and cand[1] == rej

    # A vs C and B vs C must be significant; A vs B must not
    sig = {(a, b): rej for (a, b, p, rej) in got}
    def rej_of(x, y):
        return sig.get((x, y), sig.get((y, x)))
    assert rej_of("A", "C") and rej_of("B", "C")
    assert rej_of("A", "B") is False


# --------------------------------------------------------------------------- #
# plain runner (no pytest)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    tests = sorted(k for k, v in globals().items() if k.startswith("test_") and callable(v))
    failed = 0
    for name in tests:
        try:
            globals()[name]()
            print(f"  [PASS] {name}")
        except Exception as e:   # includes pytest.skip when pytest absent it returns
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(tests)-failed}/{len(tests)} passed.")
    sys.exit(1 if failed else 0)
