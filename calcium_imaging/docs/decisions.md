# Calcium Imaging Analysis — Decision Log

This document records why the analysis pipeline is built the way it is.
It is a companion to `README.md`: README describes _what_ the code does;
this document explains _why_ specific design choices were made and what
alternatives were considered.

Read this when:
- You're about to change a parameter and want to know its history
- Someone (reviewer, collaborator, future-you) asks "why did you...?"
- You're writing the methods section of a paper

---

## 1. Pipeline overview

Input: Fiji-exported per-frame CSVs (one per movie) with columns
`channel`, `roi_name`, `frame`, `mean_bgsub`, `max_bgsub`, etc.

Output: a set of analysis tables in `_py_out_<W>min/tables/` plus plots
in `_py_out_<W>min/plots_{png,svg}/`, where `<W>` is the post analysis
window in minutes (default 10).

Three analysis "Qs":
- **Q1**: vDA/dDA ratio over time (E3 control only)
- **Q2**: single-cell event metrics (across all conditions)
- **Q3**: vDA band trace (across all conditions)

---

## 2. Normalisation pipeline

The single-cell trace `cell_trace_main` is built in three stages from the
raw Fiji output:

```
ImageJ raw fluorescence  →  background subtracted (mean_bgsub, max_bgsub)
       (Fiji)                                          ↓
                                            cell_signal_raw = max_bgsub
                                                              ↓
                                       × M (Lifeact-based intensity correction)
                                                              ↓
                                                  cell_signal_corr
                                                              ↓
                                           ÷ median(cell_signal_corr_pre)
                                                              ↓
                                                  cell_trace_main  (F/F₀)
```

### 2.1 Which pixel statistic drives cells and bands?

**Current decision: `mean_bgsub` for BOTH cells and bands**
(`cfg.CELL_SIGNAL_STAT = "mean"`, the paper main figure). `max_bgsub` is still
computed and kept in the tables, but only as an internal sanity check.

The full rationale lives with the switch itself, in `calcium_config.py` §4 -
in short, `max` is dominated by single-frame artefacts on ROIs that touch
vasculature (flowing bright vesicles, subpixel registration jitter), which
inflated event counts several-fold; `mean` gives the same direction of effect
with comparable significance and is robust to ROI-area drift between phases.

> **Superseded.** This section previously argued the opposite - `max_bgsub`
> for cells (as the more sensitive statistic for a localised transient) and
> `mean_bgsub` for bands, described as an intentional asymmetry. That was the
> original design; the QC described in `calcium_config.py` §4 overturned it.
> Kept here so the change of mind is on the record.

### 2.2 Why `M = median(Lifeact_vDA_pre) / median(Lifeact_vDA_post)`?

Between the pre and post recordings, the embryo is physically removed
from agarose, soaked in drug, and re-mounted. This causes uncontrolled
shifts in overall fluorescence intensity (changes in focal plane, optical
path, embryo orientation).

The Lifeact channel is a structural marker (actin) — its intensity
**should not change due to drug treatment**, only due to mounting
artefacts. So Lifeact provides a per-embryo "reference shift" that we
can use to scale the GCaMP signal back to the pre intensity regime.

Convention: `M = pre / post`, applied as `corrected = post × M`.

The previous code used `M = post / pre` and applied as `post / M`
(mathematically equivalent — the two errors cancel — but reversed
direction made the code harder to read). **The new convention matches
the README and is intuitively right**: M is a "scale factor that brings
post back to pre".

### 2.3 What about photo-bleaching during the post recording?

We do **not** explicitly correct for bleaching during the 30-min post
recording. Reasons:

1. Bleaching is heterogeneous between embryos (observed range: 0.65 to
   0.99 of initial intensity after 30 min).
2. ISO-treated embryos bleach faster than controls, suggesting bleaching
   carries some treatment-related information that we'd lose by
   normalising it out.
3. F1-strict (see §3.2) restricts analysis to the first 10 minutes of
   post, where bleaching is mild (~9% drop on average) — minimising
   exposure to the worst bleaching effects.
4. Using vDA band to correct cell trace would be **double correction**
   (since M already adjusts pre→post intensity).

If post window is increased to 20 or 30 min, the bleaching issue becomes
more serious — revisit this if needed.

### 2.4 Why F/F₀ and not (F − F₀)/F₀?

The code computes `F / F₀` (so pre median ≈ 1.0). Classical ΔF/F₀ is
`(F − F₀) / F₀` (so pre median ≈ 0).

Both are valid; the code's convention is chosen because:
- F values are already background-subtracted upstream (so F itself is
  arguably already "ΔF relative to background").
- The "ratio to baseline" form is easier to read on plots — 1.5 means
  "1.5× pre intensity".
- All thresholds and metrics in the pipeline are designed for the F/F₀
  convention.

This is noted in `calcium_config.py` and the README.

---

## 3. Event detection

### 3.1 Algorithm: shape + locality + global threshold

A frame `t` is flagged as a calcium event when **all three** conditions hold:

1. **Shape**: `x[t-1] < x[t] >= x[t+1]` — a local maximum.
2. **Locality** (`neighbour OR prominence`):
   - Neighbour: `x[t] ≥ 1.10 × max(x[t-1], x[t+1])`
   - Prominence: `x[t] ≥ 1.10 × P20(x[t-2 .. t+2])`
3. **Global**: `x[t] ≥ threshold` (see §3.3 for what threshold).

Adjacent surviving candidates within 1 frame are merged; the highest is
kept. This prevents a single broad transient from being counted as
multiple events.

### 3.2 Why "OR" in (2)?

Calcium signals come in two shapes:
- **Sharp spikes** (1 frame): caught by the neighbour rule.
- **Broader transients** (2-3 frames, all elevated): caught by the
  prominence rule (compares peak to a local low percentile, not adjacent
  frames).

Using `AND` would only detect very sharp peaks — missing real biology.
Using `OR` together with the global threshold (3) gives sensitivity to
both shapes without too many false positives.

### 3.3 Threshold: pre-segment self-reference

**Critical design choice.** The threshold is computed on the **pre phase**
of each cell:

```
threshold = median(cell_trace_main_pre) + k × σ_robust(cell_trace_main_pre)
            where k = 2.0,  σ_robust = 1.4826 × MAD
```

This threshold is then applied to detect peaks in the post window.

**Alternative considered**: compute threshold on the post window itself
(self-referenced). Rejected because:

- When a drug causes sustained activation (e.g. ISO), the post trace is
  uniformly elevated. Self-referenced threshold rises with it, so the
  detector sees "elevated background, occasional very high peaks" and
  reports very few events.
- Tested on real ISO data: self-ref gave 2 events for a cell that
  visually has 5-7 obvious peaks; pre-ref gave 5.
- Pre-segment reference uses the cell's own quiet baseline as "what
  counts as noise" — biologically meaningful.

### 3.4 F1-strict: detect only in the analysis window

Event detection runs **only on the first `POST_ANALYSIS_MIN` minutes** of
post (default 10). Reasons:

- Beyond ~10 min, photo-bleaching is non-negligible (typically 15-30%
  drop), so the trace shape changes for non-biological reasons.
- Including the bleached tail in the threshold computation would lower
  the threshold and inflate the event count in the early part.

If you want to analyse a different window (e.g. 20 min), change
`POST_ANALYSIS_MIN = 20.0` in `calcium_config.py`. Output paths and
column names automatically reflect the new window.

### 3.5 Robust σ vs MAD

The threshold uses `σ_robust = 1.4826 × MAD`, not just MAD. The 1.4826
factor makes σ_robust equivalent to the sample standard deviation under
a Gaussian distribution. This is the standard "robust z-score"
construction (Iglewicz & Hoaglin 1993, "How to detect and handle
outliers").

The README initially wrote `median + 2×MAD` (without the 1.4826) —
that was a documentation error; the code always used the correct
robust σ form.

---

## 4. Per-cell metrics

For each cell, three metrics are computed on the post analysis window:

### 4.1 `n_events_post_<W>min`

Count of peaks detected by the algorithm above.

### 4.2 `mean_amplitude_post_<W>min`

Plain mean of `cell_trace_main` over the post window. Captures
"sustained brightness" independent of peak count. Equivalent to
`(AUC + W) / W` where AUC = ∫(trace - 1) dt; we report mean amplitude
because it has natural units (F/F₀).

This metric is particularly useful for drugs that cause sustained
activation without discrete spikes (where event counts can saturate or
become insensitive).

### 4.3 `mean_peak_duration_post_<W>min`

For each detected peak, compute its FWHM relative to **local valleys**:

1. Find the lowest point between this peak and the previous peak
   (or start of trace).
2. Find the lowest point between this peak and the next peak
   (or end of trace).
3. `local_baseline = mean(left_valley, right_valley)`
4. `half = (peak_value + local_baseline) / 2`
5. Walk outward from peak while `trace > half`; this span is the FWHM.

**Why local valleys, not global baseline?** When the entire post trace
is elevated (e.g. ISO), peaks share a high "floor". Using the global
pre baseline as reference would give all peaks an absurdly long
duration (potentially the entire window). Local valleys give each
peak its own context.

Each cell's mean peak duration is `nanmean(per_peak_durations)`. NaN
if no peaks detected.

---

## 5. Embryo-level aggregation

Per-cell metrics are aggregated to per-embryo by `nanmean` across the
cells within each `(batch, pair, condition, embryo, cell_class)` group.

**Why nanmean and not pooling all cells together?** Each embryo has
1-5 cells; pooling would weight embryos with more cells more heavily.
The embryo is the biological replicate, so we average within embryo
first, then statistics are run across embryos.

---

## 6. Statistics

For each (pair_id, cell_class), boxplots show all conditions side by
side with two layers of statistical annotation:

### 6.1 Omnibus test (top-left of plot)

- **2 groups**: Welch's t-test (`equal_var=False`)
- **≥3 groups**: one-way ANOVA (`f_oneway`)

Auto-dispatch via `cfg.STATS["omnibus_test"] = "auto"`. Can be forced
to one or the other, or disabled entirely.

**Why Welch's t-test for 2 groups instead of ANOVA?** With 2 groups,
ANOVA reduces to t² — the p-value is the same as Student's t (equal
variances assumed). Welch's t-test relaxes the equal-variance assumption,
which is appropriate when between-embryo variance differs by treatment
(common in calcium data).

### 6.2 Pairwise post-hoc

All pairs of conditions are compared with Welch's t-test, p-values
corrected by Holm-Bonferroni.

**Holm implementation**: proper monotonic step-down. The previous
implementation was missing the cumulative-max step (step 3 of the
algorithm), making it anti-conservative on tied p-values. The new
implementation is verified against `statsmodels.stats.multitest.multipletests(method='holm')`.

---

## 7. Drug name handling

### 7.1 Why `CONDITION_ALIASES`?

Different experimenters / different times use slightly different drug
abbreviations in filenames:
- `20uMYoda1`, `Yoda1`, `Yoda`, `yoda` → all should map to canonical `Yoda`
- `GsMTx`, `GsMTx4` → both map to `GsMTx`

The `CONDITION_ALIASES` dict in `calcium_config.py` is a regex-based
mapping (canonical name → patterns). First match wins. Unknown drugs
are kept as-is and reported as warnings.

### 7.2 Why preserve `drug_raw` alongside `condition`?

`condition` is the canonical name (used for grouping, plotting, stats).
`drug_raw` is the original token from the filename (preserved for
traceability — if you ever need to know "which exact files contributed
to this point", grep `drug_raw`).

### 7.3 `DISPLAY_NAMES` for plots

Plots use `cfg.display_name(condition)` for human-readable labels
(e.g. `Yoda` → `Yoda1`, `GsMTx` → `GsMTx4`, `ISO` → `Isoprenaline`).
Internal grouping, file paths, and CSV columns still use the short
canonical name (avoids special chars in paths).

---

## 8. Output structure

Each run with a given `POST_ANALYSIS_MIN` writes to a window-suffixed
directory:

```
_py_out_10min/                       # if POST_ANALYSIS_MIN = 10
_py_out_20min/                       # if POST_ANALYSIS_MIN = 20
```

This means changing the analysis window does NOT overwrite previous
outputs. You can have multiple runs side by side.

Each output directory contains:
- `_config_snapshot.py` — a frozen copy of `calcium_config.py` at the
  time of the run, with a header noting the run datetime. **This is the
  authoritative record of what parameters this output was generated with.**
- `tables/*.csv` — analysis tables
- `plots_png/...`, `plots_svg/...` — figures

---

## 9. Currently unresolved / future work

- **Q3 (vDA trace plot)**: still uses the v5 script (`plot_Q2_DA_v5.py`).
  Will be refactored to use config + display names + new file names.
- **Q1 (vDA/dDA ratio plot)**: still uses v5. Same plan as Q3.
- **`plot_style.py`**: still owns some keys that are duplicated in
  `calcium_config.py` (font sizes, fig sizes, colors). After Q1/Q3 are
  refactored, do a cleanup pass to remove the duplication.
- **Bleaching correction in long-window analyses**: if `POST_ANALYSIS_MIN`
  is increased to 20+ minutes, revisit whether to add an explicit
  detrending step.
