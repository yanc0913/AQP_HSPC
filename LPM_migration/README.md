# LPM migration tracking pipeline

Segmentation → tracking → group comparison for zebrafish **lateral plate
mesoderm (LPM)** nuclei in timelapse movies, comparing wild type against
`aqp1a.1-/-` over 13–17 hpf.

This code produces Figure 2c–h of Kondrychyn *et al.*, *"Osmo-hydraulic
volume regulation ensures robust endothelial-to-haematopoietic transition"*.

> **Code only.** Raw movies and analysis outputs are not in this repository.
> Copy `config.example.yaml` → `config.yaml` (git-ignored) and point it at your
> own data.

## Contents

- [1. Pipeline overview](#1-pipeline-overview)
- [2. Setup](#2-setup)
- [3. Input data](#3-input-data)
- [4. Running](#4-running)
- [5. Methods](#5-methods) — what is a nucleus, the midline, tracking, bursts
- [6. Outputs](#6-outputs) — directory tree, every file, every plot
- [7. Parameters and Methods alignment](#7-parameters-and-methods-alignment)
- [8. Checking a run](#8-checking-a-run)

---

## 1. Pipeline overview

```
raw timelapse .tif  (T, Y, X)
       │
       ▼
segment_stardist.py     StarDist2D on each frame
       │                ──► <movie>_labels.tif   (T, Y, X) uint16
       ▼
track_from_labels.py    region properties  ──►  ROI crop  ──►  midline band
                        removal  ──►  size and separation filters  ──►
                        trackpy linking  ──►  gap closing  ──►  burst detection
       │                ──► per-movie tables, plots and a QC hyperstack
       ▼
compare_groups.py       per-movie summaries  ──►  group figures + statistics
```

`run_pipeline.py` runs segmentation and tracking for every movie in every
group, then the group comparison.

| Script | Role |
|---|---|
| `run_pipeline.py` | Orchestrator; writes `pipeline_status.csv` |
| `segment_stardist.py` | StarDist2D segmentation → label stack |
| `track_from_labels.py` | The core: label stack → filtered spots → tracks → metrics |
| `compare_groups.py` | Per-movie summaries → boxplots, mean±SD time series, Welch t |
| `make_qc_stack_from_labels.py` | 5-channel ImageJ QC hyperstack |
| `visualize_tracks_on_labels.py` | Track-overlay movie |
| `run_midline_v3.py` | Re-run tracking into a new results tree, reusing label stacks |
| `midline_v3_check.py` | Per-movie midline diagnostics (read-only) |
| `burst_timing_check.py` | Per-movie traces, movies-per-timepoint, burst timing |
| `check_results_text_claims.py` | Recomputes the quantities stated about Fig 2c–h |
| `check_panel_h_confound.py` | Nuclei counted with and without tracking |
| `compare_results_trees.py` | Panel-by-panel diff between two results trees |

---

## 2. Setup

Python 3.9+. **Install TensorFlow first** — StarDist and csbdeep need it, and
the right build depends on your platform and GPU.

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml     # then edit the two paths
```

---

## 3. Input data

**Movies.** One multi-page TIFF per embryo, `(T, Y, X)`, single fluorescence
channel (nuclear marker, e.g. `Tg(fli1:H2B-EGFP)`). Pixel size and frame
interval are taken from `config.yaml`, not from the file metadata.

**Filenames must carry the developmental window**, e.g.
`20250616-wt_13-17hpf.tif`. `compare_groups.py` parses `13-17hpf` to place the
movie on an absolute hpf axis. Without that token the movie falls back to
end-alignment, which is almost never what you want.

**Layout.** Group name → folder, set in `config.yaml`:

```
data_root/
  wildtype_posterior/          # dataset.groups.WT.input_dir
    20250616-wt_13-17hpf.tif
    20251118-wt_14-17hpf_f0.tif
    ...
  aqp1a1_mutant_posterior/     # dataset.groups.MUT.input_dir
    20250702_aqp1a1_13-17hpf_f2.tif
    ...
```

---

## 4. Running

```bash
python run_pipeline.py --config config.yaml
python run_pipeline.py --config config.yaml --only_group WT
python run_pipeline.py --config config.yaml --only_movie 20250616-wt_13-17hpf
python run_pipeline.py --config config.yaml --skip_compare
```

### Re-running without re-segmenting

`run_pipeline.py` always re-runs StarDist. When only tracking parameters
changed, that is hours of compute and gigabytes of labels for nothing:

```bash
python run_midline_v3.py --config config_new.yaml \
    --labels_from "<results root that already has seg/ label stacks>" \
    --python "<python with stardist and trackpy>"
```

It points `track_from_labels.py` at the existing labels and writes everything
else into the new `results_root`. It **refuses to run** if the two roots are
the same, so an existing tree cannot be overwritten by accident.

---

## 5. Methods

### 5.1 Segmentation — what counts as an object

Each frame is normalised (csbdeep percentile normalisation) and passed to the
pre-trained StarDist2D model `2D_versatile_fluo`. Detections below
`prob_thresh` (0.4) are dropped; overlapping detections are resolved by
non-maximum suppression at `nms_thresh` (0.2). The result is a `(T, Y, X)`
uint16 label stack, one integer per object per frame.

`skimage.measure.regionprops_table` then gives, per object: centroid, area,
major and minor axis length, eccentricity, solidity. Areas are converted to
µm² with `tracking.px_um` (0.325 µm/pixel).

### 5.2 What counts as a nucleus, and what counts as a fragment

Size alone separates the three classes:

| Class | Area | Used for |
|---|---|---|
| **Nucleus** | `min_area_um2` ≤ A ≤ `max_area_um2`, i.e. **20–60 µm²** | tracking, all per-frame metrics |
| **Fragment** | A ≤ `fragment_area_um2`, i.e. **≤ 10 µm²** | burst detection only; never tracked |
| Neither | everything else | discarded |

The nucleus range is deliberately narrow: it excludes both sub-nuclear debris
and merged detections, which would otherwise be linked into spurious tracks.

### 5.3 ROI — restricting to the tissue

Before anything else, an axis-aligned ROI is estimated from the object
distribution in the first `roi.early_frac` (0.30) of frames, expanded by
`x_expand_frac` / `y_expand_frac` and padded by `y_pad_um`. Objects outside it
are dropped. This removes debris at the edges of the field without any manual
drawing. The rectangle is written to `safety_roi_um.csv`.

### 5.4 Midline — locating it, and why it is removed

The LPM forms **two bilateral bands of nuclei** that converge on the midline.
Objects within a band around the midline are removed, in every frame, for both
genotypes, because that region contains the notochord and the axial structures
rather than migrating LPM cells.

`method: bilateral` (the default) locates the midline as the **gap between the
two bands**, in three steps:

1. **Density.** Build a 1-D density of the x coordinates of **nucleus-sized
   objects only**, in the first `fit_early_frac` (0.33) of frames: a 1 µm grid
   smoothed with an 8 µm Gaussian. Restricting to nucleus-sized objects matters
   — the all-object profile is dominated by sub-nuclear debris, which is
   densest in the midline region itself.

2. **Which two peaks are the bands.** Peaks at least `min_band_sep_um` (60 µm)
   apart are found, and **every pair** is scored:

   ```
   score = [ min(h_left, h_right) - valley ] × [ 1 - mass_outside_pair / total_mass ]
   ```

   The first factor asks for two tall peaks with a deep valley between them;
   the second asks that the pair bracket the tissue. Scoring pairs rather than
   simply taking the two tallest peaks matters because a movie often has
   **three** peaks — the two bands plus the cells that have already converged
   on the midline — and the middle one can be as tall as a band. The midline
   position is the midpoint of the winning pair.

3. **Tilt.** The point cloud is rotated through ±`tilt_sweep_max_deg` (25°) in
   `tilt_sweep_step_deg` (0.5°) steps, about the **centre of the y range**. At
   each angle the same pair score is recomputed, and the best-scoring angle is
   taken. The position is then re-derived in that rotated frame, since the
   anchor was measured on a projection along a different axis; a re-fit that
   moves more than `max_recenter_um` (60 µm) is not used.

Objects whose perpendicular distance to the fitted line is ≤
`band_half_width_um` (**20 µm**) are removed. The fit and its diagnostics go to
`midline_tls_spots_filter_info.csv`, including `tilt_deg`, `gap_width_um` and
`cells_on_midline`.

`gap_width_um = 0` means the midline is **populated rather than empty**, which
at 13–17 hpf is expected — the LPM is converging. It is reported, never used to
reject a fit.

`band_half_width_um` is an analysis choice, not a display setting: because the
midline is populated, a wider band removes more of the population being
measured.

`method: mode_tls` is an earlier, simpler estimator kept as an option. It is
not recommended: it assumes the midline is the densest column of objects, which
does not hold for a bilateral tissue.

### 5.5 Thinning before linking

Two guards keep the linking problem well-posed:

- **`max_spots_per_frame`** (150) — if a frame has more nucleus-sized objects
  than this, only the largest 150 are kept.
- **`min_separation_um`** (3 µm) — of any two objects closer than this, the
  larger is kept. Prevents one nucleus detected twice from creating two tracks.

### 5.6 Tracking

`trackpy.link` on the µm centroids:

- **`search_range_um`** (8.5 µm) — the maximum a nucleus may move between
  consecutive frames. With `dt_min` = 10 min that is 0.85 µm/min.
- **`memory`** (2 frames) — a nucleus may vanish for up to two frames and still
  be reconnected to the same track, covering brief segmentation failures.
- If linking fails as too ambiguous, the search range is reduced by
  `adaptive_step` (0.95) repeatedly, down to `adaptive_stop_um` (5 µm).
- **`min_track_len`** (3 frames) — shorter tracks are discarded.

**Gap closing** then joins track ends to track starts across up to
`max_gap_frames` (3) and `max_disp_um` (10 µm), with a cost combining distance,
area change, direction alignment and eccentricity change
(`w_dist`, `w_area`, `w_align`, `w_ecc`), accepting only matches below
`cost_max` (18).

**Speed** is the frame-to-frame centroid displacement divided by `dt_min`. The
first frame of each track has no preceding frame and is skipped.

### 5.7 Nuclear fragmentation ("burst") events

A burst is a large nucleus disappearing and being replaced by several small
fragments. Both a track ending (`detect_track_break`) and a track surviving
through the event (`detect_within_track`) are detected.

At frame *t* a tracked nucleus is a burst when all hold:

| Condition | Parameter | Default |
|---|---|---|
| The parent was large | `big_area_um2` | ≥ 30 µm² |
| Its area collapses | `area_drop_ratio` | area(t+1) < 0.60 × area(t) |
| Small objects appear nearby | `min_frags` … `max_frags` | 2 to 8 |
| within a short window | `lookahead_frames` | ≤ 2 frames |
| and a short distance | `search_radius_um` | ≤ 8 µm |
| each of them small relative to the parent | `frag_area_ratio_max` | ≤ 0.30 × parent |

**`frag_area_ratio_max` must stay below 0.5.** A normal mitotic division
produces two daughters each about half the parent; at 0.5 or above those would
be counted as fragmentation. 0.30 keeps mitosis out.

### 5.8 Group comparison

Each movie is placed on an **absolute hpf axis** using the window in its
filename — frames are not stretched to a common count, so a given x position
means the same developmental stage in every embryo. Movies covering different
windows contribute over different parts of the axis.

**The movie (= embryo) is the unit of analysis.** Per-frame values are reduced
to one value per movie, and groups are compared with a two-sided Welch t-test
(`scipy.stats.ttest_ind(equal_var=False)`). Individual nuclei are never the
unit — that would be pseudoreplication.

Exact p-values are printed; no significance stars.

---

## 6. Outputs

```
results_root/
├── WT/                                   # one folder per group
│   └── <movie>/
│       ├── seg/<movie>_labels.tif        # label stack from StarDist
│       ├── run_log.txt
│       └── trackpy/
│           ├── spots_raw_all_objects.csv              every detection, all frames
│           ├── spots_all_objects_roi.csv              after the ROI crop
│           ├── spots_all_objects_roi_midline_spotsfiltered.csv
│           │                                          after the midline band
│           ├── spots_for_tracking_raw.csv             nucleus-sized subset
│           ├── spots_for_tracking_final.csv           after thinning
│           ├── tracks_linked.csv                      + track_id per spot
│           ├── safety_roi_um.csv                      the ROI rectangle
│           ├── midline_tls_spots_filter_info.csv      fit: p0, v, tilt_deg,
│           │                                          gap_width_um, n_removed
│           ├── midline_tls_track_filter_info.csv
│           ├── midline_tls_dropped_tracks.csv
│           ├── burst_events.csv                       one row per event
│           ├── burst_events_per_frame.csv
│           ├── fragments_per_frame.csv
│           ├── speed_per_frame.csv
│           ├── width_per_frame.csv
│           ├── summary_per_frame.csv                  per-frame aggregates
│           ├── movie_summary.csv                      ONE ROW: the movie
│           ├── preburst_area_traces.csv               area before each burst
│           ├── qc_tracks_hyperstack.tif               5-channel QC stack
│           ├── qc_tracks_debug.txt
│           └── *.png                                  per-movie diagnostics
├── MUT/
├── _group_compare/
└── pipeline_status.csv                   one row per movie, OK or the failure
```

### The two tables that matter

**`movie_summary.csv`** — one row per movie, and the source of every boxplot:

`n_frames`, `n_tracks`, `n_burst_events`, `burst_per_track_ratio`,
`mean_tracked_nuclei`, `median_tracked_nuclei`, `max_tracked_nuclei`,
`mean_fragments_per_frame`, `mean_median_area_um2`,
`mean_median_speed_um_per_min`, `midline_spots_removed`,
`midline_band_half_width_um`, plus the tracking parameters used.

**`burst_events.csv`** — one row per event: `mode` (track break or within
track), `track_id`, `break_frame`, position, `area_prev_um2`, `area_next_um2`,
`area_ratio_next_over_prev`, `n_frags`, `sum_frag_area_um2`,
`median_frag_area_um2`.

### `_group_compare/`

| File | Contents |
|---|---|
| `all_movies_summary.csv` | every movie's `movie_summary.csv`, stacked — **the dots on the boxplots** |
| `aligned_time_series_per_movie.csv` | per-movie values on the common hpf grid — **the mean±SD bands** |
| `movie_time_windows.csv` | the hpf window parsed from each filename |
| `preburst_area_per_movie.csv` | nuclear area in the frames before each burst |
| `area_vs_time_mean_sd.*` | median nuclear area vs hpf, mean±SD across embryos |
| `speed_vs_time_mean_sd.*` | median migration speed vs hpf |
| `n_tracked_vs_time_mean_sd.*` | number of tracked nuclei vs hpf |
| `bursts_vs_time_mean_sd.*` | fragmentation events vs hpf |
| `width_vs_time_mean_sd.*` | tissue width (x-quantile spread) vs hpf |
| `preburst_area_vs_time.*` | area in the frames leading up to a burst |
| `boxplot_area.*` | per-embryo mean nuclear area, WT vs MUT, with p |
| `boxplot_speed.*` | per-embryo mean speed, with p |
| `boxplot_burst_track_ratio.*` | fragmentation events per track, with p |
| `boxplot_preburst_area_dt_minus1.*` | area one frame before a burst |

Every plot is written as both `.png` and `.svg`; the svg carries real text
(`svg.fonttype = "none"`), so labels stay editable in Illustrator.

### The QC hyperstack

`qc_tracks_hyperstack.tif` opens in Fiji as five channels:

1. the label image
2. every detected object
3. objects that were successfully tracked
4. the ROI rectangle and the **midline with its exclusion band**
5. track trajectories

Channel 4 is the one to look at first: it shows exactly which objects the
midline band removed.

---

## 7. Parameters and Methods alignment

| Methods statement | Value | Config key |
|---|---|---|
| Pixel size | 0.325 µm/px | `tracking.px_um` |
| Frame interval | 10 min | `tracking.dt_min` |
| Segmentation model | StarDist2D `2D_versatile_fluo` | `segmentation.model` |
| Detection probability threshold | 0.4 | `segmentation.prob_thresh` |
| NMS overlap threshold | 0.2 | `segmentation.nms_thresh` |
| Nuclear size range | 20–60 µm² | `tracking.min_area_um2`, `tracking.max_area_um2` |
| Fragment threshold | ≤ 10 µm² | `tracking.fragment_area_um2` |
| Max objects per frame | 150 | `tracking.max_spots_per_frame` |
| Minimum separation | 3 µm | `tracking.min_separation_um` |
| Linking search radius | 8.5 µm | `tracking.search_range_um` |
| Linking memory | 2 frames | `tracking.memory` |
| Minimum track length | 3 frames | `tracking.min_track_len` |
| Gap closing | ≤ 3 frames, ≤ 10 µm | `gap_closing.max_gap_frames`, `.max_disp_um` |
| Midline method | bilateral band midpoint | `midline_tls.method` |
| Midline exclusion band | 20 µm half-width | `midline_tls.band_half_width_um` |
| Frames used for the midline fit | first 33% | `midline_tls.fit_early_frac` |
| Tilt search range | ±25°, 0.5° steps | `midline_tls.tilt_sweep_max_deg`, `.tilt_sweep_step_deg` |
| Burst: parent size | ≥ 30 µm² | `burst_detection.big_area_um2` |
| Burst: area drop | < 60% of previous | `burst_detection.area_drop_ratio` |
| Burst: fragment size | ≤ 30% of parent | `burst_detection.frag_area_ratio_max` |
| Burst: fragment count | 2–8 | `burst_detection.min_frags`, `.max_frags` |
| Burst: search window | 2 frames, 8 µm | `burst_detection.lookahead_frames`, `.search_radius_um` |
| Analysis window | 13–17 hpf | `compare.start_hpf`, `compare.end_hpf` |
| Statistical test | Welch two-sided t-test | `scipy.stats.ttest_ind(equal_var=False)` |

All spatial quantities are in µm and µm², converted from pixels with
`tracking.px_um`. Time is in minutes or hpf.

---

## 8. Checking a run

The midline is the step most worth inspecting, since it decides which objects
enter the analysis at all:

```bash
python midline_v3_check.py --config config.yaml
```

Writes into a **new** directory beside the results tree — nothing in the
results tree is touched — one figure per movie:

- **left** — nuclei and other objects, with the fitted midline and the
  exclusion band drawn on
- **middle** — the density across the midline at the chosen angle, with the two
  band peaks marked
- **right** — the angle sweep, showing whether the tilt is well-determined or
  the curve is flat

plus `midline_v3_comparison.csv` with the centre, tilt, gap width and flags per
movie, and `_summary_v3.png` across all movies.

To pin a movie by hand, put a `midline_manual.csv` in that directory with
columns `group,movie,x_center_um,tilt_deg`; the hand-set line is drawn in
purple beside the automatic one so the value can be checked before use.

---

## License

MIT — see [LICENSE](../LICENSE) at the repository root.
