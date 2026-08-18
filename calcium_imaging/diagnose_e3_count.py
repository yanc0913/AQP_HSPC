# diagnose_e3_count.py
# -*- coding: utf-8 -*-
"""检查 E3 round cells 的数据点数和具体值"""

import pandas as pd
from pathlib import Path

import calcium_config as cfg

emb = pd.read_csv(cfg.tables_dir() / "Q2_embryo_summary_cells.csv")

# 找 E3_vs_ISO_vs_BDM pair, E3 condition, round cell_class
target = emb[
    (emb["pair_id"].astype(str) == "E3_vs_ISO_vs_BDM") &
    (emb["condition"] == "E3") &
    (emb["cell_class"] == "round")
]
events_col = cfg.events_col_per_embryo()

print(f"\n=== E3, round, pair=E3_vs_ISO_vs_BDM ===")
print(f"Rows: {len(target)}")
print()
print(target[["batch_id", "embryo_id", "n_cells", events_col]].to_string(index=False))
print()
print(f"{events_col} values (sorted): {sorted(target[events_col].dropna().tolist())}")
print(f"NaN count: {target[events_col].isna().sum()}")

# 也看看 round cells 在 Q2_events_cells.csv 里是不是有 ROI 缺失
print()
print("=== Per-cell detail (round only) ===")
events = pd.read_csv(cfg.tables_dir() / "Q2_events_cells.csv")
e3_round = events[
    (events["pair_id"].astype(str) == "E3_vs_ISO_vs_BDM") &
    (events["condition"] == "E3") &
    (events["cell_class"] == "round")
]
n_events_col = cfg.n_events_col()
print(e3_round[["embryo_id", "roi_name", n_events_col]].to_string(index=False))
