# -*- coding: utf-8 -*-
"""
Figure 4: Age-specific Rt heatmaps across seasons.

GitHub-ready changes:
- Removed hard-coded absolute DATA_DIR and output path
- Translated Korean comments to English
- Output file is written under ./paper_image by default
- Supports PROJECT_PATH environment variable for data discovery

Expected input files
--------------------
For each year in `years`, the script expects:
- {year}_Ps_Rt.csv inside {DATA_DIR}

Each file should have columns: age1..age6 (and may have extra columns).
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm

mpl.rcParams["text.usetex"] = False


PROJECT_PATH = os.environ.get("PROJECT_PATH", ".")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(PROJECT_PATH, "particle", "data"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(PROJECT_PATH, "paper_image"))


def main(save_name: str = "fig4.eps") -> str:
    # ---------------------------
    # (A) Load data
    # ---------------------------
    years = [y for y in range(2009, 2023) if y not in (2019, 2020, 2021)]

    dfs = []
    for year in years:
        file_path = os.path.join(DATA_DIR, f"{year}_Ps_Rt.csv")
        if not os.path.exists(file_path):
            print("[WARN] missing:", file_path)
            continue

        df = pd.read_csv(file_path)
        df["time"] = np.arange(len(df))
        df["year"] = year

        df_long = df.melt(id_vars=["time", "year"], var_name="age_group", value_name="Rt")
        dfs.append(df_long)

    if not dfs:
        raise FileNotFoundError(f"No Rt CSV files found in DATA_DIR={DATA_DIR}")

    rt_all = pd.concat(dfs, ignore_index=True)

    # ---------------------------
    # (B) Age labels
    # ---------------------------
    age_map = {
        "age1": "0-5y",
        "age2": "6-11y",
        "age3": "12-17y",
        "age4": "18-44y",
        "age5": "45-64y",
        "age6": "65y+",
    }
    age_order = ["0-5y", "6-11y", "12-17y", "18-44y", "45-64y", "65y+"]

    rt_all["age_group"] = rt_all["age_group"].map(age_map)
    rt_all["age_group"] = pd.Categorical(rt_all["age_group"], categories=age_order, ordered=True)
    rt_all = rt_all.sort_values(["age_group", "year", "time"])

    # ---------------------------
    # (C) Plot
    # ---------------------------
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.ravel()

    fig.subplots_adjust(left=0.08, right=0.90, top=0.90, bottom=0.12, wspace=0.15, hspace=0.12)

    max_time = int(rt_all["time"].max())

    month_labels = ["Sep", "Nov", "Jan", "Mar", "May", "Jul", "Aug"]
    month_positions = np.linspace(0, max_time, len(month_labels))
    season_labels = [f"{str(y)[2:]}-{str(y+1)[2:]}" for y in years]

    # Fix heatmap range to 0~3, centered at 1
    vmin, vmax = 0.0, 3.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color=(1, 1, 1, 1))

    for idx, (ax, age) in enumerate(zip(axes, age_order)):
        sub = rt_all[rt_all["age_group"] == age]
        mat = sub.pivot(index="year", columns="time", values="Rt")
        mat = mat.reindex(columns=range(0, max_time + 1))
        Z = np.ma.masked_invalid(mat.to_numpy())

        ax.imshow(Z, aspect="auto", origin="lower", cmap=cmap, norm=norm)

        ax.set_title(age, fontsize=20)

        if idx % 3 == 0:
            ax.set_yticks(range(len(season_labels)))
            ax.set_yticklabels(season_labels, fontsize=14)
            ax.set_ylabel("Season", fontsize=20)
        else:
            ax.set_yticks([])

        if idx < 3:
            ax.set_xticks([])
        else:
            ax.set_xticks(month_positions)
            ax.set_xticklabels(month_labels, fontsize=14)
            ax.set_xlabel("Month", fontsize=20)

    # ---------------------------
    # (D) Colorbar
    # ---------------------------
    cbar_ax = fig.add_axes([0.92, 0.16, 0.02, 0.68])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"$\mathcal{R}_t(t)$", fontsize=22)
    cbar.ax.tick_params(labelsize=18)

    # Reference line at Rt=1
    cbar.ax.axhline(1.0, color="black", linewidth=2)
    cbar.set_ticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    # ---------------------------
    # (E) Title
    # ---------------------------
    fig.suptitle(r"Age-specific $\mathcal{R}_j(t)$ heatmaps across seasons", fontsize=26, y=0.965)

    # ---------------------------
    # (F) Save
    # ---------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, save_name)
    plt.savefig(out_path, format="eps", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)
    return out_path


if __name__ == "__main__":
    main()
