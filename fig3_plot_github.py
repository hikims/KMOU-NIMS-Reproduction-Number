"""
Figure 3 plotting helpers (season-level confirmation and Rt figures).

GitHub-ready changes:
- Removed hard-coded absolute PATH
- Translated Korean comments to English
- Added PROJECT_PATH environment variable support
- Added output directory arguments
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_PATH = os.environ.get("PROJECT_PATH", ".")  # repository root by default


def _to_six_series_list(x):
    """Convert various 6-series formats into a list of six 1D numpy arrays."""
    if isinstance(x, (list, tuple)):
        out = [np.asarray(a).ravel() for a in x]
        if len(out) != 6:
            raise ValueError(f"Expected 6 series, got {len(out)}")
        return out

    if isinstance(x, pd.DataFrame):
        cols = [c for c in ["age1", "age2", "age3", "age4", "age5", "age6"] if c in x.columns]
        if len(cols) == 6:
            return [x[c].to_numpy().ravel() for c in cols]
        if x.shape[1] >= 6:
            return [x.iloc[:, i].to_numpy().ravel() for i in range(6)]
        raise ValueError(f"DataFrame must have 6 columns; got {x.shape[1]}")

    if isinstance(x, np.ndarray) and x.ndim == 2:
        if x.shape[1] == 6:
            return [x[:, i].ravel() for i in range(6)]
        if x.shape[0] == 6:
            return [x[i, :].ravel() for i in range(6)]

    raise TypeError(f"Unsupported type/shape: {type(x)}, {getattr(x, 'shape', None)}")


def confirm_plotting(year: int, data_dir: str | None = None, out_dir: str | None = None) -> None:
    """
    Plot observed vs simulated confirmations for a given season.

    Expects:
    - {year}_confirmation_obs.csv
    - {year}_confirmation_sim.csv

    Both files must include a `date` column and age columns:
    - age1 ... age6 (with suffixes _obs / _sim after merge)
    """
    if data_dir is None:
        data_dir = os.path.join(PROJECT_PATH, "particle", "data")
    if out_dir is None:
        out_dir = os.path.join(PROJECT_PATH, "paper_image")

    colors = ["red", "brown", "orange", "green", "blue", "purple"]
    age_labels = ["Age1", "Age2", "Age3", "Age4", "Age5", "Age6"]

    obs_path = os.path.join(data_dir, f"{year}_confirmation_obs.csv")
    sim_path = os.path.join(data_dir, f"{year}_confirmation_sim.csv")

    df_obs = pd.read_csv(obs_path, parse_dates=["date"])
    df_sim = pd.read_csv(sim_path, parse_dates=["date"])

    # Align by date
    df = pd.merge(df_obs, df_sim, on="date", suffixes=("_obs", "_sim"))

    fig, ax = plt.subplots(figsize=(25, 16))

    style = {
        "obs": {"linestyle": "-", "linewidth": 2.0},
        "sim": {"linestyle": "--", "linewidth": 2.0},
    }

    obs_handles, obs_labels = [], []
    sim_handles, sim_labels = [], []

    for i in range(6):
        col_obs = f"age{i+1}_obs"
        col_sim = f"age{i+1}_sim"

        h_obs, = ax.plot(df["date"], df[col_obs], color=colors[i], **style["obs"], label=f"{age_labels[i]} (obs)")
        h_sim, = ax.plot(df["date"], df[col_sim], color=colors[i], **style["sim"], label=f"{age_labels[i]} (sim)")

        obs_handles.append(h_obs)
        obs_labels.append(f"{age_labels[i]} (obs)")
        sim_handles.append(h_sim)
        sim_labels.append(f"{age_labels[i]} (sim)")

    ax.set_title(f"{year}–{year+1} Season", fontsize=50)
    ax.set_xlabel("Year–Month", fontsize=50)
    ax.set_ylabel("Confirmation", fontsize=50)

    # X ticks at selected months (Sep, Nov, Jan, Mar, May, Jul, Sep)
    start_year = year
    end_year = year + 1
    tick_year_months = [(start_year, 9), (start_year, 11), (end_year, 1), (end_year, 3), (end_year, 5), (end_year, 7), (end_year, 9)]

    ticks, tick_labels = [], []
    for y, m in tick_year_months:
        subset = df.loc[(df["date"].dt.year == y) & (df["date"].dt.month == m), "date"]
        if not subset.empty:
            ticks.append(subset.iloc[0])
            tick_labels.append(f"{y}-{m:02d}")

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=40)

    ax.tick_params(axis="x", labelsize=40)
    ax.tick_params(axis="y", labelsize=40)
    ax.grid(True, alpha=0.3)

    # Legend: 1st column obs, 2nd column sim
    handles = obs_handles + sim_handles
    labels = obs_labels + sim_labels
    ax.legend(handles, labels, fontsize=25, ncol=2, loc="upper right", frameon=True, columnspacing=1.8, handlelength=3.0)

    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    file_name = os.path.join(out_dir, f"{year}_confirmation.eps")
    fig.savefig(file_name, format="eps", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", file_name)


def Rt_plotting(year: int, data_dir: str | None = None, out_dir: str | None = None) -> None:
    """
    Plot age-specific Rt and all-ages Rt for a given season.

    Expects:
    - {year}_Ps_Rt.csv   (columns: age1..age6 [+ optional date])
    - {year}_all_Ps_Rt.csv (one Rt column [+ optional date])
    """
    if data_dir is None:
        data_dir = os.path.join(PROJECT_PATH, "particle", "data")
    if out_dir is None:
        out_dir = os.path.join(PROJECT_PATH, "paper_image")

    colors = ["red", "brown", "orange", "green", "blue", "purple"]
    age_labels = ["Age1", "Age2", "Age3", "Age4", "Age5", "Age6"]

    file_age = os.path.join(data_dir, f"{year}_Ps_Rt.csv")
    file_all = os.path.join(data_dir, f"{year}_all_Ps_Rt.csv")

    df_age = pd.read_csv(file_age)
    df_all = pd.read_csv(file_all)

    # Date handling
    if "date" in df_age.columns:
        df_age["date"] = pd.to_datetime(df_age["date"])
    else:
        start_date = pd.to_datetime(f"{year}-09-01")
        df_age["date"] = pd.date_range(start=start_date, periods=len(df_age), freq="D")

    if "date" in df_all.columns:
        df_all["date"] = pd.to_datetime(df_all["date"])
    else:
        df_all["date"] = pd.to_datetime(df_age["date"].values)

    x = df_age["date"]

    all_cols = [c for c in df_all.columns if c != "date"]
    if len(all_cols) != 1:
        raise ValueError(f"Cannot infer a single Rt column from {file_all}: {all_cols}")
    all_col = all_cols[0]

    leng = min(len(df_age), len(df_all))
    x = x.iloc[:leng]

    fig, ax = plt.subplots(figsize=(25, 16))

    # Age-specific Rt (solid)
    for i in range(6):
        age_col = f"age{i+1}"
        ax.plot(x, df_age[age_col].to_numpy()[:leng], color=colors[i], linestyle="-", linewidth=2, label=age_labels[i])

    # All-ages Rt (black dashed)
    ax.plot(x, df_all[all_col].to_numpy()[:leng], color="black", linestyle="--", linewidth=2, label="All")

    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1)

    ax.set_title(f"{year}–{year+1} Season", fontsize=50)
    ax.set_xlabel("Year–Month", fontsize=50)
    ax.set_ylabel(r"$\mathcal{R}_{t}(t)$", fontsize=50)
    ax.set_ylim(0.0, 3.0)

    # X ticks at selected months
    start_year = year
    end_year = year + 1
    tick_year_months = [(start_year, 9), (start_year, 11), (end_year, 1), (end_year, 3), (end_year, 5), (end_year, 7), (end_year, 9)]

    ticks, tick_labels = [], []
    for y, m in tick_year_months:
        subset = df_age.loc[(df_age["date"].dt.year == y) & (df_age["date"].dt.month == m), "date"]
        if not subset.empty:
            ticks.append(subset.iloc[0])
            tick_labels.append(f"{y}-{m:02d}")

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=40)

    ax.tick_params(axis="x", labelsize=40)
    ax.tick_params(axis="y", labelsize=40)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=25, loc="upper right")

    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    file_name = os.path.join(out_dir, f"{year}_Rt.eps")
    fig.savefig(file_name, format="eps", bbox_inches="tight")
    plt.close(fig)

    print("Saved:", file_name)
