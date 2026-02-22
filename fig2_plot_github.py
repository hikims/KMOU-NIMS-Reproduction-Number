"""
Figure 2 plotting utilities (synthetic data).

This file contains helper functions to visualize:
- age-specific confirmations (synthetic)
- age-specific Rt (synthetic vs comparator)
- pre-defined Rt vs Gorji Rt
- pre-defined Rt vs particle smoother Rt

Notes
-----
- This is a GitHub-ready version of the original script:
  - Korean comments were translated to English
  - Output directory is configurable
  - No hard-coded absolute paths
"""

from __future__ import annotations

import os
from typing import List, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


Series6 = Union[List[Sequence[float]], tuple]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------
# Pre-defined reproduction number (Synthetic)
# ---------------------------------------------
def pre_confirm_plotting(groups1: Series6, groups2: Series6, sentence: str, out_dir: str = "./image") -> int:
    """
    Plot age-specific confirmation curves for synthetic data.

    Parameters
    ----------
    groups1 : list of 6 arrays
        Confirmation series (e.g., synthetic) by age group.
    groups2 : list of 6 arrays
        Optional comparator series (unused by default, kept for compatibility).
    sentence : str
        Figure label / prefix for filename.
    out_dir : str
        Output directory.

    Returns
    -------
    int
        Always 0 (kept for backward compatibility).
    """
    temp_leng = min(len(groups1[0]), len(groups2[0]))
    x = np.arange(temp_leng)

    plt.figure(figsize=(18, 12))
    colors = ["red", "brown", "orange", "green", "blue", "purple"]

    for i in range(6):
        plt.plot(x, np.asarray(groups1[i])[:temp_leng], color=colors[i], label=f"Age {i+1}")
        # If you want comparator (e.g., observed), uncomment:
        # plt.plot(x, np.asarray(groups2[i])[:temp_leng], color=colors[i], linestyle='--')

    plt.xlabel("Time", fontsize=50)
    plt.ylabel("Confirmation", fontsize=50)
    plt.title("Confirmed Cases of Synthetic Data", fontsize=50)
    plt.tick_params(axis="x", labelsize=40)
    plt.tick_params(axis="y", labelsize=40)
    plt.legend(fontsize=25)
    plt.grid(True)
    plt.tight_layout()

    _ensure_dir(out_dir)
    file_name = os.path.join(out_dir, f"{sentence}_confirmation.eps")
    plt.savefig(file_name, format="eps", bbox_inches="tight")
    plt.close()
    return 0


def pre_Rt_plotting(groups1: Series6, groups2: Series6, sentence: str, out_dir: str = "./image") -> None:
    """
    Plot age-specific Rt for two series (solid vs dashed).

    Parameters
    ----------
    groups1 : list of 6 arrays
        Series 1 (solid).
    groups2 : list of 6 arrays
        Series 2 (dashed).
    sentence : str
        Figure label / prefix.
    out_dir : str
        Output directory.
    """
    shift = 2
    temp_leng = min(len(groups1[0]), len(groups2[0]))
    x = np.arange(temp_leng - shift)

    plt.figure(figsize=(18, 12))
    colors = ["red", "brown", "orange", "green", "blue", "purple"]

    for i in range(6):
        plt.plot(x, np.asarray(groups1[i])[shift:temp_leng], color=colors[i], label=f"Age Group {i+1}")
        plt.plot(x, np.asarray(groups2[i])[shift:temp_leng], color=colors[i], linestyle="--")

    plt.axhline(y=1, color="black", linestyle="--", label="Threshold = 1")
    plt.title(f"{sentence} Age Reproduction Number", fontsize=30)
    plt.tick_params(axis="x", labelsize=22)
    plt.tick_params(axis="y", labelsize=22)
    plt.xlabel("Time", fontsize=24)
    plt.ylabel(r"$\mathcal{R}_{t}(t)$", fontsize=24)
    plt.ylim(0, 3)
    plt.legend(fontsize=24)
    plt.grid(True)
    plt.tight_layout()

    _ensure_dir(out_dir)
    file_name = os.path.join(out_dir, f"{sentence}_Age_Rt.eps")
    plt.savefig(file_name, format="eps", bbox_inches="tight")
    plt.close()


def pre_Gorji_plotting(
    groups_pre: Series6,
    groups_gorji: Series6,
    sentence: str,
    shift: int = 2,
    out_dir: str = "./image",
):
    """
    Compare pre-defined Rt (solid) vs Gorji Rt (dashed with markers).

    Parameters
    ----------
    groups_pre : list of 6 arrays
        Pre-defined Rt (synthetic).
    groups_gorji : list of 6 arrays
        Gorji Rt estimate.
    sentence : str
        Figure label / prefix.
    shift : int
        Leading shift (to drop unstable early points).
    out_dir : str
        Output directory.

    Returns
    -------
    str
        Saved file path.
    """
    temp_leng = min(len(groups_pre[0]), len(groups_gorji[0]))
    x = np.arange(temp_leng - shift)
    colors = ["red", "brown", "orange", "green", "blue", "purple"]

    fig, ax = plt.subplots(figsize=(18, 12))

    for i in range(6):
        ax.plot(
            x,
            np.asarray(groups_pre[i])[shift:temp_leng],
            color=colors[i],
            linestyle="-",
            linewidth=1.5,
            label=f"Age {i+1} Synthetic data",
        )
        ax.plot(
            x,
            np.asarray(groups_gorji[i])[shift:temp_leng],
            color=colors[i],
            linestyle="--",
            linewidth=1.5,
            marker="o",
            markevery=6,
            markersize=6,
            label=f"Age {i+1} Gorji method",
        )

    ax.axhline(y=1, color="black", linestyle="--", linewidth=2, label="Threshold = 1")
    ax.set_title(f"{sentence} Method", fontsize=50)
    ax.set_xlabel("Time", fontsize=50)
    ax.set_ylabel(r"$\mathcal{R}_{t}(t)$", fontsize=50)
    ax.tick_params(axis="x", labelsize=40)
    ax.tick_params(axis="y", labelsize=40)
    ax.set_ylim(0, 3)
    ax.grid(True)

    # Custom legend: column-wise ordering (pre + threshold, then Gorji)
    legend_pre = [
        Line2D([0], [0], color=colors[i], linestyle="-", linewidth=2, label=f"Age {i+1} Synthetic data")
        for i in range(6)
    ]
    legend_gorji = [
        Line2D([0], [0], color=colors[i], linestyle="--", marker="o", linewidth=2, markersize=6,
               label=f"Age {i+1} Gorji method")
        for i in range(6)
    ]
    legend_threshold = Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Threshold = 1")

    legend_handles = legend_pre + [legend_threshold] + legend_gorji
    ax.legend(handles=legend_handles, fontsize=25, ncol=2, frameon=False, columnspacing=1.8, handlelength=3)

    plt.tight_layout()

    _ensure_dir(out_dir)
    file_name = os.path.join(out_dir, f"{sentence}_pre_vs_Gorji.eps")
    fig.savefig(file_name, format="eps", bbox_inches="tight")
    plt.close(fig)

    print("Saved:", file_name)
    return file_name


def pre_PS_plotting(
    groups_pre: Series6,
    groups_ps: Series6,
    sentence: str,
    shift: int = 2,
    out_dir: str = "./image",
):
    """
    Compare pre-defined Rt (solid) vs particle smoother Rt (dashed squares).

    Returns
    -------
    str
        Saved file path.
    """
    temp_leng = min(len(groups_pre[0]), len(groups_ps[0]))
    x = np.arange(temp_leng - shift)
    colors = ["red", "brown", "orange", "green", "blue", "purple"]

    fig, ax = plt.subplots(figsize=(18, 12))

    for i in range(6):
        ax.plot(
            x,
            np.asarray(groups_pre[i])[shift:temp_leng],
            color=colors[i],
            linestyle="-",
            linewidth=1.5,
            label=f"Age {i+1} Synthetic data",
        )
        ax.plot(
            x,
            np.asarray(groups_ps[i])[shift:temp_leng],
            color=colors[i],
            linestyle="--",
            linewidth=1.5,
            marker="s",
            markevery=6,
            markersize=6,
            label=f"Age {i+1} particle smoother",
        )

    ax.axhline(y=1, color="black", linestyle="--", linewidth=2, label="Threshold = 1")

    ax.set_title(f"{sentence} Method", fontsize=50)
    ax.set_xlabel("Time", fontsize=50)
    ax.set_ylabel(r"$\mathcal{R}_{t}(t)$", fontsize=50)
    ax.tick_params(axis="x", labelsize=40)
    ax.tick_params(axis="y", labelsize=40)
    ax.set_ylim(0, 3)
    ax.grid(True)

    legend_pre = [
        Line2D([0], [0], color=colors[i], linestyle="-", linewidth=2, label=f"Age {i+1} Synthetic data")
        for i in range(6)
    ]
    legend_ps = [
        Line2D([0], [0], color=colors[i], linestyle="--", marker="s", linewidth=2, markersize=6,
               label=f"Age {i+1} particle smoother")
        for i in range(6)
    ]
    legend_threshold = Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Threshold = 1")

    legend_handles = legend_pre + [legend_threshold] + legend_ps
    ax.legend(handles=legend_handles, fontsize=25, ncol=2, frameon=False, columnspacing=1.8, handlelength=3)

    plt.tight_layout()

    _ensure_dir(out_dir)
    file_name = os.path.join(out_dir, f"{sentence}_pre_vs_ps.eps")
    fig.savefig(file_name, format="eps", bbox_inches="tight")
    plt.close(fig)

    print("Saved:", file_name)
    return file_name
