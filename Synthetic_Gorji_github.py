"""
Gorji-style age-structured instantaneous reproduction number (R_t) estimation.

This module provides:
- Rt_gorji: core computation of R_t by age-to-age transmission contributions
- Gorji: convenience wrapper that builds serial-interval weights and returns age-specific R_t

Notes
-----
- The public function signatures are kept identical to the original code so this file can
  be dropped in as a direct replacement.
- Some imports and comments were cleaned up for GitHub publishing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import expon


def Rt_gorji(mat: np.ndarray, s_interval: np.ndarray, M: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Compute the Gorji instantaneous reproduction number tensor and return the
    age-specific R_t (summed over sources).

    Parameters
    ----------
    mat : ndarray, shape (n_age, T)
        Time series (e.g., incidence proxy) for each age group.
    s_interval : ndarray, shape (L, n_age)
        Serial interval PMF weights per age group (columns).
    M : ndarray, shape (n_age, n_age)
        Contact (mixing) matrix.
    beta : ndarray, shape (n_age,)
        Relative transmissibility factors per age group (often ones).

    Returns
    -------
    R : ndarray, shape (n_age, T)
        Age-specific instantaneous reproduction numbers.
    """
    n_rows_M = M.shape[0]
    n_cols_M = M.shape[1]
    n_cols_mat = mat.shape[1]

    Rt = np.zeros((n_rows_M, n_cols_M, n_cols_mat), dtype=float)

    # i: recipient age group, j: infector age group, t: time index
    for i in range(n_rows_M):
        for j in range(n_cols_M):
            for t in range(n_cols_mat):
                if t < len(s_interval):
                    weight_vector = s_interval[: t + 1, j] if t > 0 else np.array([1.0])
                    numerator = M[i, j] * beta[j] * mat[i, t]
                    denominator = np.sum(
                        [
                            M[i, k] * beta[k] * np.sum(np.flip(mat[k, : (t + 1)]) * weight_vector)
                            for k in range(n_cols_M)
                        ]
                    )
                else:
                    recent_weights = s_interval[:, j]
                    numerator = M[i, j] * beta[j] * mat[i, t]
                    denominator = np.sum(
                        [
                            M[i, k]
                            * beta[k]
                            * np.sum(np.flip(mat[k, (t - len(recent_weights)) : t]) * recent_weights)
                            for k in range(n_cols_M)
                        ]
                    )

                Rt[i, j, t] = numerator / denominator if denominator != 0 else np.nan

    # Sum over recipient dimension to get age-specific R_t by infector age group.
    R_sum = np.sum(Rt, axis=0)
    R = np.squeeze(R_sum)
    return R


def Gorji(
    confirm1,
    confirm2,
    confirm3,
    confirm4,
    confirm5,
    confirm6,
    contact: np.ndarray,
    sentence: str,
):
    """
    Convenience wrapper for computing age-specific Gorji R_t.

    Parameters
    ----------
    confirm1..confirm6 : array-like
        Incidence proxy (or "confirmation") time series for each age group.
        The first element is typically treated as an initial value and is excluded.
    contact : ndarray, shape (6, 6)
        Contact matrix.
    sentence : str
        Label used in log messages.

    Returns
    -------
    (R1, R2, R3, R4, R5, R6) : pandas.Series
        Age-specific R_t series for six age groups.
    """
    print("\n")
    print(f"Start Gorji Rt for {sentence}")

    date_leng = len(confirm1)

    df_confirm = pd.DataFrame(
        {
            "age1": confirm1[1:date_leng],
            "age2": confirm2[1:date_leng],
            "age3": confirm3[1:date_leng],
            "age4": confirm4[1:date_leng],
            "age5": confirm5[1:date_leng],
            "age6": confirm6[1:date_leng],
        }
    )

    # Serial interval distributions (discrete, truncated)
    spoint = 1
    epoint = 14
    si_num = 10_000_000
    n_age = 6
    sigma = 1.0 / 5.0  # mean infectious period ~ 5 days (kept as in the original code)
    max_pdf_length = 240

    # Generate random numbers (kept as in the original code: draws a fresh seed)
    np.random.seed()
    w = np.zeros((max_pdf_length, n_age), dtype=float)

    for j in range(n_age):
        si_samples = np.round(expon.rvs(scale=2 / sigma, size=si_num))
        si_dist_raw = pd.DataFrame({"si": si_samples})

        si_filtered = si_dist_raw[(si_dist_raw["si"] >= spoint) & (si_dist_raw["si"] <= epoint)]
        si_dist_freq = si_filtered.groupby("si").size().reset_index(name="freq")
        si_dist_freq["pdf"] = si_dist_freq["freq"] / si_dist_freq["freq"].sum()

        pdf_values = si_dist_freq["pdf"].values
        w[: len(pdf_values), j] = pdf_values

    # Estimate age-grouped instantaneous reproduction number (Gorji)
    data_columns = ["age1", "age2", "age3", "age4", "age5", "age6"]
    G_matrix100 = df_confirm[data_columns].values.T  # shape (6, T)
    b1 = np.ones(6, dtype=float)

    Rj = Rt_gorji(G_matrix100, w, contact, b1)
    valid_length = min(Rj.shape[1], date_leng - 1)

    gorji_rt = pd.DataFrame(
        {
            "age1": Rj[0, :valid_length],
            "age2": Rj[1, :valid_length],
            "age3": Rj[2, :valid_length],
            "age4": Rj[3, :valid_length],
            "age5": Rj[4, :valid_length],
            "age6": Rj[5, :valid_length],
        }
    )

    print(f"Complete Gorji Rt for {sentence}")
    print("\n")

    return (
        gorji_rt["age1"][:],
        gorji_rt["age2"][:],
        gorji_rt["age3"][:],
        gorji_rt["age4"][:],
        gorji_rt["age5"][:],
        gorji_rt["age6"][:],
    )
