"""
SIR model one-step update (vectorized RK4-like scheme used in the original project).

This module keeps the original public function signature so it can replace the
existing SIR_calc.py directly.
"""

from __future__ import annotations

import numpy as np


def SIR_calc(S: np.ndarray, I: np.ndarray, R: np.ndarray, N: np.ndarray, M: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute a single time-step update for an age-structured SIR model.

    Parameters
    ----------
    S, I, R : ndarray, shape (n_age,)
        Current compartment values.
    N : ndarray, shape (n_age,)
        Population by age group.
    M : ndarray, shape (n_age, n_age)
        Contact matrix.
    b : ndarray, shape (n_age,)
        Transmission-rate parameter for each age group at the current time.
    sigma : float
        Recovery rate (1 / mean infectious period).

    Returns
    -------
    output : ndarray, shape (3, n_age)
        Stacked array: [S_next, I_next, R_increment] where R_increment corresponds to the
        RK4-averaged increment used by the original code.
    """
    kS1 = -b * (S / N) * (M @ I)
    kI1 = b * (S / N) * (M @ I) - sigma * I
    kR1 = sigma * I

    S2 = S + kS1 / 2
    I2 = I + kI1 / 2
    kS2 = -b * (S2 / N) * (M @ I2)
    kI2 = b * (S2 / N) * (M @ I2) - sigma * I2
    kR2 = sigma * I2

    S3 = S + kS2 / 2
    I3 = I + kI2 / 2
    kS3 = -b * (S3 / N) * (M @ I3)
    kI3 = b * (S3 / N) * (M @ I3) - sigma * I3
    kR3 = sigma * I3

    S4 = S + kS3
    I4 = I + kI3
    kS4 = -b * (S4 / N) * (M @ I4)
    kI4 = b * (S4 / N) * (M @ I4) - sigma * I4
    kR4 = sigma * I4

    S_next = S + (kS1 + 2 * kS2 + 2 * kS3 + kS4) / 6
    I_next = I + (kI1 + 2 * kI2 + 2 * kI3 + kI4) / 6

    # Keep the original behavior: return the RK4-averaged increment for R.
    R_increment = (kR1 + 2 * kR2 + 2 * kR3 + kR4) / 6

    return np.stack([S_next, I_next, R_increment])
