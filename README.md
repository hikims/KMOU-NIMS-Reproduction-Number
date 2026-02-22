# Age-Structured Influenza Rt Estimation

This repository implements age-structured influenza transmission
modeling and time-varying reproduction number (Rt) estimation using:

-   Synthetic SIR simulations
-   NHIS-based epidemiological data
-   Gorji instantaneous Rt method
-   Particle filter / smoother estimation
-   Figure generation scripts for publication

------------------------------------------------------------------------

# Project Structure

## Synthetic

-   Synthetic_main_github.py\
-   Synthetic_SIR_calc_github.py\
-   Synthetic_Gorji_github.py\
-   Synthetic_particle_github.py

Purpose: - Generate synthetic epidemic curves - Compute theoretical
(pre-defined) Rt - Estimate Rt using Gorji method - Estimate Rt using
particle smoother - Save outputs as CSV files

Run: python Synthetic_main_github.py

Output directory: Pre_defined/data/

------------------------------------------------------------------------

## NHIS

-   NHIS_main_github.py\
-   NHIS_particle_github.py

Purpose: - Process NHIS influenza surveillance data - Compute KDCA
epidemiological weeks - Estimate instantaneous Rt - Apply particle
smoother for age-specific Rt

Run: python NHIS_main_github.py

Data location: ./data/

Optional environment variable: export PROJECT_PATH="your/project/root"

------------------------------------------------------------------------

## Figures

-   fig2_plot_github.py\
-   fig3_plot_github.py\
-   fig4_plot_github.py

### Fig 2

Synthetic validation figures: - Pre-defined vs Gorji Rt - Pre-defined vs
particle smoother Rt - Age-specific confirmations

Output: ./image/

### Fig 3

Season-level confirmation and Rt curves.

Output: ./paper_image/

### Fig 4

Age-specific Rt heatmaps across seasons.

Run: python fig4_plot_github.py

Output: ./paper_image/fig4.eps

------------------------------------------------------------------------

# Environment Variables

Optional configuration:

export INFLUENZA_PROJECT_ROOT="path/to/project" export
PROJECT_PATH="path/to/project" export DATA_DIR="path/to/data" export
OUT_DIR="path/to/output"

------------------------------------------------------------------------

# Dependencies

pip install numpy pandas scipy matplotlib

------------------------------------------------------------------------

# Model Overview

-   6 age groups
-   Age-structured contact matrix
-   Mean infectious period: \~4--5 days
-   Serial interval distribution (exponential-based)
-   Rt estimation methods:
    -   Theoretical SIR-based Rt
    -   Gorji instantaneous method
    -   Particle smoother estimation

------------------------------------------------------------------------

# Notes

-   All scripts are GitHub-ready (no hard-coded absolute paths)
-   Output directories are automatically created
-   Compatible with Python 3.9+

------------------------------------------------------------------------

# Citation

If you use this code, please cite your associated publication.
