# Age-Structured Influenza Rt Estimation (SIR + Gorji + Particle Filter)

This repository contains research code for **age-structured influenza transmission** using:
- a deterministic **age-structured SIR** simulator,
- a **Gorji-style** age-specific instantaneous reproduction number estimator, and
- a **particle filter / particle smoother** approach for age-specific Rt (computationally heavy).

## Repository structure

- `main.py`  
  End-to-end script for the synthetic SIR example and Rt estimation (Gorji + particle).  
  The GitHub-friendly version removes hard-coded local paths and uses `INFLUENZA_PROJECT_ROOT`.

- `particle.py`  
  Particle filter/smoother estimator for age-specific Rt.  
  The GitHub-friendly version removes hard-coded local paths and translates remaining Korean comments.

- `SIR_calc.py`  
  One-step update for the age-structured SIR model (vectorized RK4-style scheme).

- `Gorji.py`  
  Gorji-style age-specific instantaneous reproduction number estimator.

## Data requirements

Place the following under `./data/` (recommended), or set an explicit project root:

- `data/Pops_Dec2022.csv` (population by age; read with `EUC-KR` encoding in the current scripts)

> If you keep a different filename/location, update it in the scripts.

## Configuration: project root

The scripts can locate your project directory in one of two ways:

1) **Recommended (GitHub-friendly):** set an environment variable:
```bash
export INFLUENZA_PROJECT_ROOT=/path/to/your/project
```

2) **Default:** if `INFLUENZA_PROJECT_ROOT` is not set, the scripts assume the project root is the same folder as the script file.

## Setup

Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -U pip
pip install numpy pandas scipy matplotlib
```

## Quick start

```bash
python main.py
```

Outputs are written under:
- `Pre_defined/data/` (relative to the project root)

Typical outputs:
- `pre_confirm.csv`, `pre_Rt.csv`, `Gorji_Rt.csv`
- `pre_age_confirm.csv`, `pre_Pf_Rt.csv`, `pre_Ps_Rt.csv`

## Performance notes

- `particle.py` is the main bottleneck.
- The original code sets `cases = 10**6` (particles), which can be very slow and memory-intensive.
  For quick tests, reduce `cases` before running.

## Reproducibility

Some code paths call `np.random.seed()` without a fixed seed, so results may vary run-to-run.
Set an explicit seed if you need deterministic output.

## License

Add a license file if you plan to publish publicly (e.g., MIT, BSD-3, Apache-2.0).
