# Influenza Rt Estimation (KDCA week) + Particle-Filter SIR

This repository contains Python code to compute:
- **Instantaneous Rt** from incidence data using a **serial interval distribution** (Weibull-based).
- **Age-specific Rt** using an age-structured formulation.
- **Particle-filter SIR Rt** estimates (all ages and age-stratified), with seasonal windows and optional padding.

> Notes
> - The original scripts were written for a local directory layout; for GitHub, the code uses a configurable project root via an environment variable.

## Repository structure

```
.
├── main.py            # data preprocessing + instantaneous Rt + seasonal PF runs
├── particle.py        # particle-filter SIR implementations (all / age)
└── data/
    ├── NHIS_Flu.csv
    └── Pops_Dec2022.csv
```

Outputs (created automatically):
```
./particle/
  ├── data/            # saved CSV results
  └── image/
      ├── Age_SIR/     # age Rt plots
      ├── Age_Conf/    # age confirmation plots
      ├── All_SIR/     # all-age Rt plots (if enabled in script)
      └── All_Conf/    # all-age confirmation plots (if enabled in script)
```

## Requirements

Python 3.9+ is recommended.

Install dependencies:
```bash
pip install numpy pandas scipy matplotlib
```

## Data format

### `data/NHIS_Flu.csv`
Expected columns:
- `date`: date string (e.g., `YYYY-MM-DD`)
- `N`: daily influenza cases (or incidence proxy)
- `agg`: age-group index (1..6) for age-specific analysis

### `data/Pops_Dec2022.csv`
A single-row CSV used to construct population by age.
The current script reads population counts from columns `3:` (i.e., after the first 3 columns).

## Configuration

Both scripts use `PROJECT_PATH` as the project root. By default it is the **current directory**.

Examples:
```bash
export PROJECT_PATH="."
# or
export PROJECT_PATH="/absolute/path/to/your/project"
```

If you keep the default repository layout, you can simply run without setting it.

## Usage

Run the main pipeline:
```bash
python main.py
```

Typical workflow inside `main.py`:
1. Load population and case data.
2. Fill missing dates, compute KDCA epidemiological year/week.
3. Apply 7-day centered moving average.
4. Build a Weibull-based serial interval distribution (1–10 days).
5. Compute instantaneous Rt (all ages + age groups).
6. Run particle-filter SIR over seasons (e.g., 2009–2022) and save figures/CSVs.

## Reproducibility / notes

- The scripts use `np.random.seed()` in multiple places. For fully reproducible runs, consider setting a fixed global seed once.
- Some PF settings can be computationally heavy (e.g., `cases = 10**6`). Adjust `cases`, `sigma_rw`, etc., in `particle.py` if needed.

## Citation / attribution

If you use this code in academic work, please cite your related manuscript and data sources (NHIS / KDCA definitions) as appropriate.

## License

Add a license of your choice (e.g., MIT) before making the repository public.
