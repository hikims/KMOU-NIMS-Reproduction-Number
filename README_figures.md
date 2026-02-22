# Figure plotting scripts (fig2 / fig3 / fig4)

This folder contains scripts used to generate paper figures.

## Files

- `fig2_plot.py`  
  Helper functions to plot synthetic confirmations and Rt comparisons:
  - confirmations by age
  - Rt by age (pre-defined vs comparator)
  - pre-defined Rt vs Gorji Rt
  - pre-defined Rt vs particle smoother Rt

- `fig3_plot.py`  
  Season-level plots for:
  - observed vs simulated confirmations (`*_confirmation_obs.csv` and `*_confirmation_sim.csv`)
  - age-specific Rt + all-ages Rt (`*_Ps_Rt.csv` and `*_all_Ps_Rt.csv`)

- `fig4_plot.py`  
  Heatmaps of age-specific Rt across seasons using `*_Ps_Rt.csv`.

## Data locations

By default, the GitHub-ready scripts look for data under:

- `./particle/data/`  (relative to repository root)

You can override paths via environment variables:

```bash
export PROJECT_PATH="."
export DATA_DIR="./particle/data"
export OUT_DIR="./paper_image"
```

## Outputs

- `fig2_plot.py` outputs EPS figures under `./image/` by default.
- `fig3_plot.py` outputs EPS figures under `./paper_image/` by default.
- `fig4_plot.py` writes `fig4.eps` under `./paper_image/` by default.

## Dependencies

```bash
pip install numpy pandas matplotlib scipy
```

## Usage examples

### Fig 3: confirmation & Rt for a season

```python
from fig3_plot import confirm_plotting, Rt_plotting

confirm_plotting(2016)  # 2016–2017 season
Rt_plotting(2016)
```

### Fig 4: heatmap

```bash
python fig4_plot.py
```

or with custom dirs:

```bash
DATA_DIR="./particle/data" OUT_DIR="./paper_image" python fig4_plot.py
```
