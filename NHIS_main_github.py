
import numpy as np
import pandas as pd
import time
import os
from scipy.stats import weibull_min
from scipy.special import gamma
from particle import ps_age_Rt, ps_all_Rt 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta
from pandas.api.types import is_datetime64_any_dtype as is_datetime
PATH = os.environ.get('PROJECT_PATH', '.')  # Set via env var; defaults to current directory
def kdca_week(dt):

    # Function to compute epidemiological week based on KDCA (Korea Disease Control and Prevention Agency) definition
    # - The week containing January 1st (Sunday–Saturday) is defined as Week 1
    # - The last few days of the previous year that fall in the same week as January 1st
    #   are counted as Week 1 of the next year
      
    # Convert pandas.Timestamp → datetime.date
    if isinstance(dt, pd.Timestamp):
        dt = dt.date()

    cur_year = dt.year
    fdcy = date(cur_year, 1, 1)  # First day of the current year

    day_of_week_fdcy = (fdcy.weekday() + 1) % 7

    # Compute KDCA week
    days_diff = (dt - fdcy).days
    kdca_week = (days_diff + day_of_week_fdcy) // 7 + 1
    kdca_year = cur_year

    # First day of the next year
    next_year = cur_year + 1
    fdny = date(next_year, 1, 1)
    day_of_week_fdny = (fdny.weekday() + 1) % 7

    # If the date belongs to Week 1 of the next year
    if (fdny - dt).days <= day_of_week_fdny:
        kdca_year = next_year
        kdca_week = 1

    return kdca_year, kdca_week


# instantaneous reproduction number for all age
def All_instantaneous(array, all_w):
    print(f"\nStart All instantaneous Rt")
    
    array = np.asarray(array).flatten()  # Flatten to 1D regardless of whether input is a DataFrame or NumPy array
    all_instantaneous_R = np.zeros(len(array))

    for i in range(len(all_instantaneous_R)):
        if i < len(all_w):  
            if i == 0:
                all_instantaneous_R[i] = array[i]
            else:
                denominator = np.sum(array[:i][::-1] * all_w[:i])
                all_instantaneous_R[i] = array[i] / denominator if denominator != 0 else 0
        else:
            denominator = np.sum(array[(i - len(all_w)):i][::-1] * all_w)
            all_instantaneous_R[i] = array[i] / denominator if denominator != 0 else 0

    print(f"Complete All instantaneous Rt\n")
    return all_instantaneous_R

def plot_inst_Rt(start_year: int, ts_mod: pd.DataFrame, si_dist: pd.DataFrame, R_inst):
    # Standardize internal column names
    ts_mod = ts_mod.rename(columns={"kdca_year": "year", "kdca_week": "week"}).copy()
    ts_mod["date"] = pd.to_datetime(ts_mod["date"])

    # Select seasonal window
    ts_tmp = ts_mod[
        ((ts_mod["year"] == start_year) & (ts_mod["week"] >= 34)) |
        ((ts_mod["year"] == start_year + 1) & (ts_mod["week"] <= 36))
    ].copy()

    # Compute Rt
    Rt_full = R_inst(ts_tmp["N"].to_numpy(), si_dist["pdf"].to_numpy())

    # Use central time window only
    results = ts_tmp.iloc[14:-7].copy()
    results["Rt"] = Rt_full[14:-7]
    results = results.reset_index(drop=True)

    # ==== Plot based on calendar date ====
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # === 1) Daily N ===
    ax1.plot(results["date"], results["N"], color="steelblue", linewidth=1.5)
    ax1.set_title(f"Flu ({start_year}-{start_year + 1})", fontsize=18, fontweight="bold")
    ax1.set_ylabel("Daily Flu Cases", fontsize=15, fontweight="bold")
    ax1.tick_params(axis="y", pad=2)
    ax1.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
    # Keep axes spines visible (for clarity)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)

    # === 2) Rt ===
    ax2.plot(results["date"], results["Rt"], color="darkorange", linewidth=2)
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax2.set_xlabel("Month", fontsize=15, fontweight="bold")
    ax2.set_ylabel(r"$R_t^{inst}$", fontsize=17, fontweight="bold")
    ax2.set_yticks(np.arange(0, 3.1, 0.5))
    ax2.set_ylim(0, 3)
    ax2.tick_params(axis="y", pad=2)
    ax2.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)

    # === X-axis: monthly ticks (Sep, Nov, Jan, Mar, May, Jul) ===
    month_locator = mdates.MonthLocator(bymonth=[9, 11, 1, 3, 5, 7])
    month_formatter = mdates.DateFormatter("%b")
    ax2.xaxis.set_major_locator(month_locator)
    ax2.xaxis.set_major_formatter(month_formatter)
    ax2.set_xlim(results["date"].min(), results["date"].max())

    plt.tight_layout()
    return fig, results

# instantaneous reproduction number for age groups
def Age_instantaneous(mat, s_interval, M, beta):
    print(f"\nStart Age instantaneous Rt")
    n_rows_M = M.shape[0]  
    n_cols_M = M.shape[1]  
    n_cols_mat = mat.shape[1]  

    Rt = np.zeros((n_rows_M, n_cols_M, n_cols_mat))

    for i in range(n_rows_M):
        for j in range(n_cols_M):
            for t in range(n_cols_mat):
                if t < len(s_interval): 
                    weight_vector = s_interval[:t + 1, j] if t > 0 else np.array([1])  
                
                    numerator = M[i, j] * beta[j] * mat[i, t]
                    denominator = np.sum([M[i, k] * beta[k] * np.sum(np.flip(mat[k, :(t + 1)]) * weight_vector) 
                                      for k in range(n_cols_M)])
                
                    if denominator != 0:
                        Rt[i, j, t] = numerator / denominator
                    else:
                        Rt[i, j, t] = np.nan  

                else:
                    recent_weights = s_interval[:, j]  
                    numerator = M[i, j] * beta[j] * mat[i, t]
                    denominator = np.sum([M[i, k] * beta[k] * np.sum(np.flip(mat[k, (t - len(recent_weights)):t]) * recent_weights) 
                                      for k in range(n_cols_M)])
                
                    if denominator != 0:
                        Rt[i, j, t] = numerator / denominator
                    else:
                        Rt[i, j, t] = np.nan 
                    
    R_sum = np.sum(Rt, axis = 0)
    R = np.squeeze(R_sum)

    print(f"Complete Age instantaneous Rt\n")
    return R


def plot_inst_age_Rt(start_year: int):
    # --- Seasonal subset ---
    ts_tmp = ts_mod[
        ((ts_mod["kdca_year"] == start_year) & (ts_mod["kdca_week"] >= 34)) |
        ((ts_mod["kdca_year"] == start_year + 1) & (ts_mod["kdca_week"] <= 36))
    ].copy()

    ts_agg_tmp = ts_agg_mod[
        ((ts_agg_mod["year"] == start_year) & (ts_agg_mod["week"] >= 34)) |
        ((ts_agg_mod["year"] == start_year + 1) & (ts_agg_mod["week"] <= 36))
    ].copy()

    # --- Age-specific incidence matrix (6 x T) ---
    ts_mat = ts_agg_tmp.iloc[:, 3:9].to_numpy().T

    # --- Contact matrix / transmission scaling (beta) ---
    M = np.array([
        [1.76, 0.21, 0.03, 0.20, 0.05, 0.05],
        [0.33, 3.75, 0.30, 0.31, 0.14, 0.13],
        [0.05, 0.31, 3.65, 0.22, 0.31, 0.12],
        [2.07, 2.11, 1.38, 1.78, 1.26, 0.84],
        [0.47, 0.88, 1.87, 1.18, 1.97, 1.47],
        [0.30, 0.47, 0.42, 0.45, 0.83, 2.50],
    ], dtype=float)
    beta = np.ones(6) - 0.5 * np.array([0.757, 0.748, 0.46, 0.295, 0.489, 0.826], dtype=float)

    # --- Serial interval distribution ---
    w = si_dist["pdf"].to_numpy().astype(float)      # (L,)
    s_interval = np.tile(w.reshape(-1, 1), (1, 6))   # (L, 6)

    # --- Compute age-specific Rt (6 x T) ---
    Rj = Age_instantaneous(ts_mat, s_interval, M, beta)

    # --- Build results frame: take 'N' from ts_tmp ---
    # Use only the central interval: 14:-7 (0-based)
    # ts_tmp contains 'date' and 'N' (and kdca_year/week if needed)
    results = ts_tmp.iloc[14:-7, :].copy()  # results["N"] is available here

    # All-ages Rt
    Rt_all_full = All_instantaneous(ts_tmp["N"].to_numpy(), w)
    results["Rt"] = Rt_all_full[14:-7]

    # Add age-group-specific Rt columns
    results["Rt_agg1"] = Rj[0, 14:-7]
    results["Rt_agg2"] = Rj[1, 14:-7]
    results["Rt_agg3"] = Rj[2, 14:-7]
    results["Rt_agg4"] = Rj[3, 14:-7]
    results["Rt_agg5"] = Rj[4, 14:-7]
    results["Rt_agg6"] = Rj[5, 14:-7]

    # ==== Plot based on calendar date ====
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # === 1) Daily N ===
    ax1.plot(results["date"], results["N"], color="steelblue", linewidth=1.5)
    ax1.set_title(f"Flu ({start_year}-{start_year + 1})", fontsize=18, fontweight="bold")
    ax1.set_ylabel("Daily Flu Cases", fontsize=15, fontweight="bold")
    ax1.tick_params(axis="y", pad=2)
    ax1.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)

    # === 2) Rt (all ages + age groups) ===
    series_list = [("All", "Rt")]
    for i in range(1, 7):
        col = f"Rt_agg{i}"
        if col in results.columns:
            series_list.append((f"Age {i}", col))

    color_map = {
        "All": "darkred",
        "Age 1": "red",
        "Age 2": "brown",
        "Age 3": "orange",
        "Age 4": "green",
        "Age 5": "blue",
        "Age 6": "purple",
    }

    ax2.plot(results["date"], results["Rt"], color=color_map["All"], linewidth=2.2, label="All")
    for name, col in series_list[1:]:
        ax2.plot(results["date"], results[col], linewidth=1.3, label=name, color=color_map.get(name, None))

    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1)

    # Automatic y-axis padding
    all_vals = [results[col].to_numpy(dtype=float) for _, col in series_list if col in results.columns]
    if all_vals:
        vmax = np.nanmax(np.concatenate(all_vals))
        y_top = max(3.0, float(vmax) * 1.15)
        ax2.set_ylim(0, y_top)

    ax2.set_xlabel("Month", fontsize=15, fontweight="bold")
    ax2.set_ylabel(r"$R_t^{inst}$", fontsize=17, fontweight="bold")
    ax2.set_yticks(np.arange(0, ax2.get_ylim()[1] + 1e-9, 0.5))
    ax2.tick_params(axis="y", pad=2)
    ax2.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)

    month_locator = mdates.MonthLocator(bymonth=[9, 11, 1, 3, 5, 7])
    month_formatter = mdates.DateFormatter("%b")
    ax2.xaxis.set_major_locator(month_locator)
    ax2.xaxis.set_major_formatter(month_formatter)
    ax2.set_xlim(results["date"].min(), results["date"].max())

    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0., frameon=False, ncol=1)

    plt.tight_layout()
    return fig, results


np.random.seed()
start = time.time()

# Population
fn = PATH + '/data/Pops_Dec2022.csv'
with open(fn, encoding='EUC-KR') as file:
    pop_raw = pd.read_csv(file)

pops = [int(s.replace(",", "")) for s in pop_raw.iloc[0, 3:]]

lbeag = [0, 7, 13, 19, 50, 65]
ubeag = [6, 12, 18, 49, 64, 100]  
population = [sum(pops[lbeag[i]:ubeag[i]+1]) for i in range(6)]

# NHIS
csv_path = os.path.join(PATH, 'data/NHIS_Flu.csv')
NHIS_Flu = pd.read_csv(csv_path)

ts = NHIS_Flu.groupby("date", as_index=False)["N"].sum()

# ================================
# Process NHIS_Flu dataframe
# ================================

# Fill missing dates (set N=0 for missing days)
ts["date"] = pd.to_datetime(ts["date"])
date_range = pd.date_range(ts["date"].min(), ts["date"].max(), freq="D")
ts_full = pd.DataFrame({"date": date_range})
ts = pd.merge(ts_full, ts, on="date", how="left").fillna({"N": 0})

# Compute KDCA year/week
ts["kdca_year"], ts["kdca_week"] = zip(*ts["date"].apply(kdca_week))

# Sort by date
ts = ts.sort_values("date").reset_index(drop=True)

# 7-day moving average (weekday adjustment)
ts_mod = ts.copy()
ts_mod["N"] = ts_mod["N"].rolling(window=7, center=True).mean()

# Remove boundary days 
ts_mod = ts_mod.iloc[3:-3][["date", "kdca_year", "kdca_week", "N"]]

# # Sanity check
# print(ts_mod.head())

# ================================
# Serial interval distribution
# ================================

# Set random seed
np.random.seed(1)

# Set Weibull distribution parameters
mean_Weibull = 3.6
stdv_Weibull = 1.6

shape_Weibull = (stdv_Weibull / mean_Weibull) ** (-1.086)
scale_Weibull = mean_Weibull / gamma(1 + 1 / shape_Weibull)

# Generate 1,000,000 Weibull samples and round
si_values = np.round(weibull_min.rvs(shape_Weibull, scale=scale_Weibull, size=1_000_000))

# Filter values to 1–10
si_dist_raw = pd.DataFrame({'si': si_values})
si_dist = si_dist_raw[(si_dist_raw['si'] >= 1) & (si_dist_raw['si'] <= 10)]

# Compute frequency and pdf
si_dist = si_dist.groupby('si', as_index=False).size().rename(columns={'size': 'freq'})
si_dist['pdf'] = si_dist['freq'] / si_dist['freq'].sum()
    
    
# ================================
# Age instantaneous Rt
# ================================

# agg: 1 (0-5), 2 (6-11), 3 (12-17), 4 (18-44), 5 (45-64), 6 (65+)
NHIS_Flu["date"] = pd.to_datetime(NHIS_Flu["date"])
# Daily totals by age group
ts_agg = (
    NHIS_Flu
    .groupby(["date", "agg"], as_index=False)["N"]
    .sum()
)

# Reindex to a full (date, age group) grid so zero-count days/groups are not missing
date_min, date_max = ts_agg["date"].min(), ts_agg["date"].max()
full_dates = pd.date_range(date_min, date_max, freq="D")
full_index = pd.MultiIndex.from_product([full_dates, range(1, 7)], names=["date", "agg"])

ts_agg = (
    ts_agg
    .set_index(["date", "agg"])
    .reindex(full_index)           # Fill missing (date, agg)
    .fillna(0)                     # Missing N = 0
    .reset_index()
)
ts_agg.columns = ["date", "agg", "N"]

# Compute KDCA year/week
years_weeks = ts_agg["date"].apply(kdca_week)
ts_agg["year"] = [yw[0] for yw in years_weeks]
ts_agg["week"] = [yw[1] for yw in years_weeks]

# Sort
ts_agg = ts_agg.sort_values(["date", "agg"]).reset_index(drop=True)

# Weekday effect adjustment: 7-day moving average (center=True)
ts_agg["N_roll"] = (
    ts_agg
    .sort_values(["agg", "date"])
    .groupby("agg")["N"]
    .transform(lambda s: s.rolling(window=7, center=True).mean())
)

# Combine into a single table (columns N_agg1 ~ N_agg6)
# Pivot by date index → each column is the 7-day mean for an age group
ts_agg_pivot = (
    ts_agg
    .pivot(index="date", columns="agg", values="N_roll")
    .rename(columns={i: f"N_agg{i}" for i in range(1, 7)})
    .sort_index()
)

# Recompute and attach KDCA year/week (date-based)
years_weeks_idx = ts_agg_pivot.index.to_series().apply(kdca_week)
ts_agg_mod = ts_agg_pivot.reset_index().copy()
ts_agg_mod["year"] = [yw[0] for yw in years_weeks_idx]
ts_agg_mod["week"] = [yw[1] for yw in years_weeks_idx]

ts_agg_mod = ts_agg_mod.iloc[3:-3, :]

cols = ["date", "year", "week"] + [f"N_agg{i}" for i in range(1, 7)]
cols_existing = [c for c in cols if c in ts_agg_mod.columns]
ts_agg_mod = ts_agg_mod[cols_existing].reset_index(drop=True)


# ================================
# Age-structured SIR Rt (boundary padding + save/plot Rt and confirmations together)
# ================================

fig_path_SIR_age = os.path.join(PATH, 'particle', 'image', 'Age_SIR')
fig_path_CONF_age = os.path.join(PATH, 'particle', 'image', 'Age_Conf')  # ← Output directory for confirmation plots
os.makedirs(fig_path_SIR_age, exist_ok=True)
os.makedirs(fig_path_CONF_age, exist_ok=True)

save_dir = os.path.join(PATH, 'particle', 'data')
os.makedirs(save_dir, exist_ok=True)

pad_days = 30  # Boundary padding (slightly larger than smoother lag / serial interval length)

for start_year in range(2009, 2023):
    # --- Define the core seasonal window (week 36 to week 36 of the next year) ---
    core_mask = (
        ((ts_mod["kdca_year"] == start_year) & (ts_mod["kdca_week"] >= 36)) |
        ((ts_mod["kdca_year"] == start_year + 1) & (ts_mod["kdca_week"] <= 36))
    )
    core = ts_mod.loc[core_mask].copy().reset_index(drop=True)
    if core.empty:
        print(f"[WARN] No core window for season {start_year}-{start_year+1}")
        continue

    core_start = core["date"].iloc[0]
    core_end   = core["date"].iloc[-1]

    # --- Define the full window including padding ---
    pad_start = core_start - pd.Timedelta(days=pad_days)
    pad_end   = core_end   + pd.Timedelta(days=pad_days)

    win_mod_mask = (ts_mod["date"]   >= pad_start) & (ts_mod["date"]   <= pad_end)
    win_agg_mask = (ts_agg_mod["date"] >= pad_start) & (ts_agg_mod["date"] <= pad_end)

    ts_win     = ts_mod.loc[win_mod_mask].copy().reset_index(drop=True)       # All ages (total)
    ts_agg_win = ts_agg_mod.loc[win_agg_mask].copy().reset_index(drop=True)   # By age group

    T = len(ts_win)
    if T < 2*pad_days + 10:
        print(f"[WARN] Window too small for season {start_year}, T={T}")
        continue

    core_slice = slice(pad_days, T - pad_days)       # Use only the central window
    dates_core = ts_win["date"].iloc[core_slice].reset_index(drop=True)

    # ------------ Observations (7-day rolling mean) ------------
    for i in range(1, 7):
        col = f"N_agg{i}"
        if col not in ts_agg_win.columns:
            ts_agg_win[col] = 0.0

    # Full padded window (for model computation)
    obs1_win = ts_agg_win["N_agg1"].to_numpy(float)
    obs2_win = ts_agg_win["N_agg2"].to_numpy(float)
    obs3_win = ts_agg_win["N_agg3"].to_numpy(float)
    obs4_win = ts_agg_win["N_agg4"].to_numpy(float)
    obs5_win = ts_agg_win["N_agg5"].to_numpy(float)
    obs6_win = ts_agg_win["N_agg6"].to_numpy(float)

    # Slice the central core (1:1 with dates) — observations
    obs1 = obs1_win[core_slice]; obs2 = obs2_win[core_slice]; obs3 = obs3_win[core_slice]
    obs4 = obs4_win[core_slice]; obs5 = obs5_win[core_slice]; obs6 = obs6_win[core_slice]

    # ------------ All-ages Rt (black dashed line) ------------
    confirm_all_win = ts_win["N"].to_numpy(dtype=float)
    Pf_Rt_all_win, Ps_Rt_all_win = ps_all_Rt(confirm_all_win, f"{start_year}_All")
    Pf_Rt_all = Pf_Rt_all_win[core_slice]
    Ps_Rt_all = Ps_Rt_all_win[core_slice]

    # ------------ Age-group Rt & confirmations (model) ------------
    Lwin = len(obs1_win)
    (
        sim_c1_win, sim_c2_win, sim_c3_win,
        sim_c4_win, sim_c5_win, sim_c6_win,
        age_pf_Rt1_win, age_pf_Rt2_win, age_pf_Rt3_win,
        age_pf_Rt4_win, age_pf_Rt5_win, age_pf_Rt6_win,
        age_ps_Rt1_win, age_ps_Rt2_win, age_ps_Rt3_win,
        age_ps_Rt4_win, age_ps_Rt5_win, age_ps_Rt6_win,
    ) = ps_age_Rt(
        Lwin,
        obs1_win, obs2_win, obs3_win,
        obs4_win, obs5_win, obs6_win,
        f"{start_year}_Age_win",
    )

    # Take core window only — Rt
    Pf_Rt_ages = pd.DataFrame({
        "age1": age_pf_Rt1_win[core_slice],
        "age2": age_pf_Rt2_win[core_slice],
        "age3": age_pf_Rt3_win[core_slice],
        "age4": age_pf_Rt4_win[core_slice],
        "age5": age_pf_Rt5_win[core_slice],
        "age6": age_pf_Rt6_win[core_slice],
    }).reset_index(drop=True)

    Ps_Rt_ages = pd.DataFrame({
        "age1": age_ps_Rt1_win[core_slice],
        "age2": age_ps_Rt2_win[core_slice],
        "age3": age_ps_Rt3_win[core_slice],
        "age4": age_ps_Rt4_win[core_slice],
        "age5": age_ps_Rt5_win[core_slice],
        "age6": age_ps_Rt6_win[core_slice],
    }).reset_index(drop=True)

    # Take core window only — confirmations (simulated)
    sim1 = np.asarray(sim_c1_win, float)[core_slice]
    sim2 = np.asarray(sim_c2_win, float)[core_slice]
    sim3 = np.asarray(sim_c3_win, float)[core_slice]
    sim4 = np.asarray(sim_c4_win, float)[core_slice]
    sim5 = np.asarray(sim_c5_win, float)[core_slice]
    sim6 = np.asarray(sim_c6_win, float)[core_slice]

    # ------------ Save CSV ------------
    df=pd.DataFrame(Ps_Rt_all)
    df.to_csv(os.path.join(save_dir, f"{start_year}_all_Ps_Rt.csv"), index=False)

    