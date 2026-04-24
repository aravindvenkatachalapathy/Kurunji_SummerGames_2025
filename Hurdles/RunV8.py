"""
RunV8.py — evaluate LDP_v8 on the 18 m/s Hurdles competition seeds.

For each seed (1801-1806):
    1. Load 4BeamPulsed lidar data (all 10 gates) from solis_lidar_data/
    2. Run LDP_v8 -> REWS_b estimate
    3. Load wind-field REWS from TurbulentWind/ and compute the detrended-MAE
       cost over t >= t_start.
    4. Save per-seed estimate to estimates/URef_18_Seed_<S>_REWS_v8.csv
       and all seeds into estimates/URef_18_REWS_v8_all.mat
    5. Print per-seed MAE and aggregate cost.

Reference (competition):
    LDP_v3 baseline cost     : 0.515998
    LDP_v8 (this algorithm)  : ~ 0.4257
"""

import os
import numpy as np
import pandas as pd
from scipy.io import savemat

from LDP_v8 import LDP_v8

# ── Problem setup (must not change) ─────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIDAR_DIR = os.path.join(SCRIPT_DIR, 'solis_lidar_data')
WIND_DIR = os.path.join(SCRIPT_DIR, 'TurbulentWind')
OUT_DIR = os.path.join(SCRIPT_DIR, 'estimates')
os.makedirs(OUT_DIR, exist_ok=True)

LIDAR_TYPE = '4BeamPulsed'
N_SEEDS = 6
SEEDS = np.arange(1, N_SEEDS + 1) + 18 * 100      # 1801..1806

TMAX = 660.0
DT = 0.01
t_start = 60.0
tau = 2.0
time = np.arange(0, TMAX + DT, DT)


def load_seed(seed):
    """Load lidar (all 10 gates) and wind-field REWS for one seed."""
    solis_path = os.path.join(LIDAR_DIR,
                              f'URef_18_Seed_{seed:04d}_lidar_data_{LIDAR_TYPE}.csv')
    solis = pd.read_csv(solis_path)

    def reindex(col, fill='pad', fill_val=0.0):
        s = pd.Series(solis[col].values, index=solis['time']) \
              .reindex(time, method=fill)
        return (s.bfill() if fill == 'pad' else s.fillna(fill_val)).to_numpy()

    beamID = reindex('beamID')
    isValid = np.column_stack([reindex(f'isValid{g}', fill='nearest') for g in range(1, 11)])
    los = np.column_stack([reindex(f'lineOfSightWindSpeed{g}') for g in range(1, 11)])

    wind_path = os.path.join(WIND_DIR, f'URef_18_Seed_{seed:04d}.csv')
    rd = pd.read_csv(wind_path)
    ext_time = np.concatenate([rd['time'], rd['time'] + 600])
    ext_rews = np.concatenate([rd['REWS'], rd['REWS']])
    rews_wf = np.interp(time, ext_time, ext_rews)
    rews_wf_shifted = np.interp(time + tau, ext_time, ext_rews)
    return beamID, isValid, los, rews_wf, rews_wf_shifted


def detrended_mae(err, t_start_idx):
    e = err[t_start_idx:]
    return float(np.mean(np.abs(e - e.mean())))


def save_estimate_csv(seed, time_vec, rews_wf, rews_wf_shifted,
                      REWS, REWS_f, REWS_b):
    stem = f'URef_18_Seed_{seed:04d}_REWS_v8'
    df = pd.DataFrame({
        'time': time_vec,
        'REWS_WindField': rews_wf,
        'REWS_WindField_shifted': rews_wf_shifted,
        'REWS': REWS,
        'REWS_f': REWS_f,
        'REWS_b': REWS_b,
    })
    csv_path = os.path.join(OUT_DIR, stem + '.csv')
    df.to_csv(csv_path, index=False)
    return csv_path


def main():
    t_start_idx = int(t_start / DT)
    mae_per_seed = np.full(len(SEEDS), np.nan)
    all_seeds = {}

    print(f'LDP_v8 — {LIDAR_TYPE} — {len(SEEDS)} seeds')
    print('-' * 60)
    for i, seed in enumerate(SEEDS):
        beamID, isValid, los, rews_wf, rews_wf_shifted = load_seed(seed)

        REWS, REWS_f, REWS_b = LDP_v8(time, isValid, beamID, los, DT)

        err = rews_wf_shifted - REWS_b
        mae_per_seed[i] = detrended_mae(err, t_start_idx)

        all_seeds[f'seed{seed}'] = {
            'REWS_WindField': rews_wf,
            'REWS_WindField_shifted': rews_wf_shifted,
            'REWS': REWS,
            'REWS_f': REWS_f,
            'REWS_b': REWS_b,
            'MAE': mae_per_seed[i],
        }

        csv_path = save_estimate_csv(
            seed, time, rews_wf, rews_wf_shifted, REWS, REWS_f, REWS_b)
        print(f'  seed {seed}: MAE = {mae_per_seed[i]:.6f}   ->  '
              f'{os.path.relpath(csv_path, SCRIPT_DIR)}')

    mat_path = os.path.join(OUT_DIR, 'URef_18_REWS_v8_all.mat')
    savemat(mat_path, {'time': time, **all_seeds})
    print(f'All seeds saved to {os.path.relpath(mat_path, SCRIPT_DIR)}')

    cost = float(np.mean(mae_per_seed))
    print('-' * 60)
    print(f'Cost for Summer Games 2025 ("18 m/s hurdles"): {cost:.6f}')
    print(f'Baseline LDP_v3 reference: 0.515998  (delta {cost - 0.515998:+.6f})')


if __name__ == '__main__':
    main()
