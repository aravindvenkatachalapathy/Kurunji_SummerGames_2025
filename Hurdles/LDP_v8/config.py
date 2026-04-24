"""
LDP_v8 tuned hyperparameters.

All constants in one place so they can be reviewed / overridden without
touching the algorithmic code.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PKG_DIR)                       # .../Hurdles
MODELS_DIR = os.path.join(REPO_DIR, 'models')             # bagged checkpoints live here

# ── 4BeamPulsed lidar geometry ─────────────────────────────────────────────
N_BEAMS = 4
ANGLE_TO_CENTERLINE_DEG = 19.176
ACTIVE_GATES = [1, 3, 4, 8, 10]
GATE_DISTANCES = {1: 70, 2: 90, 3: 110, 4: 130, 5: 150,
                  6: 160, 7: 170, 8: 180, 9: 190, 10: 200}

# ── Kalman multi-gate fusion (per-gate) ────────────────────────────────────
# Tuned on seeds 1807-1830 minimizing detrended MAE.
GATE_PARAMS = {
    1:  {'Q': 10.00000, 'R': 1.0, 'omega_cutoff': 0.16500, 'T_buffer': 0.00000},
    3:  {'Q':  1.38950, 'R': 1.0, 'omega_cutoff': 0.16500, 'T_buffer': 0.00000},
    4:  {'Q':  0.37276, 'R': 1.0, 'omega_cutoff': 0.16500, 'T_buffer': 0.00000},
    8:  {'Q':  0.00373, 'R': 1.0, 'omega_cutoff': 0.27000, 'T_buffer': 0.00000},
    10: {'Q':  0.00720, 'R': 1.0, 'omega_cutoff': 0.20000, 'T_buffer': 1.18750},
}
# MAE-optimal non-negative weights over the 5 active gates
GATE_WEIGHTS = [0.103884, 0.038280, 0.204860, 0.217367, 0.435609]

# ── Ensemble of KF and averaged LSTM ───────────────────────────────────────
ENSEMBLE_ALPHA = 0.5     # weight on LSTM; (1-alpha) on KF. Optimal by sweep.

# ── Bagged LSTM ────────────────────────────────────────────────────────────
BAG_SIZE = 10
BAG_PATTERN = 'lstm_4bp_bag_{i}.pt'

# ── Self-consistency residual tracker ──────────────────────────────────────
# Observable error = nowcast - ens(t-tau).
# slow_bias(t)  = LPF(err_rt, SC_OMEGA_BIAS)
# snr(t)        = LPF(slow_bias^2) / LPF(err_rt^2)     (both at SC_OMEGA_SNR)
# gain(t)       = SC_GAIN_MAX * clip(snr, 0, 1)
# REWS_b(t)     = ensemble(t) + gain(t) * slow_bias(t)
TAU = 2.0                # [s] preview horizon to match competition metric
U_ADVECTION = 18.0       # [m/s] nominal wind speed for Taylor shift
SC_OMEGA_BIAS = 0.5      # rad/s, LPF bandwidth for slow drift extraction
SC_GAIN_MAX   = 0.15     # ceiling on correction gain
SC_OMEGA_SNR  = 0.1      # rad/s, variance smoothing bandwidth
SC_EPS        = 1e-3
