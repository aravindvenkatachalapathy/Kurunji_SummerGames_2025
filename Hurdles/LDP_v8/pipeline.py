

import numpy as np

from .config import (
    ACTIVE_GATES, ANGLE_TO_CENTERLINE_DEG, GATE_PARAMS, GATE_WEIGHTS,
    GATE_DISTANCES, N_BEAMS, ENSEMBLE_ALPHA, MODELS_DIR, BAG_PATTERN,
    TAU, U_ADVECTION, SC_OMEGA_BIAS, SC_GAIN_MAX, SC_OMEGA_SNR, SC_EPS,
)
from .kalman import ScalarKF, LPFilter, Buffer
from .features import _build_nowcast, _lpf_firstorder
from .lstm import run_bagged_lstm


def _kf_branch(los, valid, beamID, DT, n_t):
    """Per-gate KF + LPF + Buffer, weighted sum across active gates."""
    cos_a = np.cos(np.deg2rad(ANGLE_TO_CENTERLINE_DEG))
    gates = ACTIVE_GATES
    weights = np.asarray(GATE_WEIGHTS, dtype=float)
    weights = weights / weights.sum()
    n_g = len(gates)

    kfs = [ScalarKF(GATE_PARAMS[g]['Q'], GATE_PARAMS[g]['R'], cos_a) for g in gates]
    lpfs = [LPFilter(DT, GATE_PARAMS[g]['omega_cutoff']) for g in gates]
    bufs = [Buffer(DT, GATE_PARAMS[g]['T_buffer']) for g in gates]

    rews_held = np.full(n_g, np.nan)
    per_gate_b = np.zeros((n_t, n_g))
    prev_beam = -1

    for i in range(n_t):
        if beamID[i] != prev_beam:
            for gi, g in enumerate(gates):
                col = g - 1
                rews_held[gi] = kfs[gi].step(los[i, col], bool(valid[i, col]))
            prev_beam = beamID[i]
        for gi in range(n_g):
            fi = lpfs[gi](rews_held[gi])
            per_gate_b[i, gi] = bufs[gi](fi)

    return per_gate_b @ weights


def _self_consistency(ens, los, valid, beamID, DT, n_t):
    """Variance-adaptive self-consistency correction of the ensemble."""
    nowcast = _build_nowcast(
        los, valid.astype(bool), beamID.astype(int),
        ACTIVE_GATES, np.asarray(GATE_WEIGHTS), GATE_DISTANCES,
        U_ADVECTION, np.radians(ANGLE_TO_CENTERLINE_DEG), N_BEAMS, DT)

    tau_samp = int(round(TAU / DT))
    ens_past = np.full(n_t, U_ADVECTION)
    if tau_samp < n_t:
        ens_past[tau_samp:] = ens[:n_t - tau_samp]
    err_rt = nowcast - ens_past

    slow_bias = _lpf_firstorder(err_rt, SC_OMEGA_BIAS, DT)
    slow_pow = _lpf_firstorder(slow_bias ** 2, SC_OMEGA_SNR, DT)
    total_pow = _lpf_firstorder(err_rt ** 2, SC_OMEGA_SNR, DT)
    snr = np.clip(slow_pow / (total_pow + SC_EPS), 0.0, 1.0)
    gain_t = SC_GAIN_MAX * snr
    return ens + gain_t * slow_bias


def LDP_v8(time, isValid, beamID, lineOfSightWindSpeed, DT, LDP=None):
    """
    Bagged LSTM + KF ensemble with variance-adaptive self-consistency.

    Parameters
    ----------
    time                 : (n_t,)       time vector [s]
    isValid              : (n_t, 10)    validity flags, all 10 gates (float 0/1)
    beamID               : (n_t,)       beam IDs (1-4)
    lineOfSightWindSpeed : (n_t, 10)    LOS wind speeds, all 10 gates [m/s]
    DT                   : float        time step [s]
    LDP                  : dict or None All hyperparameters live in config.py;
                                        LDP is accepted only for interface
                                        compatibility with the baseline LDP
                                        family. Pass None.

    Returns
    -------
    REWS, REWS_f, REWS_b : (n_t,)
        REWS is the pre-correction ensemble, REWS_f / REWS_b both hold the
        final corrected estimate (kept identical so existing evaluators that
        read REWS_b work unchanged).
    """
    n_t = len(time)
    beam_i = beamID.astype(int)

    kf_b = _kf_branch(lineOfSightWindSpeed, isValid, beam_i, DT, n_t)
    lstm_avg = run_bagged_lstm(lineOfSightWindSpeed, isValid, beam_i,
                               MODELS_DIR, BAG_PATTERN)
    ens = ENSEMBLE_ALPHA * lstm_avg + (1.0 - ENSEMBLE_ALPHA) * kf_b
    rews_b = _self_consistency(ens, lineOfSightWindSpeed, isValid, beam_i,
                               DT, n_t)

    return ens, rews_b, rews_b
