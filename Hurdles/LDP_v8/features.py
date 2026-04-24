

import numpy as np
import pandas as pd


def _build_features_4bp(los_ev, valid_ev, beamid_ev):
    """
    Per-event 22-dim feature vector: 10 masked LOS + 10 validity + 2 sin/cos
    of beam ID (1..4). Inputs are event-rate arrays.
    """
    los_masked = los_ev * valid_ev
    angle = 2 * np.pi * (beamid_ev - 1) / 4
    sin_b = np.sin(angle).reshape(-1, 1)
    cos_b = np.cos(angle).reshape(-1, 1)
    return np.hstack([los_masked, valid_ev, sin_b, cos_b]).astype(np.float32)


def _lpf_firstorder(x, omega, dt):
    """Forward first-order low-pass. a = omega*dt / (1 + omega*dt)."""
    a = omega * dt / (1.0 + omega * dt)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y


def _per_beam_ffill(values, mask, fallback):
    v = np.where(mask, values, np.nan)
    filled = pd.Series(v).ffill().to_numpy()
    return np.where(np.isnan(filled), fallback, filled)


def _build_nowcast(los, valid, beamID, active_gates, gate_weights,
                   gate_distances, U_adv, cone_rad, n_beams, DT):
    """
    Lidar-only nowcast of REWS(t).

    For each active gate g at distance D_g, the LOS measured at time t predicts
    rotor wind at t + D_g/U_adv (Taylor frozen-turbulence assumption). Shifting
    those samples back by D_g/U_adv yields a past-aligned estimate of rotor
    wind at t. Per-beam forward-fill handles beam scanning; averaging across
    beams cancels cross-wind components. Weighted sum across gates uses the
    same weights as the KF fusion.
    """
    n_t = len(beamID)
    cos_a = np.cos(cone_rad)
    beam_int = beamID.astype(int)

    u_per_gate = np.full((10, n_t), U_adv, dtype=float)
    for g in range(1, 11):
        los_g = los[:, g - 1].astype(float)
        val_g = valid[:, g - 1].astype(bool)
        per_vals = np.zeros((n_beams, n_t))
        per_valf = np.zeros((n_beams, n_t), dtype=bool)
        for b in range(1, n_beams + 1):
            mask_b = val_g & (beam_int == b)
            per_vals[b - 1] = _per_beam_ffill(los_g, mask_b, U_adv)
            vff = pd.Series(np.where(mask_b, 1.0, np.nan)).ffill().to_numpy()
            per_valf[b - 1] = np.nan_to_num(vff, nan=0.0) > 0.5
        denom = per_valf.sum(axis=0).astype(float)
        numer = np.where(per_valf, per_vals, 0.0).sum(axis=0)
        u_per_gate[g - 1] = np.where(denom > 0,
                                     (numer / np.maximum(denom, 1)) / cos_a,
                                     U_adv)

    W_sum = 0.0
    nowcast = np.zeros(n_t)
    for i, g in enumerate(active_gates):
        D = gate_distances[g]
        shift = int(round((D / U_adv) / DT))
        u_now = np.full(n_t, U_adv)
        if shift < n_t:
            u_now[shift:] = u_per_gate[g - 1, :n_t - shift]
        nowcast += gate_weights[i] * u_now
        W_sum += gate_weights[i]
    return nowcast / max(W_sum, 1e-9)
