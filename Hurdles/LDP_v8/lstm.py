

import os
import glob
import numpy as np
import torch
import torch.nn as nn

from .features import _build_features_4bp


class LSTMPredictor(nn.Module):
    """Single-output LSTM for direct tau-seconds-ahead REWS prediction."""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def _find_checkpoints(models_dir, pattern):
    paths = sorted(glob.glob(os.path.join(models_dir, pattern.replace('{i}', '*'))))
    if not paths:
        raise FileNotFoundError(
            f'No bagged LSTM checkpoints matching {pattern} in {models_dir}. '
            f'Copy lstm_4bp_bag_*.pt into {models_dir}/ before running.')
    return paths


def _infer_one(model_path, features_raw, ev_idx, n_t, device):
    """Run a single LSTM: event-level predictions, ZOH-upsampled to n_t."""
    ck = torch.load(model_path, map_location=device, weights_only=False)
    model = LSTMPredictor(ck['n_features'], ck['hidden_size'], ck['num_layers'])
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    feat_mean = ck['feat_mean']
    feat_std = ck['feat_std']
    feat_std_s = np.where(feat_std > 1e-8, feat_std, 1.0)
    target_mean = float(ck['target_mean'])
    target_std = float(ck['target_std'])
    seq_len = int(ck['seq_len'])

    feat_norm = (features_raw - feat_mean) / feat_std_s
    pad = np.zeros((seq_len - 1, feat_norm.shape[1]), dtype=np.float32)
    feat_pad = np.vstack([pad, feat_norm])

    n_events = len(ev_idx)
    nxt_idx = np.append(ev_idx[1:], n_t)
    preds = np.empty(n_events, dtype=np.float32)

    with torch.no_grad():
        for e in range(0, n_events, 256):
            end = min(e + 256, n_events)
            w = np.stack([feat_pad[i:i + seq_len] for i in range(e, end)])
            out = model(torch.tensor(w)).cpu().numpy()
            preds[e:end] = out * target_std + target_mean

    lstm_100hz = np.empty(n_t)
    for e in range(n_events):
        lstm_100hz[ev_idx[e]:nxt_idx[e]] = preds[e]
    return lstm_100hz


def run_bagged_lstm(los, valid, beamID, models_dir, bag_pattern):
    """Uniform average over all bagged checkpoints at 100 Hz resolution."""
    device = torch.device('cpu')
    paths = _find_checkpoints(models_dir, bag_pattern)

    ev_mask = np.concatenate([[True], beamID[1:] != beamID[:-1]])
    ev_idx = np.where(ev_mask)[0]
    features = _build_features_4bp(
        los[ev_idx], valid[ev_idx].astype(float), beamID[ev_idx])

    n_t = len(beamID)
    acc = None
    for p in paths:
        pred = _infer_one(p, features, ev_idx, n_t, device)
        acc = pred if acc is None else acc + pred
    return acc / len(paths)
