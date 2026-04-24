

import numpy as np


class ScalarKF:
    def __init__(self, Q, R, cos_alpha):
        self.Q = Q
        self.R = R
        self.cos_a = cos_alpha
        self.x = None
        self.P = R / max(cos_alpha ** 2, 1e-12)

    def step(self, z_los, is_valid):
        H = self.cos_a
        if self.x is None:
            self.x = (z_los / H) if is_valid else 18.0
            return self.x
        self.P += self.Q
        if is_valid:
            S = H * H * self.P + self.R
            K = self.P * H / S
            self.x = self.x + K * (z_los - H * self.x)
            self.P = (1.0 - K * H) * self.P
        return self.x


class LPFilter:
    """Tustin-discretized first-order low-pass."""

    def __init__(self, DT, CornerFreq):
        wc = CornerFreq
        self.a1 = 2 + wc * DT
        self.a0 = wc * DT - 2
        self.b = wc * DT
        self._out_last = None
        self._in_last = None

    def __call__(self, x):
        if self._out_last is None:
            self._out_last = x
            self._in_last = x
        y = (1.0 / self.a1) * (-self.a0 * self._out_last
                               + self.b * x + self.b * self._in_last)
        self._in_last = x
        self._out_last = y
        return y


class Buffer:
    """Fixed-length delay line (ZOH pushback)."""

    def __init__(self, DT, T_buffer):
        self.nBuffer = 2000
        self._buf = None
        idx = int(np.floor(T_buffer / DT))
        self.Idx = min(max(idx, 1), self.nBuffer) - 1

    def __call__(self, x):
        if self._buf is None:
            self._buf = np.full(self.nBuffer, x)
        self._buf = np.insert(self._buf[:-1], 0, x)
        return self._buf[self.Idx]
