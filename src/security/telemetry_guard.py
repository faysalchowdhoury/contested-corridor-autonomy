"""Physics-guided telemetry & V2X anomaly verification.

Three complementary checks, all calibrated on the clean training split:

1. **Stale / replay detection** - runs of unchanged telemetry (held or replayed
   values) flagged once they exceed the training staleness scale.
2. **Statistical distribution shift** - per-feature robust z-score versus the
   training distribution.
3. **Physics-residual cross-check** - LWR continuity residual between adjacent
   nodes; telemetry that violates flow conservation relative to neighbours is
   suspect even when each value looks individually plausible.

Outputs ``integrity_score`` in [0, 1], a binary ``anomaly_flag`` and a
``quarantine_mask`` so Dec-PINC can fall back to physics-propagated estimates
instead of ingesting spoofed inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..models.physics import upstream_index

EPS = 1e-3


@dataclass
class GuardReport:
    integrity_score: np.ndarray    # [T, N] in [0, 1]
    anomaly_flag: np.ndarray       # [T, N] bool
    quarantine_mask: np.ndarray    # [T, N] bool
    components: Dict[str, np.ndarray]


class TelemetryGuard:
    def __init__(self, target_fpr: float = 0.05,
                 w_phys: float = 1.0, w_shift: float = 1.0, w_stale: float = 0.7):
        self.target_fpr = target_fpr
        self.w = (w_phys, w_shift, w_stale)

    # -- calibration on clean training telemetry ------------------------------
    def fit(self, data, edges) -> "TelemetryGuard":
        self.up = upstream_index(data.adjacency, edges)
        tr0, tr1 = data.splits["train"]
        self.sp_med = np.median(data.speed[tr0:tr1], axis=0)
        self.sp_mad = _mad(data.speed[tr0:tr1]) + EPS
        self.fl_med = np.median(data.flow[tr0:tr1], axis=0)
        self.fl_mad = _mad(data.flow[tr0:tr1]) + EPS
        # physics residual scale on clean train data
        r = self._phys_residual(data.speed[tr0:tr1], data.flow[tr0:tr1])
        self.phys_scale = np.median(np.abs(r)) + EPS
        # calibrate integrity threshold to target FPR on clean train split
        rep = self._raw_report(data.speed[tr0:tr1], data.flow[tr0:tr1])
        self.tau = float(np.quantile(rep["integrity"], self.target_fpr))
        return self

    def _phys_residual(self, speed, flow):
        s = np.clip(speed, EPS, None)
        rho = flow / s
        q_in = flow[:, self.up]
        r = np.zeros_like(rho)
        r[1:] = (rho[1:] - rho[:-1]) + (flow[1:] - q_in[1:])
        return r

    def _raw_report(self, speed, flow) -> Dict[str, np.ndarray]:
        z_s = np.abs((speed - self.sp_med) / self.sp_mad)
        z_f = np.abs((flow - self.fl_med) / self.fl_mad)
        shift = np.maximum(z_s, z_f) / 3.0                     # ~1 at 3 MAD
        phys = np.abs(self._phys_residual(speed, flow)) / self.phys_scale / 3.0
        stale = _staleness(speed, flow) / 3.0
        w_p, w_s, w_t = self.w
        signal = w_p * phys + w_s * shift + w_t * stale
        integrity = np.exp(-signal)                            # (0, 1], 1 = clean
        return {"integrity": integrity, "shift": shift, "phys": phys, "stale": stale}

    # -- scoring --------------------------------------------------------------
    def score(self, speed: np.ndarray, flow: np.ndarray) -> GuardReport:
        rep = self._raw_report(speed, flow)
        integrity = rep["integrity"]
        flag = integrity < self.tau
        return GuardReport(integrity_score=integrity, anomaly_flag=flag,
                           quarantine_mask=flag,
                           components={k: rep[k] for k in ("shift", "phys", "stale")})


def _mad(x: np.ndarray) -> np.ndarray:
    med = np.median(x, axis=0)
    return 1.4826 * np.median(np.abs(x - med), axis=0)


def _staleness(speed: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Length of the current run of unchanged (speed, flow) per node, in steps."""
    T, N = speed.shape
    unchanged = (np.abs(np.diff(speed, axis=0)) < EPS) & (np.abs(np.diff(flow, axis=0)) < EPS)
    age = np.zeros((T, N))
    for j in range(N):
        a = 0
        for t in range(1, T):
            a = a + 1 if unchanged[t - 1, j] else 0
            age[t, j] = a
    return age
