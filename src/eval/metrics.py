"""Evaluation metrics for the benchmark tables.

Forecast accuracy (RMSE/MAE/MAPE on next-step speed), physics-residual violation
rate, a control-stability cost (forecast trajectory roughness, comparable across all
methods), and anomaly-detection metrics (AUROC, F1, detection delay).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ..models.physics import physics_violation_rate, upstream_index


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def mape(a, b, eps=1.0):
    return float(np.mean(np.abs(a - b) / np.clip(np.abs(b), eps, None)) * 100)


def control_stability_cost(speed_pred: np.ndarray) -> float:
    """Normalized mean-squared second difference (trajectory 'jerk') of the forecast.

    Comparable across all methods (controller or not): an unstable controller or an
    oscillatory forecast inflates it. Normalized by the forecast variance.
    """
    if speed_pred.shape[0] < 3:
        return 0.0
    accel = speed_pred[2:] - 2 * speed_pred[1:-1] + speed_pred[:-2]
    var = np.var(speed_pred) + 1e-6
    return float(np.mean(accel ** 2) / var)


def forecast_metrics(pred: Dict[str, np.ndarray], data, edges,
                     test_slice, phys_threshold: float = None) -> Dict[str, float]:
    """Next-step speed metrics over the test window. pred arrays are [T, N] raw
    forecasts aligned so pred[t] forecasts t+1."""
    t0, t1 = test_slice
    if phys_threshold is None:
        phys_threshold = getattr(data, 'phys_threshold', 0.5)
    sp_pred = pred["speed"][t0:t1 - 1]       # forecast for t+1
    sp_true = data.speed[t0 + 1:t1]
    up = upstream_index(data.adjacency, edges)
    pvr = physics_violation_rate(pred["speed"][t0:t1], pred["flow"][t0:t1], up,
                                 phys_threshold)
    return {
        "RMSE": rmse(sp_pred, sp_true),
        "MAE": mae(sp_pred, sp_true),
        "MAPE": mape(sp_pred, sp_true),
        "PhysViolRate": pvr,
        "CtrlStabCost": control_stability_cost(pred["speed"][t0:t1]),
    }


# --- anomaly detection -------------------------------------------------------
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via rank statistic (higher score => more anomalous)."""
    labels = labels.astype(bool).ravel()
    scores = scores.ravel()
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    r_pos = ranks[labels].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def f1_score(flag: np.ndarray, labels: np.ndarray) -> float:
    flag = flag.astype(bool).ravel(); labels = labels.astype(bool).ravel()
    tp = np.sum(flag & labels); fp = np.sum(flag & ~labels); fn = np.sum(~flag & labels)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def detection_delay(flag: np.ndarray, attack_mask: np.ndarray) -> float:
    """Mean steps from attack onset to first flag, per attacked node (NaN-safe)."""
    T, N = attack_mask.shape
    delays = []
    for n in range(N):
        col = attack_mask[:, n]
        if not col.any():
            continue
        onset = np.argmax(col)
        fired = np.where(flag[onset:, n])[0]
        delays.append(fired[0] if len(fired) else (T - onset))
    return float(np.mean(delays)) if delays else float("nan")


def anomaly_metrics(integrity: np.ndarray, flag: np.ndarray,
                    attack_mask: np.ndarray, test_slice) -> Dict[str, float]:
    t0, t1 = test_slice
    anomaly_score = 1.0 - integrity[t0:t1]
    labels = attack_mask[t0:t1]
    return {
        "AUROC": auroc(anomaly_score, labels),
        "F1": f1_score(flag[t0:t1], labels),
        "DetectionDelaySteps": detection_delay(flag[t0:t1], labels),
    }
