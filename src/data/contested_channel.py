"""Contested communication & fault-injection channel (EVALUATION/STRESS-TEST ONLY).

Applied only to the evaluation telemetry stream so training never sees fabricated
corruption as ground truth. Each regime is mathematically explicit and seeded.

Regimes
-------
* Packet drop / blackout : Bernoulli(p) and bursty Gilbert-Elliott (Markov) loss.
  Dropped telemetry is held at its last value (stale), with an age counter.
* Latency / staleness    : bounded and heavy-tailed delays, Delta t in [1, 12] steps.
* Adversarial corruption : replay, step/ramp spoofing, localized sensor bias.

Outputs a :class:`ContestedOutput` carrying the corrupted observation, a keep mask,
an attack mask and the per-entry staleness age, so the telemetry guard can be scored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# Which feature columns are communicated telemetry subject to corruption
# (speed_mph=0, flow_veh=1). Weather/exogenous columns are left intact.
TELEMETRY_COLS = (0, 1)


@dataclass
class ContestedOutput:
    obs: np.ndarray            # [T, N, F] corrupted observations fed to a model
    keep_mask: np.ndarray      # [T, N] 1 = fresh packet received, 0 = dropped
    attack_mask: np.ndarray    # [T, N] 1 = adversarially corrupted
    age: np.ndarray            # [T, N] staleness in steps since last fresh packet
    regime: str = ""


def _bernoulli_keep(T: int, N: int, p_loss: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.random((T, N)) >= p_loss).astype(np.int8)


def _gilbert_elliott_keep(T: int, N: int, p_loss: float, rng: np.random.Generator,
                          mean_burst: float = 4.0) -> np.ndarray:
    """Two-state (good/bad) Markov loss. In 'bad' the packet is dropped.

    Stationary bad probability == p_loss; mean bad-run length == mean_burst sets the
    bad->good rate r = 1/mean_burst, and the good->bad rate is solved from the
    stationary distribution: pi_bad = g/(g+r) -> g = r*p/(1-p).
    """
    r = 1.0 / max(mean_burst, 1.0)
    g = r * p_loss / max(1.0 - p_loss, 1e-6)
    keep = np.ones((T, N), dtype=np.int8)
    for j in range(N):
        bad = rng.random() < p_loss
        for t in range(T):
            keep[t, j] = 0 if bad else 1
            if bad:
                bad = rng.random() >= r      # stay bad unless recovery
            else:
                bad = rng.random() < g       # may enter bad
    return keep


def _apply_staleness(obs: np.ndarray, keep: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hold-last-value where a packet is dropped; return (stale_obs, age)."""
    T, N, F = obs.shape
    out = obs.copy()
    age = np.zeros((T, N), dtype=np.int32)
    for j in range(N):
        last = obs[0, j].copy()
        a = 0
        for t in range(T):
            if keep[t, j]:
                last = obs[t, j].copy()
                a = 0
            else:
                a += 1
                for c in TELEMETRY_COLS:
                    out[t, j, c] = last[c]
            age[t, j] = a
    return out, age


class ContestedChannel:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    # -- packet loss ----------------------------------------------------------
    def packet_drop(self, obs: np.ndarray, p_loss: float, bursty: bool = False,
                    regime: str = "") -> ContestedOutput:
        T, N, _ = obs.shape
        keep = (_gilbert_elliott_keep(T, N, p_loss, self.rng) if bursty
                else _bernoulli_keep(T, N, p_loss, self.rng))
        stale, age = _apply_staleness(obs, keep)
        return ContestedOutput(stale, keep, np.zeros((T, N), np.int8), age,
                               regime or f"drop{int(p_loss*100)}{'_burst' if bursty else ''}")

    # -- latency / staleness --------------------------------------------------
    def latency(self, obs: np.ndarray, max_delay: int = 12, heavy_tail: bool = True,
                regime: str = "high_latency") -> ContestedOutput:
        T, N, F = obs.shape
        out = obs.copy()
        age = np.zeros((T, N), np.int32)
        for j in range(N):
            for t in range(T):
                if heavy_tail:  # Zipf-like heavy tail clipped to [1, max_delay]
                    d = min(max_delay, int(self.rng.zipf(2.0)))
                else:
                    d = int(self.rng.integers(1, max_delay + 1))
                src = max(0, t - d)
                for c in TELEMETRY_COLS:
                    out[t, j, c] = obs[src, j, c]
                age[t, j] = t - src
        keep = (age == 0).astype(np.int8)
        return ContestedOutput(out, keep, np.zeros((T, N), np.int8), age, regime)

    # -- adversarial corruption ----------------------------------------------
    def spoof(self, obs: np.ndarray, frac_nodes: float = 0.3, kind: str = "mixed",
              regime: str = "spoofing") -> ContestedOutput:
        T, N, F = obs.shape
        out = obs.copy()
        attack = np.zeros((T, N), np.int8)
        n_attack = max(1, int(round(frac_nodes * N)))
        targets = self.rng.choice(N, size=n_attack, replace=False)
        for j in targets:
            k = kind if kind != "mixed" else self.rng.choice(
                ["replay", "step", "ramp", "bias"])
            t0 = int(self.rng.integers(T // 4, T // 2))
            t1 = int(self.rng.integers(t0 + T // 8, T))
            seg = slice(t0, t1)
            for c in TELEMETRY_COLS:
                col = out[:, j, c]
                scale = np.std(obs[:, j, c]) + 1e-6
                if k == "replay":
                    past = max(0, t0 - (t1 - t0))
                    col[seg] = obs[past:past + (t1 - t0), j, c]
                elif k == "step":
                    col[seg] = col[seg] + 3.0 * scale
                elif k == "ramp":
                    col[seg] = col[seg] + np.linspace(0, 4.0 * scale, t1 - t0)
                elif k == "bias":
                    col[seg] = col[seg] + self.rng.normal(2.0 * scale, 0.2 * scale, t1 - t0)
            attack[seg, j] = 1
        return ContestedOutput(out, np.ones((T, N), np.int8), attack,
                               np.zeros((T, N), np.int32), regime)


REGIMES = {
    "clean":         lambda ch, obs: ContestedOutput(obs.copy(), np.ones(obs.shape[:2], np.int8),
                                                     np.zeros(obs.shape[:2], np.int8),
                                                     np.zeros(obs.shape[:2], np.int32), "clean"),
    "drop30":        lambda ch, obs: ch.packet_drop(obs, 0.30, bursty=True),
    "drop60":        lambda ch, obs: ch.packet_drop(obs, 0.60, bursty=True),
    "high_latency":  lambda ch, obs: ch.latency(obs, max_delay=12, heavy_tail=True),
    "spoofing":      lambda ch, obs: ch.spoof(obs, frac_nodes=0.3, kind="mixed"),
}
