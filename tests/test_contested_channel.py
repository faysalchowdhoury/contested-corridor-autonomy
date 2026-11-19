"""Verify the contested channel masks and that it is evaluation-only in spirit
(deterministic, parameterized, and leaves exogenous columns intact)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.contested_channel import (ContestedChannel, REGIMES, TELEMETRY_COLS)


@pytest.fixture
def obs():
    rng = np.random.default_rng(0)
    return rng.normal(0, 1, (200, 4, 5)).astype(np.float32)


def test_bernoulli_drop_rate(obs):
    ch = ContestedChannel(seed=1)
    out = ch.packet_drop(obs, 0.3, bursty=False)
    loss = 1 - out.keep_mask.mean()
    assert abs(loss - 0.3) < 0.06                 # empirical loss ~ target


def test_burst_drops_are_clustered(obs):
    ch = ContestedChannel(seed=1)
    out = ch.packet_drop(obs, 0.3, bursty=True)
    # mean run length of drops should exceed the Bernoulli expectation (~1/(1-p))
    drops = (out.keep_mask[:, 0] == 0).astype(int)
    runs, cur = [], 0
    for d in drops:
        if d:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    assert (np.mean(runs) if runs else 0) > 1.5


def test_staleness_holds_last_value(obs):
    ch = ContestedChannel(seed=2)
    out = ch.packet_drop(obs, 0.5, bursty=False)
    # where dropped, telemetry columns equal the last fresh value (age > 0)
    for t in range(1, obs.shape[0]):
        for j in range(obs.shape[1]):
            if out.keep_mask[t, j] == 0:
                assert out.age[t, j] >= 1
                for c in TELEMETRY_COLS:
                    assert out.obs[t, j, c] == out.obs[t - 1, j, c]


def test_exogenous_columns_untouched(obs):
    ch = ContestedChannel(seed=3)
    for regime in ("drop60", "high_latency", "spoofing"):
        out = REGIMES[regime](ch, obs)
        # weather columns (2,3,4) are not corrupted
        assert np.allclose(out.obs[..., 2:], obs[..., 2:])


def test_latency_bounded(obs):
    ch = ContestedChannel(seed=4)
    out = ch.latency(obs, max_delay=12)
    assert out.age.max() <= 12
    assert out.age.min() >= 0


def test_spoof_marks_attacks(obs):
    ch = ContestedChannel(seed=5)
    out = ch.spoof(obs, frac_nodes=0.5, kind="step")
    assert out.attack_mask.sum() > 0
    # attacked entries differ from clean on telemetry columns
    diff = np.abs(out.obs[..., 0] - obs[..., 0])
    assert diff[out.attack_mask.astype(bool)].mean() > 0


def test_determinism(obs):
    a = ContestedChannel(seed=9).packet_drop(obs, 0.4, bursty=True).keep_mask
    b = ContestedChannel(seed=9).packet_drop(obs, 0.4, bursty=True).keep_mask
    assert np.array_equal(a, b)
