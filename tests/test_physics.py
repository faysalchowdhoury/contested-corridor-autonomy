"""Verify the physics-residual math."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.physics import (control_stability_loss, densities,
                                load_balance_residual, physics_violation_rate,
                                traffic_conservation_residual, upstream_index)


def test_density_relation():
    speed = torch.tensor([[20.0, 40.0]])
    flow = torch.tensor([[200.0, 200.0]])
    rho = densities(speed, flow)           # q / v
    assert torch.allclose(rho, torch.tensor([[10.0, 5.0]]))


def test_conservation_zero_on_steady_state():
    # constant density and uniform flow => continuity residual must be ~0
    speed = torch.full((1, 4), 30.0)
    flow = torch.full((1, 4), 300.0)
    up = torch.tensor([0, 0, 1, 2])         # chain upstream
    r = traffic_conservation_residual(speed, flow, speed, flow, up, dt_over_dx=1.0)
    # rho unchanged and q_out == q_in everywhere -> zero
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-5)


def test_conservation_detects_imbalance():
    speed = torch.full((1, 3), 30.0)
    flow = torch.tensor([[300.0, 300.0, 300.0]])
    flow_next = torch.tensor([[300.0, 600.0, 300.0]])   # node 1 gains flow from nowhere
    up = torch.tensor([0, 0, 1])
    r = traffic_conservation_residual(speed, flow, speed, flow_next, up)
    assert r.abs().sum() > 0.1


def test_load_balance_residual_sign():
    nd = torch.tensor([1000.0]); nd_next = torch.tensor([1100.0])
    flow = torch.tensor([[100.0, 100.0]]); flow_next = torch.tensor([[150.0, 150.0]])
    r = load_balance_residual(nd, nd_next, flow, flow_next, alpha=1.0)
    # dND=100, alpha*dQ = 100 -> residual 0
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-5)


def test_control_bounds_penalized():
    u_ok = torch.tensor([0.2, 0.8]); u_bad = torch.tensor([1.5, -0.3])
    prev = torch.tensor([0.2, 0.8])
    assert control_stability_loss(u_bad, prev) > control_stability_loss(u_ok, prev)


def test_upstream_index():
    A = np.zeros((3, 3))
    up = upstream_index(A, [[0, 1], [1, 2]])
    assert list(up) == [0, 0, 1]            # 0->1->2


def test_violation_rate_bounds():
    speed = np.full((10, 3), 30.0); flow = np.full((10, 3), 300.0)
    up = np.array([0, 0, 1])
    assert physics_violation_rate(speed, flow, up, threshold=0.5) == 0.0
    flow[5, 1] = 5000.0                      # inject a violation
    assert physics_violation_rate(speed, flow, up, threshold=0.5) > 0.0
