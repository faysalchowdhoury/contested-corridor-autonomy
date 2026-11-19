"""Physics residuals for the composite PINC loss and for evaluation.

Three conservation principles ground the model:

1. **Macroscopic traffic flow conservation** (discrete continuity). With the
   fundamental relation q = rho * v, cell density rho = q / v. For a corridor cell
   fed by its upstream neighbour, mass balance is
       rho[t+1] - rho[t] + (dt/dx) * (q_out - q_in) ~ 0.
2. **Coupled mobility->energy load consistency.** Aggregate corridor flow drives an
   EV/charging load proxy; the national-demand change should track the corridor
   activity change up to a scale alpha:
       (nd[t+1] - nd[t]) - alpha * (Q[t+1] - Q[t]) ~ 0,  Q = sum_n q_n.
3. **Control stability / actuator bounds.** Local control setpoints u in [0, 1]
   must be smooth in time and respect physical bounds.

Functions operate on torch tensors (autograd-friendly) for the loss, with numpy
helpers for metrics. All are unit-tested in tests/test_physics.py.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

EPS = 1e-3


def densities(speed: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """rho = q / v (guarded). Shapes broadcast; speed in mph, flow in veh/interval."""
    return flow / torch.clamp(speed, min=EPS)


def upstream_index(adjacency: np.ndarray, edges: Sequence[Sequence[int]]) -> np.ndarray:
    """Map each node to an upstream neighbour from the directed edge list
    (edge [a, b] means a -> b, so a is upstream of b). Nodes with no upstream map
    to themselves (zero contribution)."""
    n = adjacency.shape[0]
    up = np.arange(n)
    for a, b in edges:
        up[b] = a
    return up


def traffic_conservation_residual(speed: torch.Tensor, flow: torch.Tensor,
                                  speed_next: torch.Tensor, flow_next: torch.Tensor,
                                  up: torch.Tensor, dt_over_dx: float = 1.0) -> torch.Tensor:
    """Discrete LWR continuity residual per node. Shapes [B, N] -> [B, N]."""
    rho = densities(speed, flow)
    rho_next = densities(speed_next, flow_next)
    q_out = flow_next
    q_in = flow_next.index_select(-1, up)          # upstream flow feeds the cell
    return (rho_next - rho) + dt_over_dx * (q_out - q_in)


def load_balance_residual(nd: torch.Tensor, nd_next: torch.Tensor,
                          flow: torch.Tensor, flow_next: torch.Tensor,
                          alpha: float) -> torch.Tensor:
    """Coupled mobility->energy consistency residual. nd shape [B], flow [B, N]."""
    dQ = flow_next.sum(dim=-1) - flow.sum(dim=-1)
    dND = nd_next - nd
    return dND - alpha * dQ


def control_stability_loss(u: torch.Tensor, u_prev: torch.Tensor) -> torch.Tensor:
    """Smoothness + actuator-bound hinge on control setpoints u in [0, 1]."""
    smooth = torch.mean((u - u_prev) ** 2)
    bound = torch.mean(torch.relu(u - 1.0) ** 2 + torch.relu(-u) ** 2)
    return smooth + bound


def composite_physics_loss(pred: dict, prev: dict, up: torch.Tensor,
                           lambdas: dict, alpha: float = 1.0,
                           dt_over_dx: float = 1.0):
    """Assemble weighted physics terms. Returns (total, components dict)."""
    cons = traffic_conservation_residual(
        prev["speed"], prev["flow"], pred["speed"], pred["flow"], up, dt_over_dx)
    l_cons = torch.mean(cons ** 2)
    l_grid = torch.mean(load_balance_residual(
        prev["nd"], pred["nd"], prev["flow"], pred["flow"], alpha) ** 2)
    l_smooth = control_stability_loss(pred["control"], prev["control"])
    total = (lambdas.get("flow", 0.0) * l_cons
             + lambdas.get("grid", 0.0) * l_grid
             + lambdas.get("smooth", 0.0) * l_smooth)
    return total, {"conservation": l_cons, "load_balance": l_grid,
                   "control_stability": l_smooth}


# --- numpy evaluation helpers ------------------------------------------------
def physics_violation_rate(speed: np.ndarray, flow: np.ndarray,
                           up: np.ndarray, threshold: float,
                           dt_over_dx: float = 1.0) -> float:
    """Fraction of (time, node) continuity residuals exceeding ``threshold``.

    Residual normalized by a characteristic density scale so the rate is unit-free.
    speed/flow are [T, N] sequences; residual uses consecutive timesteps.
    """
    s = np.clip(speed, EPS, None)
    rho = flow / s
    q_in = flow[:, up]
    r = (rho[1:] - rho[:-1]) + dt_over_dx * (flow[1:] - q_in[1:])
    scale = np.median(np.abs(rho)) + EPS
    return float(np.mean(np.abs(r) / scale > threshold))
