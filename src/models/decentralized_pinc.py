"""Decentralized Physics-Informed Neural Control (Dec-PINC) — primary contribution.

Each corridor node runs an identical local model (shared weights) over its own
observation history, exchanges one round of messages with graph neighbours only
(no central aggregator for the local state), and emits:
  * next-step traffic state (speed, flow),
  * a local control setpoint u in [0, 1] (ramp/flow metering), and
  * a contribution to the coupled grid-demand forecast.

Communication is explicitly droppable (edges masked), so the architecture matches
the contested deployment. Training is multi-stage:
  Stage A  data-driven warm start (prediction loss only),
  Stage B  + composite physics residual loss,
  Stage C  + communication-dropout fine-tuning (random neighbour-message loss).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .physics import composite_physics_loss, upstream_index


class DecPINC(nn.Module):
    def __init__(self, n_features: int, hidden: int = 48, gru_layers: int = 1,
                 physics: bool = True):
        super().__init__()
        self.hidden = hidden
        self.physics = physics
        self.encoder = nn.GRU(n_features, hidden, num_layers=gru_layers,
                              batch_first=True)
        self.msg = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden))
        self.head_state = nn.Linear(hidden, 2)     # next speed, flow (normalized)
        self.head_control = nn.Linear(hidden, 1)   # control setpoint (pre-sigmoid)
        self.head_grid = nn.Linear(hidden, 1)      # per-node grid contribution

    def forward(self, x_seq: torch.Tensor, adjacency: torch.Tensor,
                comm_dropout: float = 0.0) -> Dict[str, torch.Tensor]:
        """x_seq [N, T, F]; adjacency [N, N]. Returns per-node sequences."""
        N, T, F = x_seq.shape
        h, _ = self.encoder(x_seq)                 # [N, T, H] (nodes batched)
        # Row-normalized adjacency with optional communication dropout.
        A = adjacency.clone()
        if comm_dropout > 0.0 and self.training:
            mask = (torch.rand_like(A) >= comm_dropout).float()
            A = A * mask
        A = A * (1.0 - torch.eye(N, device=A.device))   # messages from neighbours only
        deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)
        A_norm = A / deg
        msg = torch.einsum("nm,mth->nth", A_norm, h)    # neighbour-aggregated latents
        z = self.msg(torch.cat([h, msg], dim=-1))       # [N, T, H]
        state = self.head_state(z)                       # [N, T, 2]
        control = torch.sigmoid(self.head_control(z)).squeeze(-1)   # [N, T]
        grid = self.head_grid(z).squeeze(-1).mean(dim=0)            # [T] system demand
        return {"speed": state[..., 0], "flow": state[..., 1],
                "control": control, "grid": grid, "latent": z}


@dataclass
class TrainConfig:
    hidden: int = 48
    lr: float = 3e-3
    epochs_a: int = 60
    epochs_b: int = 60
    epochs_c: int = 40
    lambda_flow: float = 0.1
    lambda_grid: float = 0.1
    lambda_smooth: float = 0.05
    comm_dropout: float = 0.3
    physics: bool = True
    multistage: bool = True


def _denorm(x_norm: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return x_norm * std + mean


def train_dec_pinc(data, cfg: TrainConfig, edges, seed: int = 1337,
                   device: str = "cpu") -> Tuple[DecPINC, dict]:
    """Train Dec-PINC on the corridor training split. Returns (model, history)."""
    torch.manual_seed(seed)
    N, F = data.n_nodes, data.n_features
    model = DecPINC(F, hidden=cfg.hidden, physics=cfg.physics).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    tr0, tr1 = data.splits["train"]
    X = torch.tensor(data.X[tr0:tr1]).permute(1, 0, 2).to(device)   # [N, T, F]
    A = torch.tensor(data.adjacency).to(device)
    up = torch.tensor(upstream_index(data.adjacency, edges), device=device)
    # raw-unit tensors for physics
    sp_mean, sp_std = float(data.scaler_mean[0]), float(data.scaler_std[0])
    fl_mean, fl_std = float(data.scaler_mean[1]), float(data.scaler_std[1])
    grid = torch.tensor(data.grid[tr0:tr1]).to(device)              # [T] normalized
    tgt_speed = X[:, 1:, 0]        # next-step normalized targets
    tgt_flow = X[:, 1:, 1]
    tgt_grid = grid[1:]

    mse = nn.MSELoss()
    history = {"stage": [], "data": [], "physics": []}

    def run_epochs(n, use_physics, comm_dropout):
        for _ in range(n):
            model.train()
            opt.zero_grad()
            out = model(X, A, comm_dropout=comm_dropout)
            pred_s, pred_f = out["speed"][:, :-1], out["flow"][:, :-1]
            pred_g = out["grid"][:-1]
            data_loss = mse(pred_s, tgt_speed) + mse(pred_f, tgt_flow) + mse(pred_g, tgt_grid)
            loss = data_loss
            phys_val = 0.0
            if use_physics:
                prev = {"speed": _denorm(X[:, :-1, 0], sp_mean, sp_std).t(),
                        "flow": _denorm(X[:, :-1, 1], fl_mean, fl_std).t(),
                        "nd": _denorm(grid[:-1], data.grid_mean, data.grid_std),
                        "control": out["control"][:, :-1].t()}
                pred = {"speed": _denorm(pred_s, sp_mean, sp_std).t(),
                        "flow": _denorm(pred_f, fl_mean, fl_std).t(),
                        "nd": _denorm(pred_g, data.grid_mean, data.grid_std),
                        "control": out["control"][:, 1:].t()}
                phys, comps = composite_physics_loss(
                    pred, prev, up,
                    {"flow": cfg.lambda_flow, "grid": cfg.lambda_grid,
                     "smooth": cfg.lambda_smooth}, alpha=1.0)
                # normalize physics scale so it does not dominate
                loss = data_loss + phys / (phys.detach() + 1.0) * cfg.lambda_flow
                phys_val = float(phys.detach())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            history["data"].append(float(data_loss.detach()))
            history["physics"].append(phys_val)

    if cfg.multistage:
        run_epochs(cfg.epochs_a, False, 0.0)             # Stage A
        if cfg.physics:
            run_epochs(cfg.epochs_b, True, 0.0)          # Stage B
        run_epochs(cfg.epochs_c, cfg.physics, cfg.comm_dropout)   # Stage C
    else:
        run_epochs(cfg.epochs_a + cfg.epochs_b + cfg.epochs_c, cfg.physics, cfg.comm_dropout)
    return model, history


@torch.no_grad()
def predict(model: DecPINC, data, obs: np.ndarray, adjacency: np.ndarray,
            device: str = "cpu") -> Dict[str, np.ndarray]:
    """Forecast next-step speed/flow/control/grid over an observation tensor
    ``obs`` [T, N, F] (already contested/clean). Returns raw-unit arrays."""
    model.eval()
    X = torch.tensor(obs, dtype=torch.float32).permute(1, 0, 2).to(device)
    A = torch.tensor(adjacency, dtype=torch.float32).to(device)
    out = model(X, A, comm_dropout=0.0)
    sp = out["speed"].t().cpu().numpy() * data.scaler_std[0] + data.scaler_mean[0]
    fl = out["flow"].t().cpu().numpy() * data.scaler_std[1] + data.scaler_mean[1]
    gd = out["grid"].cpu().numpy() * data.grid_std + data.grid_mean
    ctrl = out["control"].t().cpu().numpy()
    return {"speed": sp, "flow": fl, "grid": gd, "control": ctrl}
