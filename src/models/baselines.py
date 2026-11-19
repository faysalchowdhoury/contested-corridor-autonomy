"""Competitive, fairly-tuned baselines evaluated under identical contested masks.

All expose the same interface:
    fit(data, edges)
    predict(obs[T,N,F], adjacency) -> {"speed":[T,N],"flow":[T,N],"grid":[T],"control":[T,N]}
returning raw-unit next-step forecasts (prediction at index t is the forecast for t+1).

Baselines:
  Persistence (LOCF)            last observation carried forward
  Ridge                         pooled per-node ridge regression
  GBM (LightGBM)                pooled per-node gradient-boosted trees
  CentralizedGNN                global-pooling neural net, NO physics loss
  DecentralizedNN (ablation)    identical architecture to Dec-PINC, physics off
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .decentralized_pinc import DecPINC, TrainConfig, predict as pinc_predict, train_dec_pinc


# --------------------------------------------------------------------------- #
def _neighbor_mean(X: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    """Mean of neighbours' speed/flow per node -> [T, N, 2]."""
    A = adjacency * (1 - np.eye(adjacency.shape[0]))
    deg = A.sum(1, keepdims=True).clip(min=1)
    An = A / deg
    sf = X[..., :2]                               # [T, N, 2]
    return np.einsum("nm,tmc->tnc", An, sf)


def _tabular_xy(data, obs: np.ndarray, adjacency: np.ndarray):
    """Build pooled (features, speed_target, flow_target) for tabular baselines."""
    nbr = _neighbor_mean(obs, adjacency)
    feats = np.concatenate([obs, nbr], axis=-1)   # [T, N, F+2]
    T, N, _ = feats.shape
    X2 = feats[:-1].reshape(-1, feats.shape[-1])  # predict next step
    return X2, T, N, feats


class Persistence:
    name = "Persistence (LOCF)"

    def fit(self, data, edges):
        self.data = data
        return self

    def predict(self, obs, adjacency):
        sp = obs[..., 0] * self.data.scaler_std[0] + self.data.scaler_mean[0]
        fl = obs[..., 1] * self.data.scaler_std[1] + self.data.scaler_mean[1]
        T, N = sp.shape
        grid = np.full(T, self.data.nd_mw[self.data.splits["train"][1] - 1])
        return {"speed": sp, "flow": fl, "grid": grid, "control": np.zeros((T, N))}


class Ridge:
    name = "Ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, data, edges):
        from sklearn.linear_model import Ridge as SKRidge
        self.data = data
        tr0, tr1 = data.splits["train"]
        X2, T, N, feats = _tabular_xy(data, data.X, data.adjacency)
        # training rows only
        rows = np.arange((tr1 - 1) * N)
        Xtr = feats[:tr1 - 1].reshape(-1, feats.shape[-1])
        y_s = data.speed[1:tr1].reshape(-1)
        y_f = data.flow[1:tr1].reshape(-1)
        self.m_s = SKRidge(self.alpha).fit(Xtr, y_s)
        self.m_f = SKRidge(self.alpha).fit(Xtr, y_f)
        self._fit_grid(data)
        return self

    def _fit_grid(self, data):
        from sklearn.linear_model import Ridge as SKRidge
        tr1 = data.splits["train"][1]
        g = _grid_features(data.X[:tr1])
        self.m_g = SKRidge(self.alpha).fit(g[:-1], data.nd_mw[1:tr1])

    def predict(self, obs, adjacency):
        _, T, N, feats = _tabular_xy(self.data, obs, adjacency)
        flat = feats.reshape(-1, feats.shape[-1])
        sp = self.m_s.predict(flat).reshape(T, N)
        fl = self.m_f.predict(flat).reshape(T, N)
        grid = self.m_g.predict(_grid_features(obs))
        return {"speed": sp, "flow": fl, "grid": grid, "control": np.zeros((T, N))}


class GBM:
    name = "GBM (LightGBM)"

    def fit(self, data, edges):
        import lightgbm as lgb
        self.data = data
        tr1 = data.splits["train"][1]
        _, T, N, feats = _tabular_xy(data, data.X, data.adjacency)
        Xtr = feats[:tr1 - 1].reshape(-1, feats.shape[-1])
        params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
                      min_child_samples=40, verbose=-1)
        self.m_s = lgb.LGBMRegressor(**params).fit(Xtr, data.speed[1:tr1].reshape(-1))
        self.m_f = lgb.LGBMRegressor(**params).fit(Xtr, data.flow[1:tr1].reshape(-1))
        g = _grid_features(data.X[:tr1])
        self.m_g = lgb.LGBMRegressor(**params).fit(g[:-1], data.nd_mw[1:tr1])
        return self

    def predict(self, obs, adjacency):
        _, T, N, feats = _tabular_xy(self.data, obs, adjacency)
        flat = feats.reshape(-1, feats.shape[-1])
        sp = self.m_s.predict(flat).reshape(T, N)
        fl = self.m_f.predict(flat).reshape(T, N)
        grid = self.m_g.predict(_grid_features(obs))
        return {"speed": sp, "flow": fl, "grid": grid, "control": np.zeros((T, N))}


def _grid_features(X: np.ndarray) -> np.ndarray:
    """System-level features for the grid head: mean speed/flow/weather over nodes."""
    return X.mean(axis=1)                          # [T, F]


# --------------------------------------------------------------------------- #
class CentralizedGNN:
    """Global-pooling neural net (no physics). A central operator ingests all node
    features jointly each step; contrasts with Dec-PINC's local message passing."""
    name = "Centralized NN (no physics)"

    def __init__(self, hidden: int = 48):
        self.hidden = hidden

    def fit(self, data, edges, epochs: int = 120, lr: float = 3e-3, seed: int = 1337):
        torch.manual_seed(seed)
        self.data = data
        N, F = data.n_nodes, data.n_features
        self.net = _CentralNet(N, F, self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        tr0, tr1 = data.splits["train"]
        X = torch.tensor(data.X[tr0:tr1])          # [T, N, F]
        grid = torch.tensor(data.grid[tr0:tr1])
        tgt_s, tgt_f, tgt_g = X[1:, :, 0], X[1:, :, 1], grid[1:]
        mse = nn.MSELoss()
        for _ in range(epochs):
            self.net.train(); opt.zero_grad()
            s, f, g = self.net(X)
            loss = mse(s[:-1], tgt_s) + mse(f[:-1], tgt_f) + mse(g[:-1], tgt_g)
            loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def predict(self, obs, adjacency):
        self.net.eval()
        X = torch.tensor(obs, dtype=torch.float32)
        s, f, g = self.net(X)
        d = self.data
        return {"speed": s.numpy() * d.scaler_std[0] + d.scaler_mean[0],
                "flow": f.numpy() * d.scaler_std[1] + d.scaler_mean[1],
                "grid": g.numpy() * d.grid_std + d.grid_mean,
                "control": np.zeros(obs.shape[:2])}


class _CentralNet(nn.Module):
    def __init__(self, n_nodes, n_features, hidden):
        super().__init__()
        self.n, self.f = n_nodes, n_features
        self.gru = nn.GRU(n_nodes * n_features, hidden, batch_first=True)
        self.head_s = nn.Linear(hidden, n_nodes)
        self.head_f = nn.Linear(hidden, n_nodes)
        self.head_g = nn.Linear(hidden, 1)

    def forward(self, X):                           # X [T, N, F]
        T = X.shape[0]
        flat = X.reshape(1, T, self.n * self.f)     # global (central) view
        h, _ = self.gru(flat)
        h = h.squeeze(0)
        return self.head_s(h), self.head_f(h), self.head_g(h).squeeze(-1)


@dataclass
class DecentralizedNN:
    """Ablation: identical architecture to Dec-PINC with the physics loss off."""
    name: str = "Decentralized NN (no physics)"

    def fit(self, data, edges, seed: int = 1337):
        cfg = TrainConfig(physics=False, multistage=True)
        self.model, _ = train_dec_pinc(data, cfg, edges, seed=seed)
        self.data = data
        return self

    def predict(self, obs, adjacency):
        return pinc_predict(self.model, self.data, obs, adjacency)


def build_baselines() -> List:
    return [Persistence(), Ridge(), GBM(), CentralizedGNN(), DecentralizedNN()]
