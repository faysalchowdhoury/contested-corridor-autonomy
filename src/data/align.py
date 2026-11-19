"""Align traffic + grid + weather onto one spatio-temporal corridor tensor.

Produces a node-time feature tensor on a common 30-minute cadence, a corridor
adjacency matrix, chronological train/val/test splits with **zero look-ahead**,
and feature scalers fit **only on the training split**. Everything is cached to
``data/processed/corridor.npz`` with a sidecar meta JSON.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .ingest import Manifest, fetch_grid, fetch_traffic_site, fetch_weather

# Node feature order (per node, per timestep).
NODE_FEATURES = ["speed_mph", "flow_veh", "temp_c", "wind_kmh", "precip_mm"]


@dataclass
class CorridorData:
    X: np.ndarray                 # [T, N, F] normalized node features
    grid: np.ndarray              # [T] normalized national demand
    speed: np.ndarray             # [T, N] raw speed (mph)
    flow: np.ndarray              # [T, N] raw flow (veh / 30 min)
    nd_mw: np.ndarray             # [T] raw national demand (MW)
    adjacency: np.ndarray         # [N, N]
    timestamps: np.ndarray        # [T] datetime64
    splits: Dict[str, Tuple[int, int]]
    scaler_mean: np.ndarray       # [F] (+ grid at end)
    scaler_std: np.ndarray
    grid_mean: float
    grid_std: float
    sites: List[dict]
    phys_threshold: float = 1.0   # calibrated so clean train ~ 10% violation

    @property
    def n_nodes(self) -> int:
        return self.X.shape[1]

    @property
    def n_features(self) -> int:
        return self.X.shape[2]


def _common_index(start: str, end: str, cadence_min: int) -> pd.DatetimeIndex:
    e = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(minutes=cadence_min)
    return pd.date_range(pd.to_datetime(start), e, freq=f"{cadence_min}min")


def _resample_traffic(df: pd.DataFrame, idx: pd.DatetimeIndex,
                      cadence_min: int) -> pd.DataFrame:
    s = df.set_index("timestamp").sort_index()
    speed = s["speed_mph"].resample(f"{cadence_min}min").mean()
    flow = s["flow_veh"].resample(f"{cadence_min}min").sum(min_count=1)
    out = pd.DataFrame({"speed_mph": speed, "flow_veh": flow}).reindex(idx)
    return out


def _resample_weather(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    s = df.set_index("timestamp").sort_index()
    return s.reindex(idx.union(s.index)).interpolate("time").reindex(idx)


def build_corridor(cfg: dict, cache_dir: Path, manifest: Optional[Manifest] = None,
                   processed_dir: Optional[Path] = None) -> CorridorData:
    window = cfg["window"]
    start, end = window["start"], window["end"]
    cadence = int(cfg.get("cadence_min", 30))
    sites = cfg["sites"]
    idx = _common_index(start, end, cadence)
    T, N = len(idx), len(sites)

    speed = np.full((T, N), np.nan)
    flow = np.full((T, N), np.nan)
    weather = {k: np.full((T, N), np.nan) for k in ("temp_c", "wind_kmh", "precip_mm")}

    for j, site in enumerate(sites):
        tdf = fetch_traffic_site(site["id"], start, end, cache_dir, manifest)
        tr = _resample_traffic(tdf, idx, cadence)
        speed[:, j] = tr["speed_mph"].to_numpy()
        flow[:, j] = tr["flow_veh"].to_numpy()
        wdf = fetch_weather(site["lat"], site["lon"], start, end, cache_dir, manifest)
        wr = _resample_weather(wdf, idx)
        for k in weather:
            weather[k][:, j] = wr[k].to_numpy()

    gdf = fetch_grid(start, end, cache_dir,
                     resource_id=cfg["grid"]["resource_id"], manifest=manifest)
    nd = gdf.set_index("timestamp")["nd_mw"].reindex(idx).to_numpy()

    # Missingness: forward/back fill in time per node, then fill residual with column mean.
    speed = _fill(speed)
    flow = _fill(flow)
    for k in weather:
        weather[k] = _fill(weather[k])
    nd = _fill(nd.reshape(-1, 1)).ravel()

    # Assemble raw feature tensor [T, N, F].
    raw = np.stack([speed, flow, weather["temp_c"], weather["wind_kmh"],
                    weather["precip_mm"]], axis=-1)

    # Chronological splits (zero look-ahead).
    tr_end = int(T * cfg["splits"]["train"])
    val_end = tr_end + int(T * cfg["splits"]["val"])
    splits = {"train": (0, tr_end), "val": (tr_end, val_end), "test": (val_end, T)}

    # Fit scalers on TRAIN ONLY.
    tr_slice = raw[:tr_end]
    mean = tr_slice.reshape(-1, raw.shape[-1]).mean(0)
    std = tr_slice.reshape(-1, raw.shape[-1]).std(0) + 1e-6
    X = (raw - mean) / std
    g_mean = float(nd[:tr_end].mean())
    g_std = float(nd[:tr_end].std() + 1e-6)
    grid = (nd - g_mean) / g_std

    # Adjacency.
    A = np.eye(N)
    for a, b in cfg.get("edges", []):
        A[a, b] = A[b, a] = 1.0

    # Calibrate the physics-violation threshold on the CLEAN TRAIN split so that
    # clean, real data scores ~10% (the metric then measures *excess* violation).
    from ..models.physics import physics_violation_rate, upstream_index
    up = upstream_index(A, cfg.get("edges", []))
    s = np.clip(speed[:tr_end], 1e-3, None)
    rho = flow[:tr_end] / s
    q_in = flow[:tr_end][:, up]
    resid = np.abs((rho[1:] - rho[:-1]) + (flow[:tr_end][1:] - q_in[1:]))
    scale = np.median(np.abs(rho)) + 1e-3
    phys_threshold = float(np.quantile(resid / scale, 0.90))

    data = CorridorData(
        X=X.astype(np.float32), grid=grid.astype(np.float32),
        speed=speed.astype(np.float32), flow=flow.astype(np.float32),
        nd_mw=nd.astype(np.float32), adjacency=A.astype(np.float32),
        timestamps=idx.values, splits=splits,
        scaler_mean=mean.astype(np.float32), scaler_std=std.astype(np.float32),
        grid_mean=g_mean, grid_std=g_std, sites=sites, phys_threshold=phys_threshold)

    if processed_dir is not None:
        save_corridor(data, processed_dir)
    return data


def _fill(a: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(a)
    df = df.ffill().bfill()
    df = df.fillna(df.mean())
    return df.fillna(0.0).to_numpy()


def save_corridor(data: CorridorData, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        processed_dir / "corridor.npz",
        X=data.X, grid=data.grid, speed=data.speed, flow=data.flow, nd_mw=data.nd_mw,
        adjacency=data.adjacency, timestamps=data.timestamps.astype("datetime64[s]").astype("int64"),
        scaler_mean=data.scaler_mean, scaler_std=data.scaler_std)
    meta = {"splits": data.splits, "grid_mean": data.grid_mean, "grid_std": data.grid_std, "phys_threshold": data.phys_threshold,
            "features": NODE_FEATURES, "sites": data.sites}
    (processed_dir / "corridor_meta.json").write_text(json.dumps(meta, indent=2, default=str))


def load_corridor(processed_dir: Path) -> CorridorData:
    npz = np.load(processed_dir / "corridor.npz", allow_pickle=False)
    meta = json.loads((processed_dir / "corridor_meta.json").read_text())
    splits = {k: tuple(v) for k, v in meta["splits"].items()}
    return CorridorData(
        X=npz["X"], grid=npz["grid"], speed=npz["speed"], flow=npz["flow"],
        nd_mw=npz["nd_mw"], adjacency=npz["adjacency"],
        timestamps=npz["timestamps"].astype("datetime64[s]"), splits=splits,
        scaler_mean=npz["scaler_mean"], scaler_std=npz["scaler_std"],
        grid_mean=meta["grid_mean"], grid_std=meta["grid_std"], sites=meta["sites"],
        phys_threshold=meta.get("phys_threshold", 1.0))
