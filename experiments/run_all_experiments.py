"""Reproducible end-to-end benchmark: real data -> Tables 1-3 + Figures 1-3.

Run: python experiments/run_all_experiments.py [--fast]
Outputs: results/tables/*.{md,csv,json}, results/figures/*.{png,pdf,svg}, data/manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.align import build_corridor, load_corridor
from src.data.contested_channel import REGIMES, ContestedChannel
from src.data.ingest import Manifest
from src.eval import figures as figs
from src.eval.metrics import anomaly_metrics, forecast_metrics
from src.eval.tables import print_table, write_table
from src.hmi.cognitive_autonomy import (CognitiveLoadModel, derive_events, hmi_metrics,
                                        run_adaptive, run_static)
from src.models.baselines import (CentralizedGNN, DecentralizedNN, GBM, Persistence, Ridge)
from src.models.decentralized_pinc import TrainConfig, predict as pinc_predict, train_dec_pinc
from src.security.telemetry_guard import TelemetryGuard
from src.utils.config import environment_info, load_configs
from src.utils.seed import set_seed

REGIME_ORDER = ["clean", "drop30", "drop60", "high_latency", "spoofing"]
FREE_FLOW_MPH = 70.0


def denorm_sf(obs, data):
    sp = obs[..., 0] * data.scaler_std[0] + data.scaler_mean[0]
    fl = obs[..., 1] * data.scaler_std[1] + data.scaler_mean[1]
    return sp, fl


def apply_quarantine(obs, quarantine_mask):
    """Replace quarantined telemetry (speed/flow) with the last trusted value
    (physics-propagated fallback)."""
    out = obs.copy()
    T, N, _ = obs.shape
    for j in range(N):
        last = obs[0, j, :2].copy()
        for t in range(T):
            if quarantine_mask[t, j]:
                out[t, j, 0], out[t, j, 1] = last
            else:
                last = obs[t, j, :2].copy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="fewer epochs for a quick run")
    ap.add_argument("--rebuild", action="store_true", help="re-ingest + rebuild corridor")
    args = ap.parse_args()

    set_seed(1337)
    cfg = load_configs(ROOT / "configs")
    data_cfg = cfg["data"]
    edges = data_cfg["edges"]
    proc = ROOT / "data" / "processed"
    raw = ROOT / "data" / data_cfg["paths"]["raw"].split("/")[-1]
    raw = ROOT / "data" / "raw"

    # --- data (real, cached) ---
    if args.rebuild or not (proc / "corridor.npz").exists():
        manifest = Manifest(ROOT / "data" / "manifest.json")
        data = build_corridor(data_cfg, raw, manifest=manifest, processed_dir=proc)
        manifest.write()
    else:
        data = load_corridor(proc)
    test_slice = data.splits["test"]
    print(f"Corridor: {data.X.shape[0]} steps x {data.n_nodes} nodes x {data.n_features} feat "
          f"| splits {data.splits}")

    # --- train models ---
    ep = dict(epochs_a=20, epochs_b=20, epochs_c=10) if args.fast else {}
    full_cfg = TrainConfig(physics=True, multistage=True, **ep)
    print("training Dec-PINC (full)...")
    pinc_full, _ = train_dec_pinc(data, full_cfg, edges, seed=1337)
    print("training ablations + baselines...")
    pinc_single, _ = train_dec_pinc(data, TrainConfig(physics=True, multistage=False, **ep),
                                    edges, seed=1337)
    baselines = {b.name: b.fit(data, edges) for b in
                 [Persistence(), Ridge(), GBM(), CentralizedGNN(), DecentralizedNN()]}

    guard = TelemetryGuard().fit(data, edges)

    def decpinc_predict(obs, use_quarantine=True, model=pinc_full):
        if use_quarantine:
            sp, fl = denorm_sf(obs, data)
            rep = guard.score(sp, fl)
            obs = apply_quarantine(obs, rep.quarantine_mask)
        return pinc_predict(model, data, obs, data.adjacency)

    # ---------- Table 1: main benchmark across contested regimes ----------
    methods = {"Dec-PINC-V2X": lambda o: decpinc_predict(o, True),
               **{name: (lambda o, m=m: m.predict(o, data.adjacency))
                  for name, m in baselines.items()}}
    t1_rows = []
    for regime in REGIME_ORDER:
        ch = ContestedChannel(seed=7)
        obs = REGIMES[regime](ch, data.X).obs
        for name, fn in methods.items():
            m = forecast_metrics(fn(obs), data, edges, test_slice)
            t1_rows.append({"Method": name, "Regime": regime, **{k: m[k] for k in
                           ("RMSE", "MAE", "MAPE", "PhysViolRate", "CtrlStabCost")}})
    cols1 = ["Method", "Regime", "RMSE", "MAE", "MAPE", "PhysViolRate", "CtrlStabCost"]
    write_table(ROOT / "results/tables", "table1_main_benchmark", t1_rows, cols1,
                "Table 1 - Main benchmark across contested regimes (test split)")
    print_table("TABLE 1 (main benchmark)", t1_rows, cols1)

    # ---------- Table 2: ablation ----------
    ch = ContestedChannel(seed=7)
    spoof_obs = REGIMES["spoofing"](ch, data.X).obs
    ch2 = ContestedChannel(seed=9)
    drop_obs = REGIMES["drop60"](ch2, data.X).obs
    t2_rows = []
    variants = {
        "Dec-PINC (full)": lambda o: decpinc_predict(o, True, pinc_full),
        "- physics loss": lambda o: _guarded(baselines["Decentralized NN (no physics)"],
                                              o, data, guard),
        "- multi-stage": lambda o: decpinc_predict(o, True, pinc_single),
        "- quarantine": lambda o: decpinc_predict(o, False, pinc_full),
    }
    for name, fn in variants.items():
        ms = forecast_metrics(fn(spoof_obs), data, edges, test_slice)
        md = forecast_metrics(fn(drop_obs), data, edges, test_slice)
        t2_rows.append({"Variant": name, "RMSE (spoof)": ms["RMSE"],
                        "PhysViol (spoof)": ms["PhysViolRate"], "RMSE (60% drop)": md["RMSE"],
                        "PhysViol (60% drop)": md["PhysViolRate"]})
    cols2 = ["Variant", "RMSE (spoof)", "PhysViol (spoof)", "RMSE (60% drop)", "PhysViol (60% drop)"]
    write_table(ROOT / "results/tables", "table2_ablation", t2_rows, cols2,
                "Table 2 - Ablation (impact of physics loss, multi-stage, quarantine)")
    print_table("TABLE 2 (ablation)", t2_rows, cols2)

    # ---------- Table 3: security + HMI ----------
    ch = ContestedChannel(seed=7)
    co = REGIMES["spoofing"](ch, data.X)
    sp, fl = denorm_sf(co.obs, data)
    rep = guard.score(sp, fl)
    sec = anomaly_metrics(rep.integrity_score, rep.anomaly_flag, co.attack_mask, test_slice)

    pred = decpinc_predict(co.obs, True)
    t0, t1 = test_slice
    unc = _uncertainty_proxy(co.obs, data)
    events = derive_events(pred["speed"][t0:t1], data.speed[t0:t1],
                           rep.integrity_score[t0:t1], rep.anomaly_flag[t0:t1],
                           co.attack_mask[t0:t1], FREE_FLOW_MPH, unc[t0:t1])
    static = run_static(events)
    adaptive = run_adaptive(events, pred["speed"][t0:t1], FREE_FLOW_MPH)
    hm = hmi_metrics(static, adaptive, events)
    t3_rows = [
        {"Metric": "Anomaly AUROC", "Value": sec["AUROC"]},
        {"Metric": "Anomaly F1", "Value": sec["F1"]},
        {"Metric": "Detection delay (steps)", "Value": sec["DetectionDelaySteps"]},
        {"Metric": "Operator alert-volume reduction (%)", "Value": hm["alert_volume_reduction_pct"]},
        {"Metric": "High-criticality recall (%)", "Value": hm["high_criticality_recall_pct"]},
        {"Metric": "Missed critical escalation rate (%)", "Value": hm["missed_critical_escalation_rate_pct"]},
    ]
    write_table(ROOT / "results/tables", "table3_security_hmi", t3_rows, ["Metric", "Value"],
                "Table 3 - Security & HMI operational metrics (spoofing regime, test split)")
    print_table("TABLE 3 (security + HMI)", t3_rows, ["Metric", "Value"])

    # ---------- Figures ----------
    _make_figures(data, edges, pinc_full, baselines, guard, decpinc_predict, pred,
                  co, rep, unc, test_slice)

    # environment + provenance
    (ROOT / "results" / "environment.json").write_text(json.dumps(environment_info(), indent=2))
    print("\nDone. Tables in results/tables/, figures in results/figures/.")


def _guarded(model, obs, data, guard):
    sp = obs[..., 0] * data.scaler_std[0] + data.scaler_mean[0]
    fl = obs[..., 1] * data.scaler_std[1] + data.scaler_mean[1]
    rep = guard.score(sp, fl)
    oq = apply_quarantine(obs, rep.quarantine_mask)
    return model.predict(oq, data.adjacency)


def _uncertainty_proxy(obs, data):
    """Predictive-uncertainty proxy: normalized |obs - neighbour mean| of speed."""
    from src.models.baselines import _neighbor_mean
    nb = _neighbor_mean(obs, data.adjacency)[..., 0]
    u = np.abs(obs[..., 0] - nb)
    return np.clip(u / (u.std() + 1e-6) / 3.0, 0, 1)


def _make_figures(data, edges, pinc_full, baselines, guard, decpinc_predict, pred,
                  co, rep, unc, test_slice):
    out = ROOT / "results" / "figures"
    # fig1: degradation sweep
    loss_pcts = [0, 10, 30, 50, 70]
    curves = {"Dec-PINC-V2X": {"rmse": [], "pvr": []},
              "Centralized NN": {"rmse": [], "pvr": []},
              "Decentralized NN": {"rmse": [], "pvr": []}}
    cen = baselines["Centralized NN (no physics)"]
    dnn = baselines["Decentralized NN (no physics)"]
    for p in loss_pcts:
        ch = ContestedChannel(seed=11)
        obs = ch.packet_drop(data.X, p / 100.0, bursty=True).obs if p > 0 else data.X.copy()
        for label, fn in (("Dec-PINC-V2X", lambda o: decpinc_predict(o, True)),
                          ("Centralized NN", lambda o: cen.predict(o, data.adjacency)),
                          ("Decentralized NN", lambda o: dnn.predict(o, data.adjacency))):
            m = forecast_metrics(fn(obs), data, edges, test_slice)
            curves[label]["rmse"].append(m["RMSE"]); curves[label]["pvr"].append(m["PhysViolRate"])
    figs.fig1_degradation_curves(loss_pcts, curves, out)

    # fig2: spoofing resilience trace at an attacked node
    t0, t1 = test_slice
    atk_nodes = np.where(co.attack_mask[t0:t1].any(0))[0]
    node = int(atk_nodes[0]) if len(atk_nodes) else 0
    t = np.arange(t1 - t0)
    figs.fig2_spoofing_resilience(
        t, data.speed[t0:t1, node], (co.obs[t0:t1, node, 0] * data.scaler_std[0] + data.scaler_mean[0]),
        pred["speed"][t0:t1, node], rep.quarantine_mask[t0:t1, node], node, out)

    # fig3: cognitive alert tradeoff (sweep adaptive load cap; static sweeps threshold)
    from src.hmi.cognitive_autonomy import derive_events, run_adaptive, run_static, hmi_metrics
    events = derive_events(pred["speed"][t0:t1], data.speed[t0:t1], rep.integrity_score[t0:t1],
                           rep.anomaly_flag[t0:t1], co.attack_mask[t0:t1], FREE_FLOW_MPH, unc[t0:t1])
    static_pts, adaptive_pts = [], []
    for thr in (0.2, 0.35, 0.5, 0.65):
        s = run_static(events, alert_threshold=thr)
        m = hmi_metrics(s, s, events)
        static_pts.append((m["static_alert_volume"], m["high_criticality_recall_pct"]))
    for cap in (0.5, 1.0, 1.5, 2.0, 3.0):
        a = run_adaptive(events, pred["speed"][t0:t1], FREE_FLOW_MPH, load_cap=cap)
        m = hmi_metrics(run_static(events), a, events)
        adaptive_pts.append((m["adaptive_alert_volume"], m["high_criticality_recall_pct"]))
    figs.fig3_cognitive_alert_tradeoff(static_pts, adaptive_pts, out)


FREE_FLOW_MPH = 70.0

if __name__ == "__main__":
    main()
