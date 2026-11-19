"""Publication figures: 300 DPI, Times New Roman, colorblind-safe, png+pdf+svg."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Colorblind-safe (Okabe-Ito).
CB = {"decpinc": "#D55E00", "centralized": "#0072B2", "decnn": "#009E73",
      "gbm": "#CC79A7", "ridge": "#56B4E9", "persistence": "#999999",
      "static": "#0072B2", "adaptive": "#D55E00"}


def _style():
    for fam in ("Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"):
        matplotlib.rcParams["font.serif"] = [fam]
    matplotlib.rcParams.update({"font.family": "serif", "font.size": 11,
                                "axes.titleweight": "bold", "figure.facecolor": "white",
                                "savefig.facecolor": "white"})


def _save(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig1_degradation_curves(loss_pcts: Sequence[float],
                            curves: Dict[str, Dict[str, List[float]]],
                            out_dir: Path):
    """curves[label] = {'rmse': [...], 'pvr': [...]} over loss_pcts."""
    _style()
    palette = {"Dec-PINC-V2X": "#D55E00", "Centralized NN": "#0072B2",
               "Decentralized NN": "#009E73"}
    markers = {"Dec-PINC-V2X": "o", "Centralized NN": "s", "Decentralized NN": "^"}
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.3))
    for label, series in curves.items():
        c = palette.get(label, "#555555")
        mk = markers.get(label, "o")
        axl.plot(loss_pcts, series["rmse"], f"-{mk}", label=label, color=c, lw=2)
        axr.plot(loss_pcts, series["pvr"], f"-{mk}", label=label, color=c, lw=2)
    axl.set_xlabel("Packet loss (%)"); axl.set_ylabel("Speed RMSE (mph)")
    axl.set_title("(a) Forecast error vs. packet loss")
    axr.set_xlabel("Packet loss (%)"); axr.set_ylabel("Physics-violation rate")
    axr.set_title("(b) Physics-violation rate vs. packet loss")
    for ax in (axl, axr):
        ax.grid(True, ls=":", alpha=0.5)
    axl.legend(frameon=False, fontsize=8.5)
    _save(fig, out_dir, "fig1_degradation_curves")


def fig2_spoofing_resilience(t, actual, spoofed, pred, quarantine, node_id, out_dir: Path):
    _style()
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.plot(t, actual, color="#009E73", lw=2, label="actual speed (clean)")
    ax.plot(t, spoofed, color="#999999", lw=1.2, ls="--", label="spoofed telemetry")
    ax.plot(t, pred, color="#D55E00", lw=2, label="Dec-PINC estimate")
    q = np.asarray(quarantine, bool)
    if q.any():
        ax.fill_between(t, ax.get_ylim()[0], ax.get_ylim()[1], where=q, color="#D55E00",
                        alpha=0.12, label="quarantine window")
    ax.set_xlabel("Time step (test window)"); ax.set_ylabel("Speed (mph)")
    ax.set_title(f"Spoofing resilience at node {node_id}: detection + state preservation")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.grid(True, ls=":", alpha=0.5)
    _save(fig, out_dir, "fig2_spoofing_resilience")


def fig3_cognitive_alert_tradeoff(points_static, points_adaptive, out_dir: Path):
    """points_* = list of (alert_volume, critical_capture_pct)."""
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for pts, label, c, mk in ((points_static, "Static alerting", CB["static"], "s"),
                              (points_adaptive, "Cognitive-calibrated", CB["adaptive"], "o")):
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], f"-{mk}", color=c,
                lw=2, label=label, ms=7)
    ax.set_xlabel("Operator alert volume (lower = less burden)")
    ax.set_ylabel("Critical-event capture (%)")
    ax.set_title("Operator burden vs. critical capture (Pareto)")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.grid(True, ls=":", alpha=0.5)
    _save(fig, out_dir, "fig3_cognitive_alert_tradeoff")
