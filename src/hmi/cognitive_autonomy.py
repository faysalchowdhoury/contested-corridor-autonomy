"""Cognitive-load-aware adaptive autonomy decision support.

Combines predictive uncertainty, telemetry integrity, operational criticality and a
quantitative operator cognitive-load proxy to decide, per candidate event, whether
to (a) alert the operator, (b) handle routine stabilization autonomously, or
(c) escalate with a short physics-grounded explanation.

Operator cognitive load:
    L(t) = EWMA(alert arrival rate) + beta * concurrent_unresolved + gamma * complexity
Static Alerting fires on every breach; Cognitive-Calibrated Adaptive Autonomy
suppresses low-criticality alerts when L is high, auto-handles routine cases, and
always escalates high-consequence events.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Event:
    t: int
    node: int
    criticality: float          # [0, 1] operational consequence
    uncertainty: float          # [0, 1] predictive uncertainty
    integrity: float            # [0, 1] telemetry integrity
    is_critical: bool           # ground truth: truly high-consequence
    kind: str = "congestion"    # or "security"


@dataclass
class Decision:
    event: Event
    action: str                 # "alert" | "autonomous" | "escalate" | "suppressed"
    explanation: str = ""


def derive_events(pred_speed: np.ndarray, actual_speed: np.ndarray,
                  guard_integrity: np.ndarray, guard_flag: np.ndarray,
                  attack_mask: np.ndarray, free_flow_mph: float,
                  uncertainty: np.ndarray) -> List[Event]:
    """Scan the horizon and emit candidate events with criticality + ground truth."""
    T, N = pred_speed.shape
    events: List[Event] = []
    for t in range(T):
        for n in range(N):
            cong = np.clip((free_flow_mph - pred_speed[t, n]) / free_flow_mph, 0, 1)
            sec = (1.0 - guard_integrity[t, n]) if guard_flag[t, n] else 0.0
            crit = float(max(cong, sec))
            if crit < 0.2 and not guard_flag[t, n]:
                continue                        # not worth surfacing at all
            # ground-truth critical: real severe congestion OR a true attack
            true_cong = (free_flow_mph - actual_speed[t, n]) / free_flow_mph > 0.6
            is_crit = bool(true_cong or attack_mask[t, n])
            kind = "security" if sec >= cong else "congestion"
            events.append(Event(t, n, crit, float(uncertainty[t, n]),
                                float(guard_integrity[t, n]), is_crit, kind))
    return events


@dataclass
class CognitiveLoadModel:
    halflife: float = 12.0
    beta: float = 0.15
    gamma: float = 0.1

    def __post_init__(self):
        self.decay = 0.5 ** (1.0 / self.halflife)
        self.rate = 0.0

    def update(self, alerts_this_step: int, unresolved: int, complexity: float) -> float:
        self.rate = self.decay * self.rate + (1 - self.decay) * alerts_this_step
        return self.rate + self.beta * unresolved + self.gamma * complexity


def _explanation(ev: Event, pred_speed: float, free_flow: float) -> str:
    """Structured, physics-grounded 2-sentence escalation rationale."""
    if ev.kind == "security":
        return (f"Node {ev.node}: telemetry integrity {ev.integrity:.2f} with a flow-"
                f"conservation violation versus neighbours. Inputs quarantined; the "
                f"controller is running on physics-propagated state estimates.")
    drop = 100 * (free_flow - pred_speed) / free_flow
    return (f"Node {ev.node}: predicted speed {pred_speed:.0f} mph, a {drop:.0f}% drop "
            f"below free-flow, implying an forming queue. Recommend upstream ramp "
            f"metering before spillback propagates to adjacent segments.")


def run_static(events: List[Event], alert_threshold: float = 0.2):
    """Static policy: an operator-facing alert for every breach above threshold."""
    alerts = [Decision(e, "alert") for e in events if e.criticality >= alert_threshold]
    return alerts


def run_adaptive(events: List[Event], pred_speed: np.ndarray, free_flow: float,
                 crit_high: float = 0.55, load_cap: float = 2.0,
                 clm: Optional[CognitiveLoadModel] = None):
    """Cognitive-calibrated adaptive autonomy."""
    clm = clm or CognitiveLoadModel()
    by_t: Dict[int, List[Event]] = {}
    for e in events:
        by_t.setdefault(e.t, []).append(e)
    decisions: List[Decision] = []
    unresolved = 0
    for t in sorted(by_t):
        shown = 0
        load = clm.update(0, unresolved, complexity=len(by_t[t]) / 8.0)
        for e in sorted(by_t[t], key=lambda x: -x.criticality):
            if e.criticality >= crit_high:
                decisions.append(Decision(e, "escalate",
                                 _explanation(e, pred_speed[e.t, e.node], free_flow)))
                shown += 1
                unresolved += 1
            elif e.criticality < 0.35:
                # routine: autonomous stabilization, no operator alert (esp. under load)
                decisions.append(Decision(e, "autonomous" if load >= load_cap else "alert"))
                shown += 0 if load >= load_cap else 1
            else:
                if load >= load_cap and e.uncertainty < 0.5:
                    decisions.append(Decision(e, "suppressed"))
                else:
                    decisions.append(Decision(e, "alert"))
                    shown += 1
        clm.update(shown, unresolved, complexity=len(by_t[t]) / 8.0)
        unresolved = max(0, unresolved - 1)        # operators resolve over time
    return decisions


def hmi_metrics(static_alerts, adaptive_decisions, events) -> Dict[str, float]:
    """Alert-volume reduction, high-criticality recall, missed-critical rate."""
    n_static = len(static_alerts)
    operator_facing = [d for d in adaptive_decisions if d.action in ("alert", "escalate")]
    n_adaptive = len(operator_facing)
    crit_events = [e for e in events if e.is_critical]
    escalated_crit = [d for d in operator_facing if d.event.is_critical]
    # unique critical events surfaced
    surfaced = {(d.event.t, d.event.node) for d in escalated_crit}
    crit_keys = {(e.t, e.node) for e in crit_events}
    recall = len(surfaced & crit_keys) / max(1, len(crit_keys))
    return {
        "static_alert_volume": n_static,
        "adaptive_alert_volume": n_adaptive,
        "alert_volume_reduction_pct": 100.0 * (1 - n_adaptive / max(1, n_static)),
        "high_criticality_recall_pct": 100.0 * recall,
        "missed_critical_escalation_rate_pct": 100.0 * (1 - recall),
    }
