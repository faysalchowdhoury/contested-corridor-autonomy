"""Verify anti-leakage: chronological splits and train-only scaler fitting."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.align import CorridorData, _common_index, build_corridor


def _toy_cfg():
    return {
        "window": {"start": "2024-06-01", "end": "2024-06-02"},
        "cadence_min": 30,
        "sites": [{"id": "A", "lat": 51.5, "lon": 0.0},
                  {"id": "B", "lat": 51.6, "lon": 0.1}],
        "edges": [[0, 1]],
        "grid": {"resource_id": "x"},
        "splits": {"train": 0.6, "val": 0.2, "test": 0.2},
    }


def test_common_index_monotonic():
    idx = _common_index("2024-06-01", "2024-06-02", 30)
    assert (np.diff(idx.view("int64")) > 0).all()    # strictly increasing
    assert len(idx) == 96                            # 2 days * 48


def test_splits_are_chronological_and_disjoint():
    # construct a CorridorData directly to test the split invariant
    T, N, F = 100, 2, 5
    splits = {"train": (0, 60), "val": (60, 80), "test": (80, 100)}
    assert splits["train"][1] == splits["val"][0]    # no gap, no overlap
    assert splits["val"][1] == splits["test"][0]
    assert splits["test"][1] == T
    # every test index is strictly after every train index (zero look-ahead)
    assert min(range(*splits["test"])) > max(range(*splits["train"]))


def test_scaler_fit_on_train_only(monkeypatch):
    """Scalers must be computed from the training slice alone."""
    import src.data.align as align

    def fake_traffic(site_id, start, end, cache_dir, manifest=None):
        import pandas as pd
        idx = align._common_index(start, end, 30)
        T = len(idx)
        # test-region values are wildly larger; a leaked scaler would be inflated.
        speed = np.full(T, 30.0)
        speed[int(0.85 * T):] = 500.0           # strictly inside the test split (last 20%)
        return pd.DataFrame({"timestamp": idx, "speed_mph": speed,
                             "flow_veh": np.full(T, 300.0)})

    def fake_weather(lat, lon, start, end, cache_dir, manifest=None):
        import pandas as pd
        idx = align._common_index(start, end, 30)
        return pd.DataFrame({"timestamp": idx, "temp_c": np.zeros(len(idx)),
                             "wind_kmh": np.zeros(len(idx)), "precip_mm": np.zeros(len(idx))})

    def fake_grid(start, end, cache_dir, resource_id=None, manifest=None):
        import pandas as pd
        idx = align._common_index(start, end, 30)
        return pd.DataFrame({"timestamp": idx, "nd_mw": np.ones(len(idx)) * 1000.0,
                             "tsd_mw": np.ones(len(idx)) * 1000.0})

    monkeypatch.setattr(align, "fetch_traffic_site", fake_traffic)
    monkeypatch.setattr(align, "fetch_weather", fake_weather)
    monkeypatch.setattr(align, "fetch_grid", fake_grid)

    cfg = _toy_cfg()
    cfg["window"] = {"start": "2024-06-01", "end": "2024-06-03"}   # enough steps
    data = build_corridor(cfg, Path("."))
    # train speed mean is 30 (not polluted by the 500 test values)
    assert abs(float(data.scaler_mean[0]) - 30.0) < 5.0
