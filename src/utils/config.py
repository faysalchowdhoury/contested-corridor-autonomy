"""YAML config loading + environment capture for reproducibility."""
from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_configs(cfg_dir: str | Path = "configs") -> Dict[str, Any]:
    cfg_dir = Path(cfg_dir)
    out: Dict[str, Any] = {}
    for name in ("data", "model", "experiment"):
        p = cfg_dir / f"{name}.yaml"
        if p.exists():
            out[name] = load_yaml(p)
    return out


def environment_info() -> Dict[str, str]:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": _ver("numpy"),
        "torch": _ver("torch"),
        "lightgbm": _ver("lightgbm"),
        "sklearn": _ver("sklearn"),
        "pandas": _ver("pandas"),
    }
    return info


def _ver(mod: str) -> str:
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "unknown")
    except Exception:
        return "not installed"
