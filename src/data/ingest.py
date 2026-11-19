"""Real public data ingestion with deterministic caching + provenance logging.

Three keyless public sources, all region-coherent (Great Britain):

* Traffic  : UK National Highways **WebTRIS** MIDAS loop detectors (M25 corridor),
             15-minute average speed and total volume (flow).
             https://webtris.nationalhighways.co.uk/api
* Grid     : **NESO / National Grid ESO** historic demand (CKAN datastore),
             half-hourly national demand (ND) and transmission system demand (TSD).
             https://api.neso.energy
* Weather  : **Open-Meteo** archive API, hourly temperature / wind / precipitation,
             aligned to each corridor node's lat/lon.
             https://archive-api.open-meteo.com

Every response is cached under ``data/raw/`` (re-runs are offline and deterministic)
and a provenance record is appended to ``data/manifest.json`` (source URL, fetch
timestamp, spatial scope, sample counts, missingness).
"""
from __future__ import annotations

import io
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

WEBTRIS = "https://webtris.nationalhighways.co.uk/api/v1.0"
NESO = "https://api.neso.energy/api/3/action"
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
NESO_DEMAND_RESOURCE = "f6d02c0f-957b-48cb-82ee-09003f2ba759"   # historic demand 2024
USER_AGENT = "dpinc-contested-corridors/0.1 (research; keyless public data)"


def _get(url: str, timeout: int = 60, retries: int = 4) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # transient network/server errors
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url}\n{last}")


@dataclass
class Manifest:
    path: Path
    entries: List[dict] = field(default_factory=list)

    def add(self, **kw) -> None:
        kw["fetched_at"] = datetime.now(timezone.utc).isoformat()
        self.entries.append(kw)

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"generated": datetime.now(timezone.utc).isoformat(),
             "sources": self.entries}, indent=2))


# --------------------------------------------------------------------------- #
# Traffic: WebTRIS                                                            #
# --------------------------------------------------------------------------- #
def _ddmmyyyy(d: str) -> str:
    return datetime.fromisoformat(d).strftime("%d%m%Y")


def fetch_traffic_site(site_id: str, start: str, end: str, cache_dir: Path,
                       manifest: Optional[Manifest] = None) -> pd.DataFrame:
    """Fetch one WebTRIS site's 15-min speed/flow for [start, end] (ISO dates)."""
    cache = cache_dir / f"webtris_site{site_id}_{start}_{end}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["timestamp"])
    rows: List[dict] = []
    page = 1
    base = (f"{WEBTRIS}/reports/daily?sites={site_id}"
            f"&start_date={_ddmmyyyy(start)}&end_date={_ddmmyyyy(end)}&page_size=5000")
    while True:
        data = json.loads(_get(f"{base}&page={page}"))
        chunk = data.get("Rows", []) or []
        for r in chunk:
            date = r.get("Report Date", "")[:10]
            tpe = r.get("Time Period Ending", "")
            if not date or not tpe:
                continue
            ts = pd.to_datetime(f"{date} {tpe}", errors="coerce")
            if pd.isna(ts):
                continue
            rows.append({
                "timestamp": ts,
                "site_id": site_id,
                "speed_mph": _num(r.get("Avg mph")),
                "flow_veh": _num(r.get("Total Volume")),
            })
        header = data.get("Header", {}) or {}
        links = header.get("links", []) or []
        if not any(l.get("rel") == "nextPage" for l in links) or not chunk:
            break
        page += 1
    df = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    if manifest is not None:
        manifest.add(source="WebTRIS", stream="traffic", site_id=site_id,
                     url=f"{base}&page=1", date_range=[start, end],
                     rows=int(len(df)),
                     missing_speed=float(df["speed_mph"].isna().mean()) if len(df) else 1.0,
                     missing_flow=float(df["flow_veh"].isna().mean()) if len(df) else 1.0)
    return df


def fetch_traffic(sites: Sequence[dict], start: str, end: str, cache_dir: Path,
                  manifest: Optional[Manifest] = None) -> pd.DataFrame:
    """Fetch all corridor sites; returns long DataFrame [timestamp, site_id, speed, flow]."""
    frames = [fetch_traffic_site(s["id"], start, end, cache_dir, manifest) for s in sites]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _num(v) -> float:
    try:
        s = str(v).strip().replace(",", "")
        return float(s) if s not in ("", "-", "nan", "None") else np.nan
    except Exception:
        return np.nan


# --------------------------------------------------------------------------- #
# Grid: NESO historic demand                                                  #
# --------------------------------------------------------------------------- #
def fetch_grid(start: str, end: str, cache_dir: Path,
               resource_id: str = NESO_DEMAND_RESOURCE,
               manifest: Optional[Manifest] = None) -> pd.DataFrame:
    """Fetch NESO half-hourly national demand for [start, end]; timestamps built
    from SETTLEMENT_DATE + SETTLEMENT_PERIOD (period p -> (p-1)*30 min)."""
    cache = cache_dir / f"neso_demand_{start}_{end}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["timestamp"])
    sql = (f'SELECT "SETTLEMENT_DATE","SETTLEMENT_PERIOD","ND","TSD" '
           f'FROM "{resource_id}" '
           f"WHERE \"SETTLEMENT_DATE\" >= '{start}' AND \"SETTLEMENT_DATE\" <= '{end}' "
           f'ORDER BY "SETTLEMENT_DATE","SETTLEMENT_PERIOD"')
    url = f"{NESO}/datastore_search_sql?" + urllib.parse.urlencode({"sql": sql})
    data = json.loads(_get(url))
    recs = data.get("result", {}).get("records", [])
    rows = []
    for r in recs:
        p = int(float(r["SETTLEMENT_PERIOD"]))
        if not 1 <= p <= 48:
            continue
        ts = pd.to_datetime(str(r["SETTLEMENT_DATE"])[:10]) + pd.Timedelta(minutes=30 * (p - 1))
        rows.append({"timestamp": ts, "nd_mw": _num(r.get("ND")),
                     "tsd_mw": _num(r.get("TSD"))})
    df = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    if manifest is not None:
        manifest.add(source="NESO", stream="grid", resource_id=resource_id,
                     url=f"{NESO}/datastore_search_sql", date_range=[start, end],
                     rows=int(len(df)),
                     missing_nd=float(df["nd_mw"].isna().mean()) if len(df) else 1.0)
    return df


# --------------------------------------------------------------------------- #
# Weather: Open-Meteo archive                                                 #
# --------------------------------------------------------------------------- #
def fetch_weather(lat: float, lon: float, start: str, end: str, cache_dir: Path,
                  manifest: Optional[Manifest] = None) -> pd.DataFrame:
    """Fetch hourly weather for a node location; returns [timestamp, temp_c, wind_kmh, precip_mm]."""
    key = f"{round(lat,2)}_{round(lon,2)}"
    cache = cache_dir / f"openmeteo_{key}_{start}_{end}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["timestamp"])
    params = {"latitude": round(lat, 3), "longitude": round(lon, 3),
              "start_date": start, "end_date": end,
              "hourly": "temperature_2m,wind_speed_10m,precipitation", "timezone": "UTC"}
    url = f"{OPEN_METEO}?" + urllib.parse.urlencode(params)
    data = json.loads(_get(url))
    h = data.get("hourly", {})
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(h.get("time", [])),
        "temp_c": h.get("temperature_2m", []),
        "wind_kmh": h.get("wind_speed_10m", []),
        "precip_mm": h.get("precipitation", []),
    })
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    if manifest is not None:
        manifest.add(source="Open-Meteo", stream="weather", location=[lat, lon],
                     url=url, date_range=[start, end], rows=int(len(df)),
                     missing_temp=float(df["temp_c"].isna().mean()) if len(df) else 1.0)
    return df
