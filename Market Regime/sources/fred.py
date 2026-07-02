"""FRED data source — public fredgraph CSV endpoint (no API key needed).

Full history, same values as the official API. If FRED ever gates this
endpoint, swap in the keyed API here — callers won't notice.

Uses stdlib urllib via common.http_get — FRED's CDN drops requests/urllib3
TLS handshakes (see common.py).
"""
from __future__ import annotations

import io
import urllib.parse

import pandas as pd

from .common import cache_load, cache_save, http_get


class FredClient:
    CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def get_series(self, series_id: str, force_refresh: bool = False) -> pd.Series:
        """Return the full history of one FRED series as a float Series."""
        cache_name = f"fred_{series_id}"
        if not force_refresh:
            cached = cache_load(cache_name)
            if cached is not None:
                return cached.iloc[:, 0].astype(float)

        # cosd pins the chart start date: without it fredgraph falls back to
        # each series' default graph window, which for some series (e.g. the
        # ICE BofA credit indices) is only the last few years, silently
        # truncating history. 1776-07-04 is FRED's own minimum date.
        params = {"id": series_id, "cosd": "1776-07-04"}
        url = f"{self.CSV_URL}?{urllib.parse.urlencode(params)}"
        raw = http_get(url, timeout=self.timeout)

        df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
        date_col = df.columns[0]          # "observation_date"
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        # FRED encodes missing values as "."
        series = pd.to_numeric(df[series_id], errors="coerce").dropna()
        series.name = series_id

        cache_save(cache_name, series.to_frame())
        return series
