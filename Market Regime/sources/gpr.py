"""GPR data source — Geopolitical Risk index (Caldara & Iacoviello).

Monthly headline index from 1900, free, updated regularly at
matteoiacoviello.com. The export file also carries threat/act sub-indices
(GPRT/GPRA) and country-level series — extend here if those become factors.
"""
from __future__ import annotations

import io

import pandas as pd

from .common import cache_load, cache_save, http_get

GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"


def get_gpr_index(force_refresh: bool = False) -> pd.Series:
    """Return the monthly headline GPR index as a float Series."""
    cache_name = "gpr_monthly"
    if not force_refresh:
        cached = cache_load(cache_name)
        if cached is not None:
            return cached.iloc[:, 0].astype(float)

    raw = http_get(GPR_URL, timeout=60)
    df = pd.read_excel(io.BytesIO(raw))
    date_col = next(
        (c for c in df.columns if str(c).lower() in ("month", "date")), df.columns[0]
    )
    gpr_col = next((c for c in df.columns if str(c).upper() == "GPR"), None)
    if gpr_col is None:
        raise ValueError(f"GPR column not found; columns={list(df.columns)[:15]}")

    df[date_col] = pd.to_datetime(df[date_col])
    series = (
        df.set_index(date_col)[gpr_col].astype(float).dropna().sort_index()
    )
    series.name = "GPR"

    cache_save(cache_name, series.to_frame())
    return series
