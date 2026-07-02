"""CBOE data source — frozen put/call ratio archive (2006-11 → 2019-10).

CBOE's free totalpc.csv stopped updating in October 2019; current values
come from OCC (sources/occ.py). get_putcall_ratio() splices the two into
one continuous daily series.

Caveat encoded here on purpose: the CBOE ratio covers CBOE-exchange volume
only, while the OCC ratio covers ALL US options exchanges — levels differ
slightly across the 2019 seam. Fine for z-scored regime work; don't use the
raw spliced level across the seam for anything precise.
"""
from __future__ import annotations

import io

import pandas as pd

from .common import cache_load, cache_save, http_get

CBOE_ARCHIVE_URL = (
    "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv"
)


def get_cboe_putcall_archive(force_refresh: bool = False) -> pd.Series:
    """CBOE total put/call ratio, 2006-11-01 → 2019-10-04 (frozen archive)."""
    cache_name = "cboe_putcall_archive"
    if not force_refresh:
        cached = cache_load(cache_name)
        if cached is not None:
            return cached.iloc[:, 0].astype(float)

    raw = http_get(CBOE_ARCHIVE_URL, timeout=30)
    # Row 0: disclaimer, row 1: product banner, row 2: real header
    df = pd.read_csv(io.StringIO(raw.decode("utf-8")), skiprows=2)
    df.columns = [c.strip() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"])
    series = df.set_index("DATE")["P/C Ratio"].astype(float).dropna().sort_index()
    series.name = "CBOE_PC_RATIO"

    cache_save(cache_name, series.to_frame())
    return series


def get_putcall_ratio(force_refresh: bool = False) -> pd.Series:
    """Continuous daily put/call ratio: CBOE archive + OCC from 2019-10-07."""
    from .occ import get_occ_putcall

    cboe = get_cboe_putcall_archive(force_refresh=force_refresh)
    occ = get_occ_putcall(force_refresh=force_refresh)

    seam = cboe.index[-1]
    spliced = pd.concat([cboe, occ[occ.index > seam]]).sort_index()
    spliced.name = "PC_RATIO"
    return spliced
