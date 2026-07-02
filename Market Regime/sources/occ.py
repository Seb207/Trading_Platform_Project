"""OCC (Options Clearing Corporation) data source — free market data API.

Verified live 2026-07-02:
  /mdapi/daily-volume-totals?report_date=YYYY-MM-DD
      → per-exchange calls/puts volume for ONE day (put/call ratio input).
        Data from ~2018; empty entity lists before that.
  /mdapi/open-interest?report_date=MM/DD/YYYY          (note: different format!)
      → total OCC open interest, ~21 business days (one month) per call,
        so historical backfill costs ~1 request per month.

The daily put/call backfill needs one request per business day (~1,700 for
2019→present), so it is RESUMABLE: progress is cached every 50 days and
re-running continues from the last cached date. Interrupting is safe.
"""
from __future__ import annotations

import datetime as dt
import json
import urllib.parse

import pandas as pd

from .common import cache_load, cache_save, http_get

_VOLUME_URL = "https://marketdata.theocc.com/mdapi/daily-volume-totals"
_OI_URL = "https://marketdata.theocc.com/mdapi/open-interest"

# daily-volume-totals returns empty lists before ~2018; OCC put/call is used
# to extend the frozen CBOE archive (ends 2019-10), so start at the overlap.
PUTCALL_START = dt.date(2019, 1, 2)
OI_START = dt.date(2018, 1, 31)


def _get_json(base_url: str, report_date: str) -> dict:
    url = f"{base_url}?{urllib.parse.urlencode({'report_date': report_date})}"
    # Low politeness delay: the day-by-day backfill makes ~1,700 calls
    return json.loads(http_get(url, timeout=20, retries=1, polite_delay=0.1))


def get_occ_putcall(force_refresh: bool = False) -> pd.Series:
    """Daily total US options put/call VOLUME ratio from OCC (2019–present).

    Incremental: only dates after the last cached date are fetched, so the
    steady-state cost is one request per new business day.
    """
    cache_name = "occ_putcall"
    cached = None if force_refresh else cache_load(cache_name)

    values: dict[pd.Timestamp, float] = {}
    start = PUTCALL_START
    if cached is not None and len(cached) > 0:
        series = cached.iloc[:, 0].astype(float)
        values = dict(series.items())
        start = (series.index[-1] + pd.Timedelta(days=1)).date()

    today = dt.date.today()
    dates = pd.bdate_range(start, today)
    fetched = 0

    for ts in dates:
        try:
            entity = _get_json(_VOLUME_URL, ts.strftime("%Y-%m-%d")).get("entity", {})
        except Exception:
            continue                       # transient failure — next run retries
        rows = entity.get("total_volume") or []
        calls = sum(r.get("calls", 0) for r in rows)
        puts = sum(r.get("puts", 0) for r in rows)
        if calls > 0:
            values[ts] = puts / calls
        fetched += 1
        if fetched % 50 == 0:              # resumable checkpoint
            _save(cache_name, values, "PC_RATIO")

    if not values:
        raise ValueError("OCC put/call: no data collected")
    return _save(cache_name, values, "PC_RATIO")


def get_occ_open_interest(force_refresh: bool = False) -> pd.Series:
    """Daily total OCC options open interest (occTotal), 2018–present.

    /open-interest returns ~one month per call → backfill is one request
    per month (month-end anchors), incremental thereafter.
    """
    cache_name = "occ_open_interest"
    cached = None if force_refresh else cache_load(cache_name)

    values: dict[pd.Timestamp, float] = {}
    start = OI_START
    if cached is not None and len(cached) > 0:
        series = cached.iloc[:, 0].astype(float)
        values = dict(series.items())
        start = series.index[-1].date()    # re-fetch last month (fills gaps)

    # One anchor per month-end from start to today, plus today itself
    anchors = list(pd.date_range(start, dt.date.today(), freq="ME"))
    anchors.append(pd.Timestamp(dt.date.today()))

    for anchor in anchors:
        try:
            entity = _get_json(_OI_URL, anchor.strftime("%m/%d/%Y")).get("entity", {})
        except Exception:
            continue
        for row in entity.get("optionsOI") or []:
            ts = pd.Timestamp(dt.datetime.fromtimestamp(row["activityDate"] / 1000).date())
            total = row.get("occTotal", 0)
            if total:
                values[ts] = float(total)

    if not values:
        raise ValueError("OCC open interest: no data collected")
    return _save(cache_name, values, "OCC_TOTAL_OI")


def _save(cache_name: str, values: dict, col: str) -> pd.Series:
    series = pd.Series(values).sort_index()
    series.name = col
    cache_save(cache_name, series.to_frame())
    return series
