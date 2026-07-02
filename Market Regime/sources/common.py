"""Shared helpers for all data-source clients: raw cache + HTTP fetch.

Every source client writes its raw series to data/raw/<name>.csv so repeated
builds don't re-hit external APIs. Pass force_refresh=True at the call site
to bypass the cache.

HTTP goes through a curl subprocess with curl's DEFAULT User-Agent.
Empirically verified against FRED's CDN (2026-07-02):
  - python HTTP stacks (requests, urllib) → connection times out
  - curl with a custom/app UA → HTTP/2 INTERNAL_ERROR (exit 92)
  - curl with its default UA → instant 200
So: do NOT attach a custom UA here. urllib remains as a last-resort
fallback for environments without curl.
"""
from __future__ import annotations

import subprocess
import time
import urllib.request
from pathlib import Path

import pandas as pd

# data/ lives at the Market Regime module root (one level above sources/)
_MODULE_DIR = Path(__file__).parent.parent.resolve()
RAW_CACHE_DIR = _MODULE_DIR / "data" / "raw"

USER_AGENT = "MarketRegimeDetector/0.1 (personal research)"


def http_get(url: str, timeout: int = 30, retries: int = 2,
             polite_delay: float = 0.5) -> bytes:
    """Fetch a URL — curl subprocess first (see module docstring), urllib fallback.

    Retries with a short backoff; polite_delay after every call keeps burst
    rates low (lower it for bulk day-by-day backfills like OCC's).
    """
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            # No -A flag on purpose — see module docstring (custom UAs get blocked)
            proc = subprocess.run(
                ["curl", "-sS", "--fail", "--max-time", str(timeout), url],
                capture_output=True, check=True,
            )
            time.sleep(polite_delay)
            return proc.stdout
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))

    # curl unavailable or persistently failing — try stdlib urllib once
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        raise ConnectionError(f"http_get failed for {url}: {last_err}") from last_err


def cache_path(name: str) -> Path:
    safe = name.replace("/", "_").replace("=", "_").replace(".", "_")
    return RAW_CACHE_DIR / f"{safe}.csv"


def cache_load(name: str) -> pd.DataFrame | None:
    path = cache_path(name)
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


def cache_save(name: str, df: pd.DataFrame) -> None:
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path(name))
