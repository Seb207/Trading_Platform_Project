"""Phase 0 build pipeline: fetch every active factor, align weekly, persist.

    python3 build_factor_dataset.py            # use raw cache where present
    python3 build_factor_dataset.py --refresh  # force re-download everything

Steps:
  1. Fetch each active factor from its source (FRED / Refinitiv / GPR),
     writing per-series raw CSVs to data/raw/ (the cache).
  2. Align everything to a weekly (Friday) grid; lower-frequency series are
     forward-filled (LOCF) until their next observation.
  3. Persist the combined table to data/factors_weekly.parquet (+ .csv) and
     print a per-factor coverage report.

Per-source failures are reported and skipped, never fatal — a Refinitiv
outage must not block the FRED factors.

Note: values are stamped at observation dates, not publication dates.
Phase 2 validation must apply FactorSpec.pub_lag_days before any
forward-looking test. Raw levels only here — transforms happen in Phase 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from sources import (
    FredClient,
    RefinitivClient,
    get_gpr_index,
    get_occ_open_interest,
    get_putcall_ratio,
)
from factor_schema import active_factors, pending_factors

_MODULE_DIR = Path(__file__).parent.resolve()
OUT_DIR = _MODULE_DIR / "data"
OUT_PARQUET = OUT_DIR / "factors_weekly.parquet"
OUT_CSV = OUT_DIR / "factors_weekly.csv"

REFINITIV_START = "1980-01-01"


def fetch_all(force_refresh: bool = False) -> tuple[dict[str, pd.Series], list[tuple[str, str]]]:
    """Fetch every active factor. Returns ({key: series}, [(key, error), ...])."""
    fred = FredClient()
    collected: dict[str, pd.Series] = {}
    failures: list[tuple[str, str]] = []

    specs = active_factors()
    refinitiv_specs = [s for s in specs if s.source == "refinitiv"]

    for spec in specs:
        if spec.source == "refinitiv":
            continue  # batched below so the session opens at most once
        try:
            if spec.source == "fred":
                series = fred.get_series(spec.source_id, force_refresh=force_refresh)
            elif spec.source == "gpr":
                series = get_gpr_index(force_refresh=force_refresh)
            elif spec.source == "putcall":
                series = get_putcall_ratio(force_refresh=force_refresh)
            elif spec.source == "occ":
                series = get_occ_open_interest(force_refresh=force_refresh)
            else:
                raise ValueError(f"Unknown source: {spec.source}")
            collected[spec.key] = series
            print(f"[OK]   {spec.key:<22} {spec.source:>9}:{spec.source_id:<12} "
                  f"{len(series):>6} obs  {series.index[0].date()} → {series.index[-1].date()}")
        except Exception as exc:
            failures.append((spec.key, str(exc)[:120]))
            print(f"[FAIL] {spec.key:<22} {exc!s:.120}")

    if refinitiv_specs:
        try:
            with RefinitivClient() as rdp:
                for spec in refinitiv_specs:
                    try:
                        series = rdp.get_series(
                            spec.source_id,
                            start=REFINITIV_START,
                            interval=spec.interval or None,
                            field=spec.field or None,
                            force_refresh=force_refresh,
                        )
                        collected[spec.key] = series
                        print(f"[OK]   {spec.key:<22} {spec.source:>9}:{spec.source_id:<12} "
                              f"{len(series):>6} obs  {series.index[0].date()} → {series.index[-1].date()}")
                    except Exception as exc:
                        failures.append((spec.key, str(exc)[:120]))
                        print(f"[FAIL] {spec.key:<22} {exc!s:.120}")
        except Exception as exc:
            for spec in refinitiv_specs:
                failures.append((spec.key, f"session open failed: {exc!s:.100}"))
            print(f"[FAIL] Refinitiv session: {exc!s:.150}")

    return collected, failures


def align_weekly(collected: dict[str, pd.Series]) -> pd.DataFrame:
    """Weekly (Friday) grid; LOCF fills lower-frequency series between releases."""
    weekly = {
        key: series.sort_index().resample("W-FRI").last()
        for key, series in collected.items()
    }
    df = pd.DataFrame(weekly).ffill()
    # Drop the leading years where nothing has been observed yet
    return df.dropna(how="all")


def main() -> None:
    force_refresh = "--refresh" in sys.argv

    print(f"── Fetch ({'refresh' if force_refresh else 'cache-first'}) " + "─" * 40)
    collected, failures = fetch_all(force_refresh=force_refresh)

    if not collected:
        print("Nothing collected — aborting.")
        sys.exit(1)

    print("\n── Align (weekly W-FRI, LOCF) " + "─" * 34)
    df = align_weekly(collected)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET)
    df.to_csv(OUT_CSV)

    print(f"rows={len(df)}  cols={len(df.columns)}  "
          f"{df.index[0].date()} → {df.index[-1].date()}")
    print(f"saved: {OUT_PARQUET.relative_to(_MODULE_DIR)}, {OUT_CSV.relative_to(_MODULE_DIR)}")

    print("\n── Coverage " + "─" * 52)
    for col in df.columns:
        s = df[col].dropna()
        print(f"{col:<22} {s.index[0].date()} → {s.index[-1].date()}  ({len(s)} wk)")

    if failures:
        print("\n── Failures " + "─" * 52)
        for key, err in failures:
            print(f"{key:<22} {err}")

    pend = pending_factors()
    if pend:
        print(f"\n── Pending (no source yet): {', '.join(p.key for p in pend)}")


if __name__ == "__main__":
    main()
