"""Phase 2 — validation: does analog-conditioning carry statistical content?

Question: when find_analogs() surfaces "similar" historical periods, do their
ACTUAL subsequent market outcomes differ from what would have happened
picking any random period? If not, the tool is a plausible-looking toy, not
a signal (see README §Risk #1).

Method: run find_analogs across many historical query dates (a walk-forward
backtest, not a single point check), pool the forward S&P (SPY proxy)
returns of the top-k analogs at several horizons, and compare that pooled
distribution against the unconditional forward-return distribution (same
universe of dates, same horizons) with a two-sample Kolmogorov-Smirnov test.
A significant KS result means analog-conditioning shifts the outcome
distribution — the tool is finding something, not noise.

Walk-forward integrity (both required, not optional):
  - Publication-lag adjustment (build_pub_lag_adjusted) is always applied —
    a factor is only "seen" once it would actually have been public.
  - find_analogs_from_zscore slices to date <= query BEFORE computing
    correlation groups or searching candidates — nothing after the query
    date can influence its own analog set.

Usage:
    python3 validation.py                      # default factor set, all horizons
    from validation import run_validation
    report = run_validation(themes=["inflation", "risk"], k=5)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from similarity_engine import (
    DEFAULT_DATA_PATH,
    find_analogs_from_zscore,
    prepare_zscore_frame,
)

DEFAULT_HORIZONS_WEEKS = {"1m": 4, "3m": 13, "6m": 26, "12m": 52}


def compute_forward_returns(
    price: pd.Series, horizons_weeks: dict[str, int]
) -> pd.DataFrame:
    """Forward simple return starting at each date, one column per horizon.
    NaN at the tail where there isn't enough future data yet."""
    return pd.DataFrame({
        name: price.shift(-weeks) / price - 1.0
        for name, weeks in horizons_weeks.items()
    })


def run_validation(
    factors: list[str] | None = None,
    themes: list[str] | None = None,
    k: int = 5,
    exclude_weeks: int = 26,
    min_separation_weeks: int = 26,
    corr_threshold: float = 0.7,
    min_history_weeks: int = 52,
    query_interval_weeks: int = 4,
    horizons_weeks: dict[str, int] | None = None,
    price_factor: str = "spx",
    data_path: str | Path | None = None,
) -> dict:
    """Walk-forward validation of find_analogs against forward market returns.

    Samples one query date every `query_interval_weeks` (a full weekly
    backtest is unnecessary and expensive — this still yields hundreds of
    largely-independent query points) across all dates where the selected
    factors have enough history (min_history_weeks) and there is at least
    one horizon's worth of forward return data remaining.

    Returns:
        factors_used, n_query_dates, k, horizons: {
            horizon_name: {n_conditioned, n_conditioned_unique,
                           n_unconditional, conditioned_mean,
                           conditioned_unique_mean, unconditional_mean,
                           conditioned_std, unconditional_std,
                           ks_statistic, ks_pvalue, significant_at_5pct,
                           ks_statistic_unique, ks_pvalue_unique,
                           significant_at_5pct_unique}
        }
    Trust the *_unique fields — see the pseudo-replication note above
    conditioned_unique_dates below.
    """
    horizons_weeks = horizons_weeks or DEFAULT_HORIZONS_WEEKS
    data_path = data_path or DEFAULT_DATA_PATH

    raw = pd.read_parquet(data_path)
    if price_factor not in raw.columns:
        raise ValueError(f"price_factor {price_factor!r} not in dataset")
    fwd = compute_forward_returns(raw[price_factor], horizons_weeks)

    z_full, keys = prepare_zscore_frame(
        factors, themes, data_path=data_path,
        min_history_weeks=min_history_weeks, apply_pub_lag=True,
    )
    if z_full.empty:
        raise ValueError(
            "No fully-covered history for this factor selection (after "
            "publication-lag shifting) — try a smaller/deeper-history subset."
        )

    result = _run_validation_core(
        z_full, fwd, k=k, exclude_weeks=exclude_weeks,
        min_separation_weeks=min_separation_weeks, corr_threshold=corr_threshold,
        query_interval_weeks=query_interval_weeks, horizons_weeks=horizons_weeks,
    )
    result["factors_used"] = keys
    return result


def _run_validation_core(
    z_full: pd.DataFrame,
    fwd: pd.DataFrame,
    k: int,
    exclude_weeks: int,
    min_separation_weeks: int,
    corr_threshold: float,
    query_interval_weeks: int,
    horizons_weeks: dict[str, int],
) -> dict:
    """The actual backtest loop, factored out so sensitivity_sweep can reuse
    the same (expensive-to-build) z_full/fwd across many k/window settings."""
    max_h = max(horizons_weeks.values())
    fwd_valid_until = fwd.dropna(how="all").index[-1] - pd.Timedelta(weeks=max_h)
    eligible = z_full.index[z_full.index <= fwd_valid_until]
    query_dates = eligible[::query_interval_weeks]

    # Pool ALL (query, analog) pairs — "conditioned" — and separately track
    # the SET of unique analog dates ever selected — "conditioned_unique".
    # The pooled version overstates the effective sample size: adjacent
    # query dates (4 weeks apart) often re-select the same analog, so the
    # same historical return gets counted many times, violating the KS
    # test's i.i.d. assumption and inflating apparent significance. The
    # unique-date version is the conservative, honest robustness check —
    # report both, trust "unique" more.
    conditioned: dict[str, list[float]] = {h: [] for h in horizons_weeks}
    unique_analog_dates: set[pd.Timestamp] = set()
    n_used = 0
    for date in query_dates:
        try:
            result = find_analogs_from_zscore(
                z_full, date, k=k, exclude_weeks=exclude_weeks,
                min_separation_weeks=min_separation_weeks,
                corr_threshold=corr_threshold,
            )
        except ValueError:
            continue
        n_used += 1
        for analog in result["analogs"]:
            analog_date = pd.Timestamp(analog["date"])
            if analog_date not in fwd.index:
                continue
            unique_analog_dates.add(analog_date)
            for h_name in horizons_weeks:
                r = fwd.loc[analog_date, h_name]
                if pd.notna(r):
                    conditioned[h_name].append(r)

    conditioned_unique: dict[str, list[float]] = {h: [] for h in horizons_weeks}
    for analog_date in unique_analog_dates:
        for h_name in horizons_weeks:
            r = fwd.loc[analog_date, h_name]
            if pd.notna(r):
                conditioned_unique[h_name].append(r)

    report_horizons = {}
    for h_name in horizons_weeks:
        cond = np.array(conditioned[h_name])
        cond_unique = np.array(conditioned_unique[h_name])
        uncond = fwd[h_name].dropna().values
        if len(cond) < 5:
            report_horizons[h_name] = {
                "n_conditioned": len(cond),
                "note": "too few analog observations for a meaningful test",
            }
            continue

        stat, pvalue = ks_2samp(cond, uncond)
        stat_u, pvalue_u = ks_2samp(cond_unique, uncond)
        report_horizons[h_name] = {
            "n_conditioned": len(cond),
            "n_conditioned_unique": len(cond_unique),
            "n_unconditional": len(uncond),
            "conditioned_mean": float(np.mean(cond)),
            "conditioned_unique_mean": float(np.mean(cond_unique)),
            "unconditional_mean": float(np.mean(uncond)),
            "conditioned_std": float(np.std(cond)),
            "unconditional_std": float(np.std(uncond)),
            "ks_statistic": float(stat),
            "ks_pvalue": float(pvalue),
            "significant_at_5pct": bool(pvalue < 0.05),
            "ks_statistic_unique": float(stat_u),
            "ks_pvalue_unique": float(pvalue_u),
            "significant_at_5pct_unique": bool(pvalue_u < 0.05),
        }

    return {
        "n_query_dates": n_used,
        "k": k,
        "query_interval_weeks": query_interval_weeks,
        "horizons": report_horizons,
    }


def sensitivity_sweep(
    factors: list[str] | None = None,
    themes: list[str] | None = None,
    k_values: tuple[int, ...] = (3, 5, 10, 15),
    exclude_weeks_values: tuple[int, ...] = (13, 26, 52),
    min_separation_weeks: int = 26,
    corr_threshold: float = 0.7,
    min_history_weeks: int = 52,
    query_interval_weeks: int = 4,
    horizon: str = "3m",
    price_factor: str = "spx",
    data_path: str | Path | None = None,
) -> pd.DataFrame:
    """Grid sweep over k and exclude_weeks — is "no significant signal" a
    property of this one configuration, or does it hold across settings?

    z_full/fwd are built ONCE and reused across every grid cell (they don't
    depend on k or exclude_weeks), so the sweep only re-pays the backtest
    loop's cost, not the data-prep cost. Reports the unique-date KS result
    only — the pooled one is the pseudo-replication-inflated version this
    sweep exists to not be fooled by.
    """
    horizons_weeks = {horizon: DEFAULT_HORIZONS_WEEKS[horizon]}
    data_path = data_path or DEFAULT_DATA_PATH

    raw = pd.read_parquet(data_path)
    fwd = compute_forward_returns(raw[price_factor], horizons_weeks)
    z_full, keys = prepare_zscore_frame(
        factors, themes, data_path=data_path,
        min_history_weeks=min_history_weeks, apply_pub_lag=True,
    )

    rows = []
    for k in k_values:
        for exclude_weeks in exclude_weeks_values:
            result = _run_validation_core(
                z_full, fwd, k=k, exclude_weeks=exclude_weeks,
                min_separation_weeks=min_separation_weeks,
                corr_threshold=corr_threshold,
                query_interval_weeks=query_interval_weeks,
                horizons_weeks=horizons_weeks,
            )
            h = result["horizons"].get(horizon, {})
            rows.append({
                "k": k,
                "exclude_weeks": exclude_weeks,
                "n_query_dates": result["n_query_dates"],
                "n_unique_analogs": h.get("n_conditioned_unique"),
                "ks_pvalue_unique": h.get("ks_pvalue_unique"),
                "significant": h.get("significant_at_5pct_unique"),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from similarity_engine import _deep_history_factors

    deep_factors = _deep_history_factors(before="1996-01-01")
    report = run_validation(factors=deep_factors, query_interval_weeks=4)

    print(f"Factors used ({len(report['factors_used'])}): {report['factors_used']}")
    print(f"Query dates evaluated: {report['n_query_dates']}  (k={report['k']})\n")

    print(f"{'':>4}      {'pooled (n reused across queries)':<55} {'unique analog dates only':<40}")
    for h_name, stats in report["horizons"].items():
        if "note" in stats:
            print(f"{h_name:>4}: {stats['note']} (n={stats['n_conditioned']})")
            continue
        sig = "***" if stats["significant_at_5pct"] else "   "
        sig_u = "***" if stats["significant_at_5pct_unique"] else "   "
        print(
            f"{h_name:>4} {sig} n={stats['n_conditioned']:>5} "
            f"cond={stats['conditioned_mean']:+.3%} uncond={stats['unconditional_mean']:+.3%} "
            f"KS={stats['ks_statistic']:.3f} p={stats['ks_pvalue']:.4f}   |   "
            f"{sig_u} n={stats['n_conditioned_unique']:>4} "
            f"cond={stats['conditioned_unique_mean']:+.3%} "
            f"KS={stats['ks_statistic_unique']:.3f} p={stats['ks_pvalue_unique']:.4f}"
        )

    print("\n── Sensitivity sweep (3m horizon, unique-date KS test) " + "─" * 20)
    sweep = sensitivity_sweep(factors=deep_factors, query_interval_weeks=4)
    print(sweep.to_string(index=False))
    print(f"\nSignificant cells: {sweep['significant'].sum()} / {len(sweep)}")
