"""Phase 1 — similarity engine: find historical analogs of a given week.

Design principles (see README §Locked Design Decisions and §Roadmap):
  - No hardcoded factor lists or correlation groups anywhere in this file.
    Every call resolves its own factor subset (factor_schema.resolve_keys)
    and recomputes its own correlation-based grouping from that subset.
    This is what makes both "pick your own factors" and "add a factor later"
    free: a new FactorSpec just becomes another selectable column; nothing
    here needs to change.
  - Comparison unit is z-scored, transform-appropriate representations
    (yoy/mom/log_level/level/ma4 per FactorSpec.transforms) — never raw
    levels — so a factor's "trend" character is captured, not just its level.
  - Brute-force k-NN, not a vector DB (see README: dataset is ~6k rows x
    a few dozen dims — HNSW would be over-engineering here).
  - Neighbor hygiene is mandatory: excludes a window around the query date
    (serial correlation) and enforces minimum separation between the
    returned analogs (so they aren't 5 consecutive weeks of the same event).

Usage:
    from similarity_engine import find_analogs
    result = find_analogs("2026-06-01", factors=["cpi_headline", "vix"])
    result = find_analogs("2026-06-01", themes=["inflation", "risk"])
    result = find_analogs("2026-06-01")   # every active factor
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from factor_schema import get_spec, resolve_keys

_MODULE_DIR = Path(__file__).parent.resolve()
DEFAULT_DATA_PATH = _MODULE_DIR / "data" / "factors_weekly.parquet"


# ── Transforms (applied to weekly-aligned raw series) ───────────────────
# Data is weekly (W-FRI); "yoy" ~52 weeks back, "mom" ~4 weeks back.

def _t_level(s: pd.Series) -> pd.Series:
    return s


def _t_yoy(s: pd.Series) -> pd.Series:
    return s.pct_change(52) * 100.0


def _t_mom(s: pd.Series) -> pd.Series:
    return s.diff(4)


def _t_log_level(s: pd.Series) -> pd.Series:
    return np.log(s.clip(lower=1e-6))


def _t_ma4(s: pd.Series) -> pd.Series:
    return s.rolling(4, min_periods=1).mean()


TRANSFORM_FUNCS = {
    "level": _t_level,
    "yoy": _t_yoy,
    "mom": _t_mom,
    "log_level": _t_log_level,
    "ma4": _t_ma4,
}


def build_feature_frame(raw: pd.DataFrame, factor_keys: list[str]) -> pd.DataFrame:
    """One or more transform columns per factor key, named '{key}__{transform}'."""
    out: dict[str, pd.Series] = {}
    for key in factor_keys:
        spec = get_spec(key)
        if key not in raw.columns:
            raise ValueError(
                f"Factor {key!r} is in the schema but not in the dataset — "
                "run build_factor_dataset.py, or check it isn't still 'pending'."
            )
        for t in spec.transforms:
            out[f"{key}__{t}"] = TRANSFORM_FUNCS[t](raw[key])
    return pd.DataFrame(out)


# ── Normalization ────────────────────────────────────────────────────────

def expanding_zscore(df: pd.DataFrame, min_periods: int = 52) -> pd.DataFrame:
    """Per-column z-score using only data up to and including each row.

    Expanding (not a fixed trailing window) so early history isn't dropped —
    columns are NaN until min_periods observations have accumulated.
    """
    mean = df.expanding(min_periods=min_periods).mean()
    std = df.expanding(min_periods=min_periods).std()
    return (df - mean) / std.mask(std == 0)


# ── Correlation-driven selective grouping (the locked design decision) ──

def correlated_groups(zdf: pd.DataFrame, threshold: float = 0.7) -> list[list[str]]:
    """Connected components of columns with pairwise |corr| >= threshold.

    Returns every column exactly once, in a group of size 1 if it isn't
    correlated with anything above threshold. Recomputed fresh from
    whatever columns are passed in — never cached across factor subsets.
    """
    cols = list(zdf.columns)
    corr = zdf.corr().abs()
    adjacency: dict[str, set[str]] = {c: set() for c in cols}
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if corr.loc[a, b] >= threshold:
                adjacency[a].add(b)
                adjacency[b].add(a)

    visited: set[str] = set()
    groups: list[list[str]] = []
    for c in cols:
        if c in visited:
            continue
        stack, component = [c], []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            stack.extend(adjacency[node] - visited)
        groups.append(sorted(component))
    return groups


def build_vector_frame(
    zdf: pd.DataFrame, groups: list[list[str]]
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Collapse each correlated group to one averaged column (already z-scored).

    Returns (vector_frame, membership) where membership maps each final
    vector-column name to the original columns it represents — singleton
    groups map to themselves, so `len(members) > 1` identifies a merge.
    """
    vector_cols: dict[str, pd.Series] = {}
    membership: dict[str, list[str]] = {}
    for group in groups:
        name = group[0] if len(group) == 1 else "+".join(group)
        vector_cols[name] = zdf[group].mean(axis=1) if len(group) > 1 else zdf[group[0]]
        membership[name] = group
    return pd.DataFrame(vector_cols), membership


# ── Publication-lag adjustment (Phase 2 walk-forward requirement) ───────

def build_pub_lag_adjusted(raw: pd.DataFrame) -> pd.DataFrame:
    """Shift each factor by its FactorSpec.pub_lag_days (rounded up to whole
    weeks) so a value only becomes visible when it would actually have been
    published — not on its observation date.

    Practical approximation: shifts the already weekly-aligned/LOCF'd column
    forward by ceil(pub_lag_days / 7) weeks. Not applied by default in
    find_analogs (see README's documented Phase 1 limitation); Phase 2
    validation always applies it.
    """
    out = {}
    for col in raw.columns:
        try:
            lag_weeks = math.ceil(get_spec(col).pub_lag_days / 7)
        except ValueError:
            lag_weeks = 0
        out[col] = raw[col].shift(lag_weeks) if lag_weeks > 0 else raw[col]
    return pd.DataFrame(out)


# ── Reusable prepare/search split (what makes Phase 2 backtesting cheap) ─
#
# expanding_zscore is causal per row — a row's z-score depends only on data
# up to and including that row, regardless of how many extra rows trail it
# in the same call. So the z-score frame can be computed ONCE across all
# history and safely reused for every query date in a walk-forward backtest;
# only the correlation grouping (which looks at a WINDOW of rows) must be
# recomputed after slicing to date <= as_of for each query, which
# find_analogs_from_zscore does before anything else touches the data.

def prepare_zscore_frame(
    factors: list[str] | None = None,
    themes: list[str] | None = None,
    data_path: str | Path | None = None,
    min_history_weeks: int = 52,
    apply_pub_lag: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Load + transform + expanding-z-score once. Returns (z_full, resolved_keys)."""
    keys = resolve_keys(factors, themes)
    raw = pd.read_parquet(data_path or DEFAULT_DATA_PATH)
    if apply_pub_lag:
        raw = build_pub_lag_adjusted(raw)

    features = build_feature_frame(raw, keys)
    z = expanding_zscore(features, min_periods=min_history_weeks).dropna(how="any")
    return z, keys


def find_analogs_from_zscore(
    z_full: pd.DataFrame,
    as_of: str | pd.Timestamp,
    k: int = 5,
    exclude_weeks: int = 26,
    min_separation_weeks: int = 26,
    corr_threshold: float = 0.7,
) -> dict:
    """Core analog search on an already-prepared z-score frame.

    Walk-forward safe: slices to rows <= as_of BEFORE computing correlation
    groups or the candidate pool, so nothing after the query date can ever
    enter the result.
    """
    as_of_ts = pd.Timestamp(as_of)
    z = z_full[z_full.index <= as_of_ts]
    if z.empty:
        raise ValueError(f"No data available at or before {as_of}")
    query_date = z.index[-1]

    groups = correlated_groups(z, threshold=corr_threshold)
    vec, membership = build_vector_frame(z, groups)
    query_vec = vec.loc[query_date]
    query_z = z.loc[query_date]   # pre-merge z-scores, for the member breakdown below

    excl_start = query_date - pd.Timedelta(weeks=exclude_weeks)
    excl_end = query_date + pd.Timedelta(weeks=exclude_weeks)
    candidates = vec[(vec.index < excl_start) | (vec.index > excl_end)]
    if candidates.empty:
        raise ValueError("No candidates remain outside the exclusion window.")

    diff_sq = (candidates - query_vec) ** 2
    dist_sq = diff_sq.sum(axis=1)
    dist = dist_sq.pow(0.5)
    ranked = dist.sort_values()

    selected: list[pd.Timestamp] = []
    min_sep = pd.Timedelta(weeks=min_separation_weeks)
    for date in ranked.index:
        if all(abs(date - d) >= min_sep for d in selected):
            selected.append(date)
        if len(selected) >= k:
            break

    merged_groups = {name: members for name, members in membership.items() if len(members) > 1}

    analogs = []
    for date in selected:
        total = dist_sq.loc[date]
        contributions = (
            (diff_sq.loc[date] / total).sort_values(ascending=False).to_dict()
            if total > 0 else {}
        )

        # Member breakdown for merged dimensions: each member's own pre-merge
        # squared z-score difference, expressed as a share WITHIN that group.
        # This is informational, not a decomposition of the group's own
        # contribution above — the group's score is based on the AVERAGED
        # z-score (build_vector_frame), which doesn't decompose linearly into
        # its members' individual squared differences. It answers "which
        # member moved the most," not "how much did each contribute to the
        # group's official percentage."
        analog_z = z.loc[date]
        member_contributions: dict[str, dict[str, float]] = {}
        for name, members in merged_groups.items():
            member_diff_sq = {m: float((query_z[m] - analog_z[m]) ** 2) for m in members}
            member_total = sum(member_diff_sq.values())
            member_contributions[name] = (
                {m: v / member_total for m, v in member_diff_sq.items()}
                if member_total > 0 else {m: 1.0 / len(members) for m in members}
            )

        analogs.append({
            "date": date.date().isoformat(),
            "distance": float(dist.loc[date]),
            "contributions": {c: float(v) for c, v in contributions.items()},
            "member_contributions": member_contributions,
        })

    return {
        "query_date": query_date.date().isoformat(),
        "vector_dimensions": list(vec.columns),
        "groups": merged_groups,
        "analogs": analogs,
    }


# ── Main entry point ─────────────────────────────────────────────────────

def find_analogs(
    as_of: str,
    factors: list[str] | None = None,
    themes: list[str] | None = None,
    k: int = 5,
    exclude_weeks: int = 26,
    min_separation_weeks: int = 26,
    corr_threshold: float = 0.7,
    min_history_weeks: int = 52,
    data_path: str | Path | None = None,
) -> dict:
    """Find the k most similar historical weeks to `as_of`.

    factors / themes select which columns participate (see
    factor_schema.resolve_keys — both None means every active factor).
    Everything downstream (z-scoring, correlation grouping, distances,
    per-feature contributions) is recomputed fresh for that selection.

    Does NOT apply publication-lag adjustment — see README's documented
    Phase 1 limitation. Phase 2's validation (validation.py) always does.

    Returns a dict:
        query_date        — the actual dataset date used (snapped to <= as_of)
        factors_used       — resolved factor keys
        vector_dimensions   — final vector columns after grouping
        groups              — {merged_name: [original columns]} for any
                              correlated groups that got combined
        analogs             — [{date, distance, contributions,
                              member_contributions}, ...]. contributions are
                              each vector dimension's share of the squared
                              distance (sums to 1 per analog).
                              member_contributions[merged_dim] breaks a
                              merged dimension down into its original factors'
                              individual shares (sums to 1 within that group;
                              informational only, not a decomposition of the
                              group's own contribution — see
                              find_analogs_from_zscore's inline comment)
    """
    z_full, keys = prepare_zscore_frame(
        factors, themes, data_path=data_path,
        min_history_weeks=min_history_weeks, apply_pub_lag=False,
    )
    if z_full.empty:
        raise ValueError(
            "No fully-covered history for this factor selection — "
            "check for a very recently added factor with a short backfill."
        )
    result = find_analogs_from_zscore(
        z_full, as_of, k=k, exclude_weeks=exclude_weeks,
        min_separation_weeks=min_separation_weeks, corr_threshold=corr_threshold,
    )
    result["factors_used"] = keys
    return result


def _deep_history_factors(before: str = "1996-01-01", data_path: str | Path | None = None) -> list[str]:
    """Active factors whose data starts before `before` — derived from the
    dataset itself, not a hardcoded name list, so it stays correct as
    factors are added or their backfills deepen."""
    from factor_schema import active_factors
    raw = pd.read_parquet(data_path or DEFAULT_DATA_PATH)
    cutoff = pd.Timestamp(before)
    return [f.key for f in active_factors() if raw[f.key].dropna().index[0] <= cutoff]


if __name__ == "__main__":
    # Roadmap smoke test: known crisis dates should surface intuitive analogs.
    #
    # NOTE: using every active factor (factors=None) would fail here on
    # purpose — dropna(how="any") requires ALL selected factors to have data,
    # and several (option_oi, credit_ig/hy, sofr_ois_2y, dollar_index,
    # put_call_ratio, fed_balance_sheet) only start 2002-2024. Querying 2008
    # against the full set leaves too little pre-query history once the
    # exclusion window + min_history_weeks are applied. This is exactly the
    # scenario "pick your own factors" solves: use only factors with deep
    # enough history for the date being queried.
    long_history_factors = _deep_history_factors(before="1996-01-01")

    for date in ["2008-09-15", "2020-03-15", "2022-06-15"]:
        try:
            result = find_analogs(date, factors=long_history_factors, k=5)
        except Exception as exc:
            print(f"FAILED for {date}: {exc}")
            continue

        print(f"\n=== Query {date} (snapped to {result['query_date']}) "
              f"— {len(result['vector_dimensions'])} dims ===")
        if result["groups"]:
            print("Correlated groups merged:")
            for name in result["groups"]:
                print(f"  {name}")
        for a in result["analogs"]:
            top = list(a["contributions"].items())[:3]
            print(f"  {a['date']}  dist={a['distance']:.2f}  top factors: {top}")
