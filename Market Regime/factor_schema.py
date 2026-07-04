"""Declarative factor registry for the Market Regime Detector.

Every factor lives here as one FactorSpec — its source, identifier, theme,
transform hint, and approximate publication lag. Adding a factor to the
system means adding one entry to FACTORS; the build pipeline and (later)
the similarity engine read everything from this table.

Transform hints (applied in Phase 1, NOT during raw collection):
    level      — use the raw value
    yoy        — 12-month percent change
    mom        — 1-month / period-over-period change
    log_level  — log of the raw value (for skewed series like VIX)
    ma4        — 4-period moving average (weekly series like jobless claims)

pub_lag_days is the approximate delay between the observation date stamped
on the series and when the value was actually public. FRED stamps CPI for
January as 01-01, but it is released mid-February — validation (Phase 2)
must shift each series by this lag to avoid look-ahead. Collection ignores it.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FactorSpec:
    key: str               # unique column name in the final dataset
    name: str              # human-readable label
    theme: str             # inflation | growth | rates | risk | liquidity | geopolitical | policy | fx | equity
    source: str            # fred | refinitiv | gpr | putcall | occ | pending
    source_id: str         # FRED series id / Refinitiv RIC / etc.
    transforms: tuple[str, ...] = ("level",)   # hints for Phase 1
    interval: str = ""     # refinitiv only: get_history interval ("monthly", "daily", ...)
    field: str = ""        # refinitiv only: pin the value column (e.g. "MID_PRICE")
    pub_lag_days: int = 0
    notes: str = ""

    @property
    def active(self) -> bool:
        return self.source != "pending"


FACTORS: list[FactorSpec] = [
    # ── Inflation ────────────────────────────────────────────────────
    FactorSpec("cpi_headline", "CPI (headline)", "inflation", "fred", "CPIAUCSL",
               transforms=("yoy",), pub_lag_days=45),
    FactorSpec("cpi_core", "CPI (core)", "inflation", "fred", "CPILFESL",
               transforms=("yoy",), pub_lag_days=45),
    FactorSpec("ppi", "PPI (all commodities)", "inflation", "fred", "PPIACO",
               transforms=("yoy",), pub_lag_days=45),
    FactorSpec("pce_headline", "PCE price index", "inflation", "fred", "PCEPI",
               transforms=("yoy",), pub_lag_days=60),
    FactorSpec("pce_core", "PCE core", "inflation", "fred", "PCEPILFE",
               transforms=("yoy",), pub_lag_days=60),

    # ── Growth ───────────────────────────────────────────────────────
    FactorSpec("ism_pmi", "ISM Manufacturing PMI", "growth", "refinitiv", "USPMI=ECI",
               transforms=("level",), interval="monthly", pub_lag_days=32,
               notes="Confirmed via RDP entitlement check 2026-06-22; not on FRED"),
    FactorSpec("unemployment", "Unemployment rate", "growth", "fred", "UNRATE",
               transforms=("level", "mom"), pub_lag_days=38),
    FactorSpec("jobless_claims", "Initial jobless claims", "growth", "fred", "ICSA",
               transforms=("ma4",), pub_lag_days=5),

    # ── Rates / yield curve ──────────────────────────────────────────
    FactorSpec("ust_2y", "2Y Treasury yield", "rates", "fred", "DGS2",
               transforms=("level", "mom")),
    FactorSpec("ust_10y", "10Y Treasury yield", "rates", "fred", "DGS10",
               transforms=("level", "mom")),
    FactorSpec("ust_30y", "30Y Treasury yield", "rates", "fred", "DGS30",
               transforms=("level",)),
    FactorSpec("spread_2s10s", "2s10s term spread", "rates", "fred", "T10Y2Y",
               transforms=("level",)),
    FactorSpec("spread_3m10y", "3M-10Y term spread", "rates", "fred", "T10Y3M",
               transforms=("level",)),

    # ── Risk appetite / volatility ───────────────────────────────────
    FactorSpec("vix", "VIX", "risk", "fred", "VIXCLS",
               transforms=("log_level",),
               notes="FRED chosen over Refinitiv .VIX — identical data, no session needed"),
    FactorSpec("credit_ig", "IG credit spread (OAS)", "risk", "fred", "BAMLC0A0CM",
               transforms=("level",),
               notes="ICE licensing: FRED serves only a rolling 3-year window "
                     "(verified 2026-07-02) — recent data only; credit_baa_spread "
                     "carries the long history"),
    FactorSpec("credit_hy", "HY credit spread (OAS)", "risk", "fred", "BAMLH0A0HYM2",
               transforms=("level",),
               notes="Same rolling 3-year window as credit_ig"),
    FactorSpec("credit_baa_spread", "Moody's Baa - 10Y Treasury spread", "risk", "fred", "BAA10Y",
               transforms=("level",),
               notes="Long-history credit factor (1986-), unrestricted — "
                     "complements the 3-year-capped ICE OAS series"),
    FactorSpec("spx", "S&P 500 proxy (SPY close)", "equity", "refinitiv", "SPY",
               transforms=("level",), interval="daily",
               notes="Index-level .SPX blocked on this account (TSCC UserNotPermission 92000, "
                     "verified 2026-07-02); SPY ETF used as proxy, history from 1995. "
                     "Input for realized vol / vol-risk-premium and forward-return validation"),

    # ── Liquidity / funding ──────────────────────────────────────────
    FactorSpec("m2", "M2 money supply", "liquidity", "fred", "M2SL",
               transforms=("yoy",), pub_lag_days=28),
    FactorSpec("fed_balance_sheet", "Fed balance sheet (total assets)", "liquidity", "fred", "WALCL",
               transforms=("yoy",), pub_lag_days=6),

    # ── Geopolitical ─────────────────────────────────────────────────
    FactorSpec("gpr", "Geopolitical Risk index", "geopolitical", "gpr", "GPR",
               transforms=("level",), pub_lag_days=30),

    # ── Policy ───────────────────────────────────────────────────────
    FactorSpec("fed_funds", "Effective Fed funds rate", "policy", "fred", "FEDFUNDS",
               transforms=("level", "mom")),

    # ── FX ───────────────────────────────────────────────────────────
    FactorSpec("dollar_index", "Trade-weighted dollar (broad)", "fx", "fred", "DTWEXBGS",
               transforms=("level",),
               notes="Free proxy for the proprietary ICE DXY"),

    # ── Options market (resolved 2026-07-02) ─────────────────────────
    FactorSpec("put_call_ratio", "US options put/call ratio", "risk", "putcall",
               "CBOE archive + OCC",
               transforms=("level",),
               notes="Spliced: CBOE totalpc.csv (2006-11 → 2019-10, frozen) + "
                     "OCC daily-volume-totals API (2019-). CBOE covers CBOE volume "
                     "only vs OCC all-exchange — small level shift at the seam, "
                     "fine after z-scoring. Refinitiv .PCALL / CBOE-OPTTOT-* "
                     "confirmed not permitted on this account"),
    FactorSpec("option_oi", "Total US options open interest (OCC)", "micro", "occ",
               "occTotal",
               transforms=("yoy",), pub_lag_days=1,
               notes="OCC /open-interest API, free, 2018-. OI trends structurally "
                     "upward — use yoy/change, not raw level"),
    FactorSpec("sofr_ois_2y", "2Y SOFR OIS - Treasury swap spread", "liquidity",
               "refinitiv", "USDSR2YOTS=TWEB",
               transforms=("level",), interval="daily", field="MID_PRICE",
               notes="Tradeweb via RDP, bps, 2021-08- (short history — SOFR era "
                     "only). Found via rd.discovery.search; USDSROTS=TWEB (no "
                     "tenor) does not resolve"),

    # ── Pending (source unresolved — see README §Confirmed access) ───
    FactorSpec("breadth", "Market breadth (% above 200dma)", "micro", "pending", "",
               notes="Needs full-universe price pulls — deferred, heavier job"),
    FactorSpec("fomc_tone", "FOMC statement tone (LLM-scored)", "policy", "pending", "",
               notes="Phase 4 — backfillable from federalreserve.gov (1994-)"),
]


def active_factors() -> list[FactorSpec]:
    return [f for f in FACTORS if f.active]


def pending_factors() -> list[FactorSpec]:
    return [f for f in FACTORS if not f.active]


_SPEC_BY_KEY = {f.key: f for f in FACTORS}


def get_spec(key: str) -> FactorSpec:
    try:
        return _SPEC_BY_KEY[key]
    except KeyError:
        raise ValueError(f"Unknown factor key: {key!r}") from None


def resolve_keys(
    factors: list[str] | None = None,
    themes: list[str] | None = None,
) -> list[str]:
    """Resolve a factor/theme selection to a concrete list of active factor keys.

    This is the single place the similarity engine (and anything else) goes
    to turn a user's "just these factors" / "just this theme" request into
    columns — so selecting a subset and adding a new factor both stay
    one-line changes: a new FactorSpec is automatically selectable by key
    and by its theme with no other code changes.

    - factors=None, themes=None → every active factor (the default: use everything)
    - factors=[...]             → exactly those keys (validated)
    - themes=[...]              → every active factor in those themes
    - both given                → union of the two
    Raises ValueError on any unknown key or theme (fail loud, not silent).
    """
    active = active_factors()
    order = [f.key for f in active]           # preserve FACTORS declaration order
    valid_keys = set(order)
    theme_map: dict[str, list[str]] = {}
    for f in active:
        theme_map.setdefault(f.theme, []).append(f.key)

    if factors is None and themes is None:
        return order

    selected: set[str] = set()
    if factors:
        unknown = set(factors) - valid_keys
        if unknown:
            raise ValueError(f"Unknown or inactive factor keys: {sorted(unknown)}")
        selected.update(factors)
    if themes:
        unknown_themes = set(themes) - set(theme_map)
        if unknown_themes:
            raise ValueError(f"Unknown themes: {sorted(unknown_themes)}")
        for t in themes:
            selected.update(theme_map[t])

    return [k for k in order if k in selected]
