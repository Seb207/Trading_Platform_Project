"""
Market Regime endpoints:
  GET  /api/regime/factors   — selectable factors (key, name, theme, coverage)
  POST /api/regime/analogs   — find_analogs() — core retrieval, always-on
  POST /api/regime/validate  — run_validation() / sensitivity_sweep() — opt-in

The tool's scope is context retrieval, not prediction (see Market Regime/
README.md §Goal): /analogs is what the dashboard calls automatically on every
query. /validate is a statistical check the frontend only calls when the
user explicitly asks "how robust is this?" — never called automatically.
"""
import math
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.modules.regime.regime_bridge import get_regime_modules

router = APIRouter(prefix="/api/regime", tags=["regime"])


# ── Request schemas ──────────────────────────────────────────────────────

class AnalogsRequest(BaseModel):
    as_of: str
    factors: list[str] | None = None
    themes: list[str] | None = None
    k: int = 5
    exclude_weeks: int = 26
    min_separation_weeks: int = 26
    corr_threshold: float = 0.7


class ValidateRequest(BaseModel):
    factors: list[str] | None = None
    themes: list[str] | None = None
    k: int = 5
    exclude_weeks: int = 26
    min_separation_weeks: int = 26
    corr_threshold: float = 0.7
    query_interval_weeks: int = 4
    mode: str = "validate"   # "validate" | "sweep"


# ── Helpers (presentation-shaping — kept out of similarity_engine.py, ────
# which stays a clean research module, not dashboard-specific) ──────────

def _clean(value):
    """NaN/None-safe scalar for JSON — FastAPI's default encoder chokes on NaN."""
    import pandas as pd
    if value is None or pd.isna(value):
        return None
    return float(value)


def _event_series(price, center, lookback_weeks: int, forward_weeks: int):
    """Event-time-aligned % return series (0 at `center`) for the overlay chart."""
    import pandas as pd
    start = center - pd.Timedelta(weeks=lookback_weeks)
    end = center + pd.Timedelta(weeks=forward_weeks)
    window = price[(price.index >= start) & (price.index <= end)].dropna()
    if center not in window.index:
        return []
    base = window.loc[center]
    return [
        {
            "offset_weeks": round((idx - center).days / 7),
            "return_pct": _clean((v / base - 1.0) * 100),
        }
        for idx, v in window.items()
    ]


def _forward_returns(price, date, horizons_weeks: dict[str, int]):
    """Forward return at each horizon from `date`, None where data is missing."""
    import pandas as pd
    if date not in price.index or pd.isna(price.loc[date]):
        return {h: None for h in horizons_weeks}
    base = price.loc[date]
    out = {}
    for h_name, h_weeks in horizons_weeks.items():
        target = date + pd.Timedelta(weeks=h_weeks)
        out[h_name] = (
            _clean(price.loc[target] / base - 1.0)
            if target in price.index else None
        )
    return out


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.get("/factors")
def list_factors():
    """Every active factor (for the UI's factor/theme picker) with its
    actual data coverage from the built dataset."""
    _, _, factor_schema = get_regime_modules()
    import pandas as pd
    from similarity_engine import DEFAULT_DATA_PATH

    raw = pd.read_parquet(DEFAULT_DATA_PATH)
    factors = []
    for f in factor_schema.active_factors():
        s = raw[f.key].dropna() if f.key in raw.columns else None
        factors.append({
            "key": f.key,
            "name": f.name,
            "theme": f.theme,
            "start": s.index[0].date().isoformat() if s is not None and len(s) else None,
            "end": s.index[-1].date().isoformat() if s is not None and len(s) else None,
        })
    themes = sorted({f["theme"] for f in factors})
    return {"factors": factors, "themes": themes}


@router.post("/analogs")
def get_analogs(req: AnalogsRequest):
    """Core retrieval — find_analogs() enriched with event-time price paths
    and forward returns so the frontend can render the overlay + fan chart
    without a second round trip."""
    similarity_engine, validation, _ = get_regime_modules()
    import pandas as pd

    try:
        result = similarity_engine.find_analogs(
            req.as_of,
            factors=req.factors,
            themes=req.themes,
            k=req.k,
            exclude_weeks=req.exclude_weeks,
            min_separation_weeks=req.min_separation_weeks,
            corr_threshold=req.corr_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    raw = pd.read_parquet(similarity_engine.DEFAULT_DATA_PATH)
    price = raw["spx"]
    horizons = validation.DEFAULT_HORIZONS_WEEKS

    # Fetch a generously wide window (~2 years each side) so the frontend can
    # zoom/pan the event-time overlay entirely client-side without a second
    # round trip. The chart's own default view is still narrower than this —
    # see EventTimeOverlay.tsx's initial domain.
    lookback_weeks = 104
    forward_weeks = 104

    query_ts = pd.Timestamp(result["query_date"])
    result["query_event_series"] = _event_series(price, query_ts, lookback_weeks, 0)

    for analog in result["analogs"]:
        analog_ts = pd.Timestamp(analog["date"])
        analog["event_series"] = _event_series(price, analog_ts, lookback_weeks, forward_weeks)
        analog["forward_returns"] = _forward_returns(price, analog_ts, horizons)

    return result


@router.post("/validate")
def validate(req: ValidateRequest):
    """Opt-in statistical check (KS test or k/window sensitivity sweep).
    Only ever called when the user explicitly asks for it — never on page load."""
    _, validation, _ = get_regime_modules()

    try:
        if req.mode == "sweep":
            df = validation.sensitivity_sweep(
                factors=req.factors,
                themes=req.themes,
                query_interval_weeks=req.query_interval_weeks,
            )
            rows = [
                {k: (None if isinstance(v, float) and math.isnan(v) else v)
                 for k, v in row.items()}
                for row in df.to_dict("records")
            ]
            return {"mode": "sweep", "rows": rows}

        result = validation.run_validation(
            factors=req.factors,
            themes=req.themes,
            k=req.k,
            exclude_weeks=req.exclude_weeks,
            min_separation_weeks=req.min_separation_weeks,
            corr_threshold=req.corr_threshold,
            query_interval_weeks=req.query_interval_weeks,
        )
        return {"mode": "validate", **result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
