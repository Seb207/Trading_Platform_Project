"""Refinitiv data source — RDP (Refinitiv Data Platform) API via refinitiv.data.

IMPORTANT: this session only authenticates while the local Refinitiv platform
(Workspace) is running on this machine. Start Refinitiv Workspace BEFORE
running any collection that touches this source; otherwise the session open
fails and Refinitiv-sourced factors are skipped by the build pipeline.

Credentials come from refinitiv-data.config.json (session "platform.rdp").
The session opens lazily on first use — importing this module never triggers
a network login.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import cache_load, cache_save

_MODULE_DIR = Path(__file__).parent.parent.resolve()


class RefinitivClient:
    """Lazy-open RDP session wrapper. Use as a context manager:

        with RefinitivClient() as rdp:
            pmi = rdp.get_series("USPMI=ECI", start="1990-01-01")
    """

    DEFAULT_CONFIG = str(
        _MODULE_DIR.parent / "Quant Models" / "refinitiv-data.config.json"
    )

    # Column preference when a specific field isn't requested: economic
    # indicator RICs return VALUE, price RICs return TRDPRC_1 / CLOSE.
    _VALUE_COLUMNS = ["VALUE", "TRDPRC_1", "TR.PriceClose", "CLOSE", "MID_PRICE"]

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path or self.DEFAULT_CONFIG
        self._session = None

    def _ensure_session(self):
        """Open a session: platform.rdp first, desktop-session fallback.

        The desktop fallback matches the pattern used in the earlier data
        work (Quant Models/Irrationality_Index/Factor_Management.py) —
        rd.session.desktop with the same app key, which authenticates
        through the locally running Refinitiv Workspace.
        """
        if self._session is None:
            import refinitiv.data as rd
            self._rd = rd
            try:
                self._session = rd.open_session(config_name=self.config_path)
            except Exception as platform_exc:
                try:
                    import json
                    with open(self.config_path, encoding="utf-8") as f:
                        cfg = json.load(f)
                    app_key = cfg["sessions"]["platform"]["rdp"]["app-key"]
                    session = rd.session.desktop.Definition(app_key=app_key).get_session()
                    rd.session.set_default(session)
                    session.open()
                    self._session = session
                except Exception as desktop_exc:
                    raise ConnectionError(
                        "Refinitiv session failed on both paths. "
                        f"platform.rdp: {platform_exc} | "
                        f"desktop fallback: {desktop_exc} — "
                        "is Refinitiv Workspace running locally?"
                    ) from desktop_exc
        return self._session

    def get_history(
        self,
        ric: str,
        start: str = "1980-01-01",
        end: str | None = None,
        interval: str | None = None,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """Raw get_history passthrough returning the full DataFrame.

        end and interval are ALWAYS sent: without an explicit end,
        rd.get_history silently caps the response at ~20 rows regardless
        of start (observed live: USPMI=ECI start=1980 → 20 obs from 2024).
        """
        self._ensure_session()
        from datetime import datetime
        kwargs = {
            "start":    start,
            "end":      end or datetime.now().strftime("%Y-%m-%d"),
            "interval": interval or "daily",
        }
        if fields:
            kwargs["fields"] = fields
        return self._rd.get_history(ric, **kwargs)

    def get_series(
        self,
        ric: str,
        start: str = "1980-01-01",
        end: str | None = None,
        interval: str | None = None,
        field: str | None = None,
        force_refresh: bool = False,
    ) -> pd.Series:
        """One RIC → one float Series, picking a sensible value column.

        field pins the column explicitly; otherwise the first match from
        _VALUE_COLUMNS is used, else the first numeric column.
        """
        cache_name = f"rdp_{ric}"
        if not force_refresh:
            cached = cache_load(cache_name)
            if cached is not None:
                return cached.iloc[:, 0].astype(float)

        df = self.get_history(ric, start=start, end=end, interval=interval)
        if df is None or len(df) == 0:
            raise ValueError(f"Refinitiv returned no data for {ric!r}")

        candidates = [field] if field else self._VALUE_COLUMNS
        col = next((c for c in candidates if c in df.columns), None)
        if col is None:
            numeric = df.select_dtypes("number")
            if numeric.empty:
                raise ValueError(
                    f"No usable value column for {ric!r}; columns={list(df.columns)[:10]}"
                )
            col = numeric.columns[0]

        series = pd.to_numeric(df[col], errors="coerce").dropna()
        series.index = pd.to_datetime(series.index)
        series.name = ric

        cache_save(cache_name, series.to_frame())
        return series

    def close(self) -> None:
        if self._session is not None:
            try:
                self._rd.close_session()
            finally:
                self._session = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
