---
name: add-market-regime-factor
description: Procedure for adding a new macro/market factor to the Market Regime Detector — new FactorSpec entry, data source client, dataset rebuild, and README update. Use when the user asks to track a new economic indicator, market factor, or data series in Market Regime/.
---

# Add Market Regime Factor

## Why this exists

Every factor in Market Regime follows the same pipeline
(source client → `FactorSpec` → dataset build → similarity engine input), and
skipping a step leaves the factor silently absent from search results or
breaks the dataset build for everyone else. See
`Market Regime/README.md` §2 for the full factor list and source rationale,
and `CLAUDE.md` §1 for how this module bridges into the dashboard.

## Steps

1. **Pick or build a source client** in `Market Regime/sources/`. Existing
   clients: `fred.py`, `refinitiv_rdp.py` (with desktop-session fallback),
   `gpr.py`, `occ.py`, `cboe.py`. Reuse an existing client if the new factor
   comes from an already-integrated source; otherwise follow the same shape
   (fetch → return a raw time series) as a new file.
   - If fetching over HTTP directly (not via a vendor SDK), use
     `Market Regime/sources/common.py`'s `http_get()` — it shells out to curl
     with curl's **default** User-Agent. A custom UA gets blocked by FRED's
     CDN (verified); don't add a custom `-A` flag when adapting this for a
     new source without checking first.
2. **Add a `FactorSpec` entry** to `FACTORS` in
   `Market Regime/factor_schema.py`: `key`, `name`, `theme`, `source`,
   `source_id`, `transforms`, `interval`, `field`, `pub_lag_days`, `notes`.
   - `pub_lag_days` matters — it feeds `build_pub_lag_adjusted()` in
     `similarity_engine.py` to prevent look-ahead bias in analog search.
     Get this right from the release calendar of the actual source, not a
     guess.
   - If the factor isn't ready for production use yet (e.g. data quality
     unverified), add it to the pending list instead of `active_factors()`.
3. **Rebuild the dataset**:
   ```bash
   cd "Market Regime"
   python3 build_factor_dataset.py
   ```
   This writes `data/factors_weekly.parquet`. Note the pyarrow version
   coupling in `CLAUDE.md` §2 — if this write happens with a newer pyarrow
   than what's in `backend/requirements.txt`, bump the floor there too, or
   the dashboard's read will break.
4. **Verify via the bridge**, not just locally — restart the backend (see
   `restart-backend`) and confirm the new factor appears in
   `GET /api/regime/factors`.
5. **Update `Market Regime/README.md` §2** (factor list/data sources) with
   the new entry — this doc is meant to stay in sync with `factor_schema.py`
   as the human-readable mirror of it.

## Common pitfall

Don't add the `FactorSpec` without rebuilding the dataset — the similarity
engine reads from the parquet file, not live from the source client, so a
factor can be fully wired in code and still be invisible to `find_analogs()`
until the dataset is regenerated.
