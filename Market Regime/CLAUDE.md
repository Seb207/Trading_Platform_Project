# Market Regime — Claude Working Notes

Module-local notes for `Market Regime/`. Cross-module rules (bridge pattern,
conda/PATH gotcha, dependency sync, verification discipline) live in the
repo-root `../CLAUDE.md` — read that first. This file only holds facts
specific to this module. Same maintenance protocol applies: update this file
when a session surfaces something new and generalizable, don't let it go
stale.

## What this module is

A standalone Python analysis engine — no server of its own. Computes weekly
macro/market factor datasets and runs similarity search over them to find
historical analogs to a given date. Bridged into
`Consulting Dashboard/backend/modules/regime/regime_bridge.py`, which
lazy-imports `similarity_engine.py`, `validation.py`, and `factor_schema.py`
directly from this directory. Full design rationale, methodology, and
factor/visualization inventory live in `README.md` in this directory — this
file is only for Claude-specific operational notes that don't belong in a
design doc.

## Design Precedent: retrieval, not prediction

The Market Regime Detector's locked scope decision (confirmed 2026-07-02) is
the reference for any future analysis feature in this project: the tool
retrieves historically similar market contexts and stops there — it does
**not** attempt to predict or validate outcomes as a blocking step.
Statistical validation (`validation.py`, the Validation panel in the
dashboard) is offered as an **opt-in** feature the user can invoke on demand,
not a gate the core retrieval feature must pass.

**Why**: the intended use is "pull up historically similar context in one
shot; interpretation is the user's call" — baking in a mandatory statistical
gate would turn a context-retrieval tool into a claimed-predictive one, which
overstates what it does. Default to this shape (retrieve/present first,
validate on demand) for future features here unless the user explicitly asks
for a predictive/validating tool instead.

## pyarrow version coupling

`build_factor_dataset.py` writes `data/factors_weekly.parquet`. Whatever
pyarrow version does that write becomes a **floor** for every reader:
`Consulting Dashboard/backend/requirements.txt` pins `pandas`/`pyarrow`
versions that must be at least as new, or reading raises `OSError:
Repetition level histogram size mismatch`. **If you rebuild the dataset with
a newer pyarrow, bump the floor in `backend/requirements.txt` to match** —
this is the kind of cross-repo edit that's easy to forget because the two
files are in different directories with no dependency link between them.

## Adding a new factor

Don't hand-edit `factor_schema.py` and stop there — follow the full
`add-market-regime-factor` skill (`../.claude/skills/add-market-regime-factor/`),
which covers the source client → `FactorSpec` → dataset rebuild → README
update sequence. Skipping the dataset rebuild step is the most common way
to end up with a factor that's "wired in code" but invisible to
`find_analogs()`.

## Data source quirks

- **FRED**: fetched via `sources/common.py`'s `http_get()`, which shells out
  to curl using curl's **default** User-Agent. FRED's CDN blocks both
  `requests`/`urllib` and curl with a custom `-A` UA string — verified
  directly. Don't add a custom UA when touching this.
- **Refinitiv (`refinitiv_rdp.py`)**: `rd.get_history` caps at 20 rows unless
  `end` and `interval` are passed explicitly — always pass both. Has a
  desktop-session fallback for when the RDP Platform session isn't
  available.
- **`.SPX` is not permitted** under the current entitlement — use the SPY
  ETF as a proxy instead.
- **Credit spreads**: ICE data on FRED only covers ~3yr of history — Moody's
  BAA10Y is used alongside it for long-history coverage.
