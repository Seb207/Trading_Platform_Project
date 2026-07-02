# Market Regime Detector

## Goal

Compare the current market state across a broad set of macro, rates, volatility,
liquidity, microstructure, sentiment, geopolitical, and policy factors against
historical periods to find the closest historical analog(s), then visualize
what happened after those analog periods.

This is a design/planning document. No implementation exists yet.

---

## 1. Similarity Search Methodology

### Decision: Hybrid approach

- **Numeric factors** (CPI, rates, VIX, credit spreads, etc.) → standardized
  (z-score), selectively decorrelated, compared via Euclidean/Mahalanobis
  distance. This path stays fully interpretable so each factor's contribution
  to the similarity score can be shown directly (drives the factor-importance
  colormap).
- **Text factors** (news/Reddit trending keywords) → embedded separately via
  a sentence-transformer model (reusing the existing `BAAI/bge-base-en-v1.5`
  infra already used for arXiv paper search), kept as its own vector — not
  mixed numerically with the macro factors.
- Final similarity score combines the numeric distance and the text-embedding
  distance with a tunable weight between the two.
- Retrieval: k-NN search (ChromaDB or a simpler vector index — exact choice
  not yet locked in) over the historical numeric vectors; text vectors used
  as a secondary signal/filter.

### Options considered and rejected (for reference)

| Option | Why not chosen as primary |
|---|---|
| Embed everything (including numeric macro factors) directly via ChromaDB/text-embedding models | Text embedding models aren't designed for raw numeric series; forcing it loses information |
| Pure black-box learned embedding (autoencoder / contrastive learning) | Not interpretable — cannot attribute similarity to individual factors, which the colormap requires. Also limited historical sample size for training a reliable encoder |
| Hidden Markov Model (HMM) regime switching | Works well in low dimensions (1-3 variables); not designed for the full heterogeneous factor set (macro + micro + text) used here |
| PCA + Clustering (original placeholder idea in the dashboard) | See dimensionality-reduction decision below — not rejected outright, used as a diagnostic tool instead of the primary pipeline |

### PCA vs. manual theme grouping vs. selective correlation-based grouping

| | PCA | Manual theme grouping | **Selected: correlation-driven selective grouping** |
|---|---|---|---|
| Interpretability | Low — PC axes are linear combinations, not human-readable | High — themes map to concepts (inflation, rates, risk appetite, etc.) | High — only genuinely correlated factors are merged; everything else stays as its own dimension |
| Handles factor redundancy | Yes, automatically | Manually, by design | Yes — driven by an actual correlation matrix, not intuition |
| Stability across re-fits | Low — PC axis meaning can shift when re-fit on a new window | High — theme definitions are fixed | High |
| Sample size sensitivity | Higher — covariance estimation needs enough independent observations; monthly macro series have few | Low | Low |
| Adding a new factor later | Requires full PCA re-fit + recompute all historical vectors + rebuild index | Just add to the relevant theme, no other impact | Same as theme grouping — low impact |

**Decision:** Do not use PCA as the primary dimensionality-reduction step. Use
the **correlation matrix as a diagnostic** to decide which factors are
genuinely redundant (e.g., `|r| > 0.7`) and merge only those into a single
combined score (e.g., CPI + PPI + PCE → one inflation score). Leave all other,
less-correlated factors (VIX, credit spreads, GPR index, Fed funds rate, etc.)
as standalone dimensions in the vector. This avoids both extremes:
- Naively using all ~25-30 raw factors as separate vector dimensions
  (over-weights correlated factor families, worsens the curse of
  dimensionality, adds more implicit per-factor weighting decisions, and is
  more fragile to missing data for newer/derived factors).
- Collapsing everything into a small number of opaque PCA components
  (loses interpretability needed for the colormap, unstable across re-fits,
  expensive to maintain as factors are added).

### Re-fit / index update cadence

- Heavy step (fit the z-score scaler, decide which factors to group, build
  the initial vector index) happens once on the historical backfill.
- Ongoing step (new period arrives) only requires applying the existing
  (frozen) transforms — no re-fitting — then querying the vector index for
  nearest neighbors and inserting the new vector. This is fast regardless of
  index size.
- Recommended re-fit cadence: rolling z-score normalization (e.g., trailing
  N-year window) updated continuously, with the factor grouping definitions
  and the correlation-based grouping decision revisited infrequently (e.g.,
  annually) rather than on every new data point — a full re-fit requires
  recomputing all historical vectors and rebuilding the index, so it should
  not happen often.

---

## 2. Factor List, Data Sources, and Initial Processing

Primary data source priority: **Refinitiv/LSEG first** (already licensed) →
**FRED** for anything not licensed or for canonical free macro series → other
verified sources for everything else.

> **Caveat:** Refinitiv/LSEG account terms commonly restrict bulk extraction
> and persistent local storage of historical data for non-trading-desk use.
> Confirm the current license's redistribution/storage terms before backfilling
> years of history locally.

### Confirmed access (live entitlement check, 2026-06-22)

Two separate Refinitiv/LSEG access paths exist for this project, with
**different and independently-checked entitlements**:

1. **Refinitiv RDP Platform session** (`refinitiv-data.config.json`, personal/academic
   account) — opened live via the `refinitiv.data` Python SDK and probed across
   content categories.
2. **LSEG MCP connector** (separate tool/app, not the same credentials) — probed
   by calling its tools directly.

| Content | RDP Platform | LSEG MCP connector |
|---|---|---|
| Equity/bond/FX pricing, fundamentals | ✅ Confirmed working | ❌ Blocked (`API_ACCESS_CONTROL/MCP_DATA_ON`) |
| CPI, ISM PMI (economic indicators) | ✅ Confirmed working (RIC `USCPI=ECI`, `USPMI=ECI`) | ❌ Blocked (`MCP_DATA_QA_ON`) |
| VIX, Treasury yields | ✅ Confirmed working | ❌ Blocked |
| News (major media keywords) | ❌ Blocked — `trapi.data.news.read` scope missing | ❌ Blocked — `MCP_DATA_MRN_ON` |
| Credit spread / credit curve | ⚠️ Inconclusive on RDP (wrong RIC tried) | ❌ Blocked — `LFA_API/EP_CREDITCURVES` |
| Option analytics / open interest | ⚠️ Inconclusive on RDP (wrong field name tried) | ❌ Blocked — `MCP_DATA_ON` |
| Fixed income curves (YieldBook) | Not tested | ❌ Blocked — `LFA_API/EP_SECURITIZED_BONDS` |
| Datastream, FTSE Russell IXM | Not applicable (different platform) | ❌ Blocked — `MCP_DATA_ON` |

**Key finding:** the LSEG MCP connector currently has **no active entitlements
for any content category tested** — including categories the RDP session can
already access (basic pricing, macro indicators). It cannot be used to fill
gaps the RDP session has (News, credit spreads, option data); it would need
entitlements activated by whoever administers that connector before it's
usable for anything in this project. All Refinitiv/LSEG access for now should
go through the RDP Platform session only.

**Resolved:** ISM PMI is confirmed available via Refinitiv (`USPMI=ECI`) —
no need to verify a separate licensing path for it.

**Still unresolved:** News access (media trending keywords) is confirmed
blocked on both paths — needs a different source entirely (not LSEG/Refinitiv).
Option open interest and credit-spread-via-Refinitiv remain unconfirmed
(wrong identifiers were used in testing, not a proven entitlement block) —
retest with correct RICs/field codes before falling back to OCC/CBOE or FRED.

### Inflation
| Factor | Source | Initial processing |
|---|---|---|
| CPI (headline / core) | FRED `CPIAUCSL` / `CPILFESL` | YoY % change → rolling z-score |
| PPI | FRED `PPIACO` | YoY % change → z-score |
| PCE (headline / core) | FRED `PCEPI` / `PCEPILFE` | YoY % change → z-score |
| → Inflation composite | — | Correlation matrix expected to show these as highly correlated; merge into one combined z-score if confirmed |

### Growth
| Factor | Source | Initial processing |
|---|---|---|
| ISM PMI (manufacturing / services) | **Confirmed via Refinitiv RDP** (`USPMI=ECI`) | Level vs. 50 threshold used directly, not z-scored |
| Unemployment rate | FRED `UNRATE` | Level + MoM change → z-score |
| Initial jobless claims | FRED `ICSA` | 4-week moving average → z-score |

### Rates / Yield Curve
| Factor | Source | Initial processing |
|---|---|---|
| Absolute yields (2Y / 10Y / 30Y) | LSEG `interest_rate_curve` or FRED `DGS2` / `DGS10` / `DGS30` | Level z-score |
| Term spread (2s10s) | FRED `T10Y2Y` | z-score |
| Term spread (3M-10Y) | FRED `T10Y3M` | z-score |
| Rate of change | Differencing the above series | z-score |

### Risk Appetite / Volatility
| Factor | Source | Initial processing |
|---|---|---|
| VIX | LSEG `historical_pricing_summaries` (`.VIX`) or FRED `VIXCLS` | Level z-score (consider log transform — distribution is skewed) |
| Realized volatility (e.g., 20-day S&P) | Computed directly from LSEG price data | VIX − realized vol = volatility risk premium, itself a useful derived factor |
| Credit spreads (IG / HY OAS) | FRED `BAMLC0A0CM` (IG) / `BAMLH0A0HYM2` (HY) — **ICE licensing caps FRED at a rolling 3-year window (verified 2026-07-02)**. Long history carried by Moody's `BAA10Y` spread instead (1986–, unrestricted). LSEG `credit_curve` confirmed blocked; RDP path untested with a valid RIC | Level z-score |
| Put/Call ratio | CBOE (direct, free) | z-score |

### Liquidity / Funding
| Factor | Source | Initial processing |
|---|---|---|
| M2 money supply | FRED `M2SL` | YoY % change → z-score |
| Fed balance sheet | FRED `WALCL` | Level / change → z-score |
| SOFR-OIS / repo spread | LSEG `ir_swap` or equivalent | z-score |

### Microstructure
| Factor | Source | Initial processing |
|---|---|---|
| Option open interest | LSEG RDP inconclusive (wrong field name tested, retest needed) — **confirmed blocked on the LSEG MCP connector** (`option_value` → `MCP_DATA_ON` entitlement error) → fallback OCC/CBOE daily reports if RDP retest also fails | OI change → z-score |
| VWAP | Computed directly from LSEG intraday bars (not a raw pull) | Close-vs-VWAP deviation → z-score |
| Market breadth (e.g., % of S&P 500 above 200dma) | Computed directly from LSEG full-universe price data | Ratio → z-score |

### Sentiment / Narrative
| Factor | Source | Initial processing |
|---|---|---|
| Major media trending keywords | **Confirmed blocked on both Refinitiv RDP (`trapi.data.news.read` missing) and the LSEG MCP connector (`news_nl_search`/`important_company_news` → `MCP_DATA_MRN_ON` entitlement error) — needs a non-LSEG news source, not yet identified** | Keyword frequency → sentence-transformer embedding (kept as a separate text vector, not folded into the numeric vector) |
| Reddit trending keywords | Reddit API (PRAW) — **no reliable historical backfill; build the corpus going forward only** | Same embedding approach as above |
| Analyst consensus / earnings revisions | LSEG MCP `qa_ibes_consensus` | EPS revision breadth → z-score (numeric factor, unlike the two above) |

### Geopolitical
| Factor | Source | Initial processing |
|---|---|---|
| Geopolitical risk | GPR Index ([matteoiacoviello.com/gpr.htm](https://www.matteoiacoviello.com/gpr.htm)) | Level z-score |

### Policy
| Factor | Source | Initial processing |
|---|---|---|
| Fed funds rate | FRED `FEDFUNDS` | Level / change → z-score |
| FOMC statement tone (hawkish/dovish) | Verbatim text from federalreserve.gov, scored via the existing LLM chat pipeline (same pattern as the paper-analysis task mode) | LLM structured score (e.g., -1 to +1) → z-score |

### FX / Cross-Asset
| Factor | Source | Initial processing |
|---|---|---|
| Dollar strength | FRED `DTWEXBGS` (free proxy for the proprietary ICE DXY ticker) | z-score |

### Frequency alignment

Factors span monthly (CPI, PPI, PCE), daily (VIX, rates), event-driven (FOMC,
~8x/year), and daily/weekly (news keywords) frequencies. Align everything to
the lowest common practical frequency (daily or weekly) and forward-fill
(last-observation-carried-forward) lower-frequency series until their next
release — standard macro nowcasting practice.

---

## 3. Visualization

### Confirmed
1. **Factor-importance colormap + data table** — rows = factors, columns =
   time, color intensity = z-score / contribution to similarity distance.
2. **Time-series overlay** — current period vs. matched historical analog(s),
   aligned at "event time 0," showing the subsequent trajectory.
3. **Keyword bubble / packed-circle chart** — news/Reddit trending keywords,
   bubble size scaled by exposure share.

### Additional options proposed (priority candidates)
4. **Forward-outcome distribution (box plot / fan chart)** — instead of a
   single historical path, show the distribution of forward 1/3/6/12-month
   returns across the top-k analog periods. Communicates uncertainty more
   honestly than one path. **Recommended as a near-term addition.**

### Additional options proposed (secondary candidates)
5. Radar/spider chart — current period's normalized factor "fingerprint"
   overlaid against the matched historical analog.
6. 2D/3D regime map (PCA/UMAP scatter) — all historical periods projected,
   current point highlighted with nearest neighbors.
7. Network graph of analog periods — node = period, edge weight = similarity;
   reveals whether the current period sits in a dense cluster (high
   confidence) or is isolated (novel regime).
8. Small-multiples sparkline grid — one mini time series per factor.
9. Regime timeline ribbon — full history as a horizontal strip, colored by
   regime cluster, with the current period and top analogs marked.
10. Keyword streamgraph / co-occurrence network — alternatives to the bubble
    chart that show keyword trends over time or thematic clustering, instead
    of just a single-snapshot frequency view.

---

## 4. Feasibility Assessment

**Verdict: feasible — conditional on phasing the scope as
"numeric-factor MVP → validation → text/narrative expansion."**

### Strengths (already secured)
- Core numeric data is confirmed by live entitlement checks: CPI/PMI/rates/
  VIX/FX via Refinitiv RDP; credit spreads, unemployment, M2, etc. via FRED
  (free, full history, no key even required for the CSV endpoint). The
  numeric-factor pipeline has effectively **zero data risk**.
- Infrastructure reuse: FastAPI router pattern, Next.js dashboard, the LLM
  pipeline (for FOMC scoring), and embedding infra all exist; the
  `regime/page.tsx` placeholder is already reserved in the dashboard.
- The methodology is interpretable by design — similarity can be decomposed
  per factor, which makes the system explainable and defensible.

### Risks (ranked by severity)

| # | Risk | Description | Mitigation |
|---|---|---|---|
| 1 | Becoming a "visualization toy" | Finding analogs is meaningless if analog-conditioned forward outcomes have no statistical content | Dedicated validation phase (Phase 2): test whether the forward-return distribution conditioned on top-k analogs differs from the unconditional distribution |
| 2 | Neighbor overlap | The nearest neighbors of "today" are almost always "yesterday" and "last week" (serial correlation) | Temporal exclusion window (±6 months around the query) + minimum separation (e.g., 6 months) enforced between returned analogs. **Mandatory — results are meaningless without this** |
| 3 | Effective sample size | Monthly macro over ~60 years ≈ 700 independent observations; the number of true distinct regimes is in the dozens | Keep k small (5–10), display fan-chart uncertainty honestly, avoid over-interpretation |
| 4 | Text-factor cold start | News blocked on both LSEG paths; Reddit has no backfill | Ship the MVP numeric-only; text is Phase 4. Two upgrades soften this (below) |

### Two upgrades that soften risk #4
- **FOMC statements ARE backfillable.** Full verbatim statements are free on
  federalreserve.gov back to ~1994 — LLM hawkish/dovish scoring can be applied
  retroactively, so the policy-text axis starts with full history.
- **Free news alternative: GDELT.** Global news themes/tone/keywords, free,
  from 2015 (events DB from 1979). NYT Archive API (headlines from 1851, free)
  as a secondary. This resolves the "needs a non-LSEG source" gap above.

---

## 5. Locked Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Vector index (numeric) | **No vector DB — brute-force k-NN (numpy/sklearn)** | Weekly frequency × 60 years ≈ 3,000 rows × ~15 dims; HNSW is over-engineering, brute force runs in milliseconds. ChromaDB reserved for Phase 4 text embeddings only |
| Comparison unit | **Level + 3-month change (trend) per factor**, not a point-in-time snapshot | "Inflation at 5% and rising" vs. "at 5% and falling" are different regimes — level alone cannot distinguish them |
| Base frequency | **Weekly** (not daily) | Macro-driven system; daily adds autocorrelation, not information. Also mitigates neighbor overlap |
| Theme weighting | **Equal weights initially**; revisit after Phase 2 sensitivity analysis | Tuning weights before validation puts the cart before the horse |
| Neighbor hygiene | Exclude ±6 months around query; enforce min 6-month separation between analogs | See risk #2 |

---

## 6. Roadmap

### Phase 0 — Data foundation ← **complete**
- [x] FRED access confirmed (no-key CSV endpoint verified, full history)
- [x] GPR index file availability confirmed (updated regularly)
- [x] Reusable data-collection package (`sources/`) — one module per source:
      `fred.py` (FredClient), `refinitiv_rdp.py` (RefinitivClient),
      `gpr.py` (GPR downloader), `cboe.py` (CBOE archive + splice),
      `occ.py` (OCC put/call + open interest, resumable backfill),
      `common.py` (shared raw cache + curl-based http_get).
      **Note: the Refinitiv RDP session only authenticates while the local
      Refinitiv platform (Workspace) is running — start it before collecting.**
- [x] Factor schema registry (`factor_schema.py`): every factor's source,
      identifier, theme, transform hint, and publication lag in one
      declarative table (add a factor = add one entry)
- [x] Build pipeline (`build_factor_dataset.py`): fetch all → align weekly
      (W-FRI) → LOCF → persist to `data/factors_weekly.parquet` (+ .csv),
      with per-source failure isolation (a Refinitiv outage never blocks FRED)
- [x] Option open interest resolved — OCC `/open-interest` API (free), and
      put/call ratio resolved — CBOE frozen archive (2006-11 → 2019-10)
      spliced with OCC daily volume (2019-10 → present)
- [x] SOFR-OIS spread resolved — `USDSR2YOTS=TWEB` via `rd.discovery.search`
- [x] **Deliverable achieved: `data/factors_weekly.parquet` — 26 factors,
      5,923 weekly rows (1913-01-03 → 2026-07-03), zero collection failures
      on the final run, all sources verified live (2026-07-02)**
- [ ] Confirm Refinitiv/LSEG license terms for bulk historical storage
      (still needs a human check — not resolvable from code)

Collection quirks discovered while building (encoded in `sources/`):
- FRED's CDN blocks python HTTP stacks AND custom User-Agents at the TLS/
  header level — `http_get` shells out to curl with curl's *default* UA
  (a custom UA via curl's `-A` flag still gets blocked — verified)
- `rd.get_history` silently caps at ~20 rows without an explicit `end` —
  the client always sends `end` + `interval`
- Refinitiv session: `platform.rdp` first, desktop-session fallback (same
  pattern as the earlier Factor_Management.py work); Workspace must be
  running locally
- `.SPX` index history is not permitted on this account → SPY ETF proxy
- ICE BofA credit series on FRED: rolling 3-year window only (`cosd` param
  doesn't help — it's an ICE licensing restriction, not a FRED default) →
  Moody's `BAA10Y` added as the long-history credit factor
- Refinitiv `.PCALL` and `CBOE-OPTTOT-*` (put/call, options stats) are not
  permitted on this account → CBOE/OCC used instead
- OCC `/open-interest` and `/daily-volume-totals` use different date formats
  (`MM/DD/YYYY` vs `YYYY-MM-DD`) and different pagination (one month per
  call vs. one day per call) — handled inside `occ.py`, not exposed to callers

### Currently collected (Phase 0 output, `data/factors_weekly.parquet`)

| Theme | Factors | Coverage |
|---|---|---|
| Inflation | cpi_headline, cpi_core, ppi, pce_headline, pce_core | 1913–1959 start → 2026-07 |
| Growth | ism_pmi, unemployment, jobless_claims | 1948–1980 start → 2026-07 |
| Rates / curve | ust_2y, ust_10y, ust_30y, spread_2s10s, spread_3m10y | 1962–1982 start → 2026-07 |
| Risk / volatility | vix, credit_ig, credit_hy, credit_baa_spread, put_call_ratio | 1986–2023 start → 2026-07 |
| Liquidity | m2, fed_balance_sheet, sofr_ois_2y | 1959–2021 start → 2026-07 |
| Microstructure | option_oi | 2024-06 → 2026-07 |
| Geopolitical | gpr | 1985 → 2026-06 |
| Policy | fed_funds | 1954 → 2026-06 |
| FX | dollar_index | 2006 → 2026-06 |
| Equity | spx (SPY proxy) | 1993 → 2026-07 |

**Not yet collected (deferred, unchanged from the original plan):**
`breadth` (market breadth — needs full-universe price pulls, heavier job,
deferred) and `fomc_tone` (FOMC statement LLM scoring — Phase 4).

### Phase 1 — Similarity engine MVP (numeric only)
- Correlation-matrix diagnostic → merge `|r| > 0.7` groups (+ PCA-loading
  cross-check to justify the grouping with data, not intuition)
- Rolling z-score (level + change) → vectorize → k-NN with temporal
  exclusion → per-factor contribution decomposition
- Smoke test: query known crisis dates (2008-09, 2020-03, 2022) and check
  the analogs match intuition
- **Deliverable: `find_analogs(date) → [(analog, similarity, per-factor contribution)]`**

### Phase 2 — Validation (where credibility is decided)
- Forward 1/3/6/12-month S&P return distribution of top-k analogs vs. the
  unconditional distribution (KS test or similar)
- Walk-forward protocol: the index only contains data prior to the query
  date (no look-ahead)
- Sensitivity analysis: theme weights, k, exclusion window
- **Deliverable: a quantitative report on whether analog-conditioning carries
  statistical content**

### Phase 3 — Dashboard integration
- New `backend/routers/regime.py` (same pattern as papers/chat routers)
- Implement `regime/page.tsx`: ① factor colormap + table ② event-time-aligned
  time-series overlay ③ forward-outcome fan chart
- **Deliverable: "top-5 most similar historical periods + what followed"
  queryable from the dashboard**

### Phase 4 — Text / narrative axis (can start in parallel after Phase 1)
- FOMC statements 1994–present: backfill + LLM hawkish/dovish scoring
  (reuse the task-mode pattern) → joins the numeric factor set
- GDELT pipeline for media keywords (2015–); Reddit PRAW collector starts
  accumulating forward (**start this early — its value grows with time**)
- Keyword bubble chart + hybrid text-embedding similarity
- **Deliverable: full system including the narrative axis + keyword visuals**

### Phase 5 — Extensions (optional)
- Regime cluster labels + timeline ribbon, radar chart, analog network graph
- Scheduled auto-refresh of all data sources

Dependency chain: 0 → 1 → 2 → 3 serial; Phase 4 parallelizable after Phase 1.
