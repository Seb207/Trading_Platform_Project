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
| ISM PMI (manufacturing / services) | LSEG Datastream economics mnemonic, or ISM directly — **FRED licensing for this series needs to be verified, likely unavailable** | Level vs. 50 threshold used directly, not z-scored |
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
| Credit spreads (IG / HY OAS) | FRED `BAMLC0A0CM` (IG) / `BAMLH0A0HYM2` (HY) | Level z-score |
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
| Option open interest | LSEG (license tier needs verification) → fallback OCC/CBOE daily reports | OI change → z-score |
| VWAP | Computed directly from LSEG intraday bars (not a raw pull) | Close-vs-VWAP deviation → z-score |
| Market breadth (e.g., % of S&P 500 above 200dma) | Computed directly from LSEG full-universe price data | Ratio → z-score |

### Sentiment / Narrative
| Factor | Source | Initial processing |
|---|---|---|
| Major media trending keywords | LSEG `news_nl_search`, `important_company_news` | Keyword frequency → sentence-transformer embedding (kept as a separate text vector, not folded into the numeric vector) |
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

## Open / Next Steps

- Lock in theme weighting (how much each factor group contributes to the
  overall similarity score).
- Decide on the exact vector index implementation (ChromaDB vs. a simpler
  library).
- Verify current Refinitiv/LSEG license terms for bulk historical storage,
  option open interest access, and ISM PMI availability.
- Build the factor collection pipeline once the above is settled.
