# Task: Academic paper analysis

You are analyzing a quantitative finance research paper for a quant researcher.
The full paper text is provided in the context. Produce a structured analysis
using the EXACT section headers below, in order. Methodology must be dissected
in detail — this is the highest-priority section in both branches.

## Step 1 — Classify the paper direction

Determine and state the paper type, then follow the matching branch.

Apply this test strictly:

- **Branch A — ALPHA/STRATEGY** — ONLY if the paper ITSELF builds a tradable
  signal / factor / alpha / trading or portfolio strategy AND reports its
  trading performance from a backtest (e.g. returns, Sharpe, IC, drawdown).
- **Branch B — GENERAL** — everything else: papers that propose a model,
  method, theory, or empirical finding but do NOT themselves build and
  backtest a trading strategy (e.g. option pricing, volatility models, risk
  models, microstructure, econometric or ML techniques, surveys).

Decision rules:
- If you would have to INVENT the trading application yourself, it is Branch B,
  not Branch A. Pricing/valuation models that merely *could* enable a trade are
  Branch B — put any derived trading ideas in Branch B section 5 (Quant
  relevance), not in a Branch A strategy section.
- A paper is Branch A only when the trading strategy and its measured
  performance are explicitly in the paper.
- When unsure, choose Branch B.

Examples:
- "A momentum factor backtested on the S&P 500 with reported Sharpe 1.4"
  → Branch A (builds AND backtests a strategy).
- "A new option pricing model (Heston / GARCH / jump-diffusion) benchmarked
  against observed market prices" → Branch B (prices derivatives; no backtested
  trading strategy).
- "An LSTM that forecasts volatility, evaluated by RMSE" → Branch B (a method,
  no trading strategy).
- "An alpha-mining framework that discovers factors and reports their IC and
  portfolio returns" → Branch A.

---

## Branch A — ALPHA/STRATEGY papers

### 0. Triage
- Paper type · Asset class · Universe · Data frequency · Sample period
- Verdict: `pursue` / `adapt` / `skip` — one line on why.

### 1. Purpose & agenda
- The problem addressed and the core claim (2–4 bullets).

### 2. Methodology (detailed)
- Dissect each method step by step: inputs, transformations, signal
  construction, parameters, estimation, rebalancing, position sizing.
- If multiple methods exist, give each its own sub-heading and dissect each.

### 3. Performance metrics
- One table per method with: Sharpe, annualized return, volatility, max
  drawdown, IC / t-stat, turnover, benchmark, and net-or-gross of costs.
- Use ONLY values the paper reports. Mark anything unreported `N/R`. Never
  fabricate a number.
- If methods are comparable, rank them by Sharpe (or the paper's primary
  risk-adjusted metric) and state the ranking. If not directly comparable
  (different universe / period), say so explicitly.

### 4. Data requirements
- Exact data needed to reproduce the method (price, volume, fundamentals,
  options, alt-data, etc.), with required frequency and history length.

### 5. Credibility & red flags
- Checklist, each marked ✓ / ✗ / unclear:
  look-ahead bias · survivorship bias · transaction costs included ·
  out-of-sample / walk-forward validation · overfitting risk (parameters vs
  sample size) · multiple-testing / data-snooping.

### 6. Key results
- The paper's main empirical findings (bullets), numbers as reported.

### 7. Strategy applications
- Propose realistic ways to apply this as an investment / trading strategy.
  Every proposal MUST be implementable — no hand-waving.
- One card per option:
  - Strategy idea
  - Data needed
  - Implementation complexity: `L` / `M` / `H`
  - Cost sensitivity
  - Feasibility verdict: `directly usable` / `needs adaptation` / `research-only`
- Then name the ONE you recommend implementing first, and why.

### 8. Candidate signal / alpha definitions
- Extract the precise signal(s) as a formula or pseudo-expression (NOT prose),
  ready to implement. One block per candidate.

---

## Branch B — GENERAL papers

### 0. Triage
- Paper type · Field / topic · One-line takeaway.

### 1. Purpose & question
- The problem or question addressed (2–4 bullets).

### 2. Methodology (detailed)
- Dissect the approach step by step: data, model / technique, assumptions,
  estimation, validation. Same depth as Branch A — do NOT abbreviate.
- If multiple methods exist, give each its own sub-heading.

### 3. Key findings & final results
- The paper's actual final results and conclusions, stated accurately. Keep
  reported numbers as written. Tag anything you infer with `(inferred)`.

### 4. Insights & implications
- The core insights, why they matter, transferable ideas, and any limitations
  the authors themselves state.

### 5. Quant relevance
- If the findings have a concrete use for quant research or trading, state it.
  If there is none, write `None` — do not force a connection.

---

## Rules (all branches)

- Extract only what the paper states. Do not invent numbers, methods, or
  results. Every inference must be tagged `(inferred)`.
- Use the exact section headers above, in the given order.
- Methodology is the priority section — detailed and fully dissected.
- Use tables for metrics, bullets elsewhere.
- If the paper text is missing or unreadable, say so rather than guessing.
