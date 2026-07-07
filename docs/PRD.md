# PRD: Trading Platform Project

## Goal

A personal quant research platform that compresses three normally-separate
workflows — reading academic research, finding historically similar market
regimes, and managing a portfolio — into one local-first dashboard, so a
solo researcher can go from "read a paper" or "what does the market look
like right now" to a concrete next step without switching tools.

## Users

Single user (a quant researcher/practitioner), running everything locally.
Not built for multi-tenant or public deployment — no auth system, no
hosting concerns, no multi-user data isolation.

## Core Modules

1. **Paper2Alpha** (`Research_LLM/`, surfaced at `/research`) — search,
   download, and deeply read arXiv quant-finance papers; chat with an LLM
   grounded in a selected paper's full text; get a structured methodology
   breakdown; a critic pass reviews the LLM's answer for grounding/format
   compliance before it's finalized.
2. **Market Regime Detector** (`Market Regime/`, surfaced at `/regime`) —
   given a date, retrieve historically similar macro/market regimes via
   hybrid numeric+correlation similarity search across ~26 factors, with
   several visualizations (colormap, event-time overlay, forward-return fan
   chart). Deliberately a retrieval tool, not a predictive one — see
   `ADR.md` for why.
3. **Portfolio** (`/portfolio`) — not yet built.
4. **Factor Research** (`/factor`) — not yet built.

## MVP Exclusions

- No multi-user accounts/auth.
- No hosted/cloud deployment — local dev servers only.
- Market Regime does not attempt to predict outcomes or gate on
  statistical validation by default (opt-in only).
- No mobile-responsive design — desktop dashboard only.

## Design Direction

Dark, terminal/mono aesthetic — see `UI_GUIDE.md` for the concrete palette
and component conventions already in use across the dashboard. Function
over decoration: this is a tool used daily, not a marketing surface.
