# PRD: Research_LLM (Paper2Alpha)

## Goal

Let an AI assistant autonomously search, download, and deeply read arXiv
quant-finance papers, then generate grounded quant strategy ideas from
them — without the accuracy loss of chunked RAG retrieval on papers whose
sections (methodology ↔ results ↔ math) are tightly coupled.

## Users

Single user (a quant researcher), driving this through an MCP-connected AI
assistant (Claude Code, or the Consulting Dashboard's Paper2Alpha chat).
Not a hosted/multi-user service.

## Core Features

1. Search arXiv by keyword/date/category; bulk download as clean Markdown
   (HTML→MD, PDF fallback).
2. Read a downloaded paper with pagination, or get a full methodology
   breakdown for strategy generation.
3. Two-tier semantic search (ChromaDB) for topic/section discovery across
   a large local library — abstract-level and section-level — as a
   complement to full-text reading, not a replacement for it.
4. Bridged into the Consulting Dashboard's chat UI (`/research`), where a
   critic pass reviews the LLM's grounded answers before finalizing them.

## MVP Exclusions

- No paper recommendation/ranking beyond similarity search.
- No automatic strategy backtesting — this module produces methodology
  breakdowns and candidate signal definitions; implementing/backtesting
  them is a separate, manual step.
- No multi-user library — one local `papers/` directory.

## Design

Not a UI project — this is an MCP server + library. Visual conventions
(when surfaced through the Consulting Dashboard) follow that project's own
`docs/UI_GUIDE.md`, not a separate one here.
