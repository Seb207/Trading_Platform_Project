# Architecture Decision Records

## Philosophy

Optimize for accurate deep reading of a small curated set of papers over
broad retrieval across thousands. Prefer the model actually seeing the
paper's real text over any intermediate representation of it.

---

### ADR-001: Full-text reading over RAG chunking

**Decision**: papers are downloaded as clean Markdown and passed whole into
the LLM's context window, instead of chunking into a vector DB as the
primary retrieval mechanism.

**Reason**: quant papers have tightly coupled sections (methodology ↔
results ↔ math) that chunking breaks apart, and the target workflow is
deep reading of a small curated set, not broad retrieval across thousands.
Modern context windows (200k+ tokens) comfortably hold several full papers
at once.

**Tradeoff**: doesn't scale to "search across 10,000 papers" as cheaply as
pure embedding retrieval. Mitigated by a separate two-tier semantic layer
(ChromaDB abstract + section embeddings, Phase 4) for topic/section
discovery on top of, not instead of, full-text reading.

---

### ADR-002: Metadata source of truth is the arXiv API, not local parsing

**Decision**: `backfill_metadata` (the arXiv API) is canonical for title,
authors, abstract, and `published` date. Local `.md`/`.pdf` parsing is only
a stopgap for when the API is rate-limited.

**Reason**: local extraction is heuristic (title/abstract only, no reliable
`published` date) — good enough to keep the library usable during an API
outage, but never as accurate as the API record.

**Tradeoff**: entries downloaded during an API outage carry an empty
`published` field and need a later `refresh_api_metadata.py` pass (plus an
abstract-index rebuild) once the API recovers — an extra manual step, not
automatic.

---

### ADR-003: Separate metadata fetch from content download

**Decision**: `download_paper(fetch_metadata=False)` during bulk downloads
skips the per-paper metadata API call; `bulk_download_papers` uses this
internally plus one batched `backfill_metadata` pass (50 IDs/call)
afterward.

**Reason**: content (`arxiv.org/html`, `arxiv.org/pdf`) and metadata/search
(`export.arxiv.org/api/query`) are different hosts that throttle
independently — metadata throttles far more aggressively under load. This
cuts per-paper metadata calls by roughly 50× during bulk downloads.

**Tradeoff**: metadata isn't available immediately after a bulk download;
it lands after the batched backfill pass completes.
