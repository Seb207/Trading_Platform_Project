#!/usr/bin/env python3
"""Upgrade locally-extracted metadata to canonical arXiv API metadata.

Local PDF/MD extraction is a stopgap used while export.arxiv.org was rate-limited.
The API is the source of truth (accurate title, authors, abstract, published date).

This script finds entries with an empty `published` field (the local-extracted
marker), re-fetches them from the API in gently-paced batches, and OVERWRITES them
with the accurate API record — preserving local file location (relative_path,
category, format).

Run when the API is reachable:
    curl --max-time 12 "https://export.arxiv.org/api/query?search_query=cat:q-fin.CP&max_results=1"
    → HTTP 200 means safe to run.

After it completes, rebuild the abstract index so the accurate abstracts replace
the heuristic ones:
    python3 -c "import sys; sys.path.insert(0,'.'); \
        from arxiv_client import ArxivToolClient; \
        ArxivToolClient(download_dir='papers/arXiv').build_search_index()"
"""

import sys
import time

RESEARCH_LLM = "/Users/ahnsebin/Documents/Personal Project/Quant/Trading_Platform_Project/Research_LLM"
DOWNLOAD_DIR = f"{RESEARCH_LLM}/papers/arXiv"

sys.path.insert(0, RESEARCH_LLM)
from arxiv_client import ArxivToolClient  # noqa: E402

client = ArxivToolClient(download_dir=DOWNLOAD_DIR)

BATCH_SIZE = 25          # gentle on the API
INTER_BATCH_DELAY = 8.0  # seconds between batches
MAX_ROUNDS = 5


def stamp() -> str:
    return time.strftime("%H:%M:%S")


def local_ids() -> list[str]:
    """Entries whose metadata came from local extraction (no published date)."""
    meta = client._load_metadata()
    return [aid for aid, v in meta.items() if not str(v.get("published", "")).strip()]


for rnd in range(1, MAX_ROUNDS + 1):
    ids = local_ids()
    if not ids:
        print(f"[{stamp()}] All metadata is API-sourced. Nothing to refresh.", flush=True)
        break

    print(f"[{stamp()}] === Round {rnd}/{MAX_ROUNDS}: {len(ids)} local entries to upgrade ===", flush=True)
    upgraded = 0

    for i in range(0, len(ids), BATCH_SIZE):
        batch = ids[i:i + BATCH_SIZE]
        papers = client.fetch_paper_metadata_batch(batch)

        if papers and "error" in papers[0]:
            print(f"[{stamp()}]   batch {i//BATCH_SIZE+1}: throttled — retry next round", flush=True)
        else:
            meta = client._load_metadata()
            for p in papers:
                aid = p.get("arxiv_id", "")
                old = meta.get(aid, {})
                # Overwrite with API record; keep local file-location fields
                client._upsert_paper_metadata(
                    arxiv_id=aid,
                    title=p.get("title", old.get("title", "")),
                    authors=p.get("authors", []),
                    published=p.get("published", ""),
                    summary=p.get("summary", old.get("summary", "")),
                    category=old.get("category", p.get("category", "Unknown")),
                    fmt=old.get("format", "md"),
                    relative_path=old.get("relative_path", ""),
                )
                upgraded += 1
            print(f"[{stamp()}]   batch {i//BATCH_SIZE+1}: +{len(papers)} upgraded (round total {upgraded})", flush=True)

        time.sleep(INTER_BATCH_DELAY)

    remaining = len(local_ids())
    print(f"[{stamp()}] Round {rnd} done — upgraded={upgraded}, still local={remaining}", flush=True)
    if upgraded == 0:
        print(f"[{stamp()}] No progress — cooling down 60s…", flush=True)
        time.sleep(60)

final_local = local_ids()
print(f"\n[{stamp()}] === REFRESH COMPLETE ===", flush=True)
print(f"  still local-only: {len(final_local)}", flush=True)
if final_local:
    print(f"  (these IDs may be withdrawn/renamed on arXiv): {', '.join(final_local[:20])}", flush=True)
print(f"\n  Next: rebuild abstract index (see header docstring).", flush=True)
