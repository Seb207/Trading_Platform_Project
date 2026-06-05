"use client";

import { useState, useRef } from "react";

const BASE_URL = "http://localhost:8000";

// ── arXiv category list ─────────────────────────────────────────────────
// Local DB categories (sorted by paper count) + extra categories for arXiv search
const ARXIV_CATEGORIES = [
  { value: "",           label: "All categories" },
  // Local DB
  { value: "q-fin.RM",  label: "q-fin.RM · Risk Management         (111 papers)" },
  { value: "q-fin.MF",  label: "q-fin.MF · Mathematical Finance    (105 papers)" },
  { value: "q-fin.CP",  label: "q-fin.CP · Computational Finance    (96 papers)"  },
  { value: "q-fin.ST",  label: "q-fin.ST · Statistical Finance      (90 papers)"  },
  { value: "q-fin.PM",  label: "q-fin.PM · Portfolio Management     (84 papers)"  },
  { value: "q-fin.TR",  label: "q-fin.TR · Trading & Microstructure (62 papers)"  },
  { value: "q-fin.GN",  label: "q-fin.GN · General Finance          (46 papers)"  },
  { value: "q-fin.PR",  label: "q-fin.PR · Pricing of Securities    (36 papers)"  },
  // Additional arXiv categories
  { value: "q-fin.EC",  label: "q-fin.EC · Economics" },
  { value: "cs.AI",     label: "cs.AI   · Artificial Intelligence" },
  { value: "cs.LG",     label: "cs.LG   · Machine Learning" },
  { value: "econ.EM",   label: "econ.EM · Econometrics" },
];

// ── Types ───────────────────────────────────────────────────────────────
interface PreviewPaper {
  arxiv_id: string;
  title: string;
  authors: string[];
  published: string;
  summary: string;
  category: string;
}

type LogLine =
  | { kind: "info";    text: string }
  | { kind: "ok";      text: string }
  | { kind: "error";   text: string }
  | { kind: "step";    text: string }
  | { kind: "done";    text: string };

// ── Component ───────────────────────────────────────────────────────────
export default function DownloadPanel() {
  // Search form
  const [query,      setQuery]      = useState("");
  const [category,   setCategory]   = useState("");
  const [dateFrom,   setDateFrom]   = useState("");
  const [dateTo,     setDateTo]     = useState("");
  const [maxResults, setMaxResults] = useState("10");
  const [autoIndex,  setAutoIndex]  = useState(true);

  // Preview state
  const [previewing,    setPreviewing]    = useState(false);
  const [previewPapers, setPreviewPapers] = useState<PreviewPaper[]>([]);
  const [previewError,  setPreviewError]  = useState<string | null>(null);
  const [selected,      setSelected]      = useState<Set<string>>(new Set());

  // Download state
  const [downloading, setDownloading] = useState(false);
  const [log,         setLog]         = useState<LogLine[]>([]);
  const logRef = useRef<HTMLDivElement>(null);

  const appendLog = (line: LogLine) => {
    setLog((prev) => [...prev, line]);
    // Scroll to bottom
    setTimeout(() => logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" }), 20);
  };

  // ── Preview ────────────────────────────────────────────────────────
  const handlePreview = async () => {
    if (!query && !category) {
      setPreviewError("Enter a query or select a category.");
      return;
    }
    setPreviewing(true);
    setPreviewError(null);
    setPreviewPapers([]);
    setSelected(new Set());
    setLog([]);

    try {
      const res = await fetch(`${BASE_URL}/api/arxiv/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          category,
          date_from: dateFrom,
          date_to: dateTo,
          max_results: Math.max(1, Math.min(50, parseInt(maxResults, 10) || 10)),
        }),
      });
      if (!res.ok) {
        // Read the actual error detail from FastAPI response body
        let detail = `HTTP ${res.status}`;
        try { const err = await res.json(); detail = err.detail ?? detail; } catch { /* ignore */ }
        throw new Error(detail);
      }
      const data = await res.json();
      setPreviewPapers(data.results ?? []);
      if ((data.results ?? []).length === 0) setPreviewError("No results found.");
      else {
        // Select all by default
        setSelected(new Set(data.results.map((p: PreviewPaper) => p.arxiv_id)));
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Preview failed";
      if (msg.includes("429") || msg.includes("503") || msg.includes("502") || msg.includes("504")) {
        setPreviewError("arXiv API rate limited / timed out — please try again shortly.");
      } else {
        setPreviewError(msg);
      }
    } finally {
      setPreviewing(false);
    }
  };

  // ── Toggle selection ───────────────────────────────────────────────
  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const toggleAll = () => {
    if (selected.size === previewPapers.length) {
      setSelected(new Set());
    } else {
      setSelected(new Set(previewPapers.map((p) => p.arxiv_id)));
    }
  };

  // ── Download ───────────────────────────────────────────────────────
  const handleDownload = async () => {
    const ids = [...selected];
    if (ids.length === 0) return;

    setDownloading(true);
    setLog([{ kind: "info", text: `Starting download of ${ids.length} paper(s)…` }]);

    // Category for file organisation: use selected category or "Unknown"
    const folderCat = category || "Unknown";

    let response: Response;
    try {
      response = await fetch(`${BASE_URL}/api/papers/download`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ arxiv_ids: ids, category: folderCat, auto_index: autoIndex }),
      });
    } catch {
      appendLog({ kind: "error", text: "Cannot reach backend — is FastAPI running on port 8000?" });
      setDownloading(false);
      return;
    }

    if (!response.ok || !response.body) {
      appendLog({ kind: "error", text: `API error ${response.status}` });
      setDownloading(false);
      return;
    }

    const reader  = response.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        try {
          const ev = JSON.parse(line.slice(6));

          if (ev.type === "paper_start") {
            appendLog({ kind: "info", text: `[${ev.index}/${ev.total}] Downloading: ${ev.arxiv_id}` });
          } else if (ev.type === "paper_done") {
            appendLog({ kind: "ok", text: `  ✓ ${ev.arxiv_id} (${ev.format})` });
          } else if (ev.type === "paper_error") {
            appendLog({ kind: "error", text: `  ✗ ${ev.arxiv_id}: ${ev.message}` });
          } else if (ev.type === "step") {
            appendLog({ kind: "step", text: `⟳ ${ev.message}` });
          } else if (ev.type === "step_done") {
            const n = ev.name;
            if (n === "backfill")       appendLog({ kind: "ok", text: `  ✓ Metadata updated (${ev.backfilled ?? 0} entries)` });
            if (n === "abstract_index") appendLog({ kind: "ok", text: `  ✓ Abstract index built (${ev.total_in_collection ?? ev.indexed ?? 0} papers total)` });
            if (n === "section_index")  appendLog({ kind: "ok", text: `  ✓ Section index built (${ev.sections_indexed ?? 0} sections)` });
          } else if (ev.type === "step_error") {
            appendLog({ kind: "error", text: `  ✗ ${ev.name}: ${ev.message}` });
          } else if (ev.type === "done") {
            appendLog({
              kind: "done",
              text: `Done — ${ev.downloaded} saved, ${ev.failed} failed`,
            });
          }
        } catch { /* ignore */ }
      }
    }

    setDownloading(false);
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-border flex-shrink-0">
        <span className="font-mono text-[11px] text-text font-medium tracking-wide uppercase">
          arXiv Download
        </span>
        <span className="font-mono text-[9px] text-text-dim">
          arXiv API → local folder → ChromaDB
        </span>
      </div>

      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto flex flex-col gap-0">

        {/* ── Search form ── */}
        <div className="px-5 py-4 flex flex-col gap-3 border-b border-border">
          {/* Query */}
          <div>
            <label className="font-mono text-[9px] text-text-dim tracking-widest uppercase mb-1 block">
              Search Query (optional)
            </label>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handlePreview()}
              placeholder="e.g. cross-sectional momentum low volatility"
              className="w-full bg-bg3 border border-border rounded-sm px-3 py-1.5 font-mono text-[11px] text-text placeholder:text-text-dim outline-none focus:border-accent"
            />
          </div>

          {/* Category */}
          <div>
            <label className="font-mono text-[9px] text-text-dim tracking-widest uppercase mb-1 block">
              Category
            </label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="w-full bg-bg3 border border-border rounded-sm px-3 py-1.5 font-mono text-[11px] text-text outline-none focus:border-accent"
            >
              {ARXIV_CATEGORIES.map((c) => (
                <option key={c.value} value={c.value}>{c.label}</option>
              ))}
            </select>
          </div>

          {/* Date range */}
          <div className="flex gap-3">
            <div className="flex-1">
              <label className="font-mono text-[9px] text-text-dim tracking-widest uppercase mb-1 block">
                Date From
              </label>
              <input
                type="date"
                value={dateFrom}
                onChange={(e) => setDateFrom(e.target.value)}
                className="w-full bg-bg3 border border-border rounded-sm px-3 py-1.5 font-mono text-[11px] text-text outline-none focus:border-accent"
              />
            </div>
            <div className="flex-1">
              <label className="font-mono text-[9px] text-text-dim tracking-widest uppercase mb-1 block">
                Date To
              </label>
              <input
                type="date"
                value={dateTo}
                onChange={(e) => setDateTo(e.target.value)}
                className="w-full bg-bg3 border border-border rounded-sm px-3 py-1.5 font-mono text-[11px] text-text outline-none focus:border-accent"
              />
            </div>
          </div>

          {/* Max results + auto-index row */}
          <div className="flex items-end gap-4">
            <div>
              <label className="font-mono text-[9px] text-text-dim tracking-widest uppercase mb-1 block">
                Max Results
              </label>
              <input
                type="number"
                min={1}
                max={50}
                value={maxResults}
                onChange={(e) => {
                  // Allow free editing (incl. empty) — only keep digits, no clamping here
                  const v = e.target.value.replace(/[^0-9]/g, "");
                  setMaxResults(v);
                }}
                onBlur={() => {
                  // Normalize to [1, 50] when the field loses focus
                  const n = Math.max(1, Math.min(50, parseInt(maxResults, 10) || 10));
                  setMaxResults(String(n));
                }}
                className="w-24 bg-bg3 border border-border rounded-sm px-3 py-1.5 font-mono text-[11px] text-text outline-none focus:border-accent"
              />
            </div>
            <label className="flex items-center gap-2 cursor-pointer pb-0.5">
              <input
                type="checkbox"
                checked={autoIndex}
                onChange={(e) => setAutoIndex(e.target.checked)}
                className="accent-accent w-3.5 h-3.5"
              />
              <span className="font-mono text-[10px] text-text-mid">Auto-embed (ChromaDB)</span>
            </label>
          </div>

          {/* Preview button */}
          <button
            onClick={handlePreview}
            disabled={previewing}
            className="w-full py-2 border border-accent text-accent font-mono text-[11px] rounded-sm hover:bg-accent/10 transition-colors disabled:opacity-50"
          >
            {previewing ? "Searching…" : "Preview arXiv Search"}
          </button>

          {previewError && (
            <p className="font-mono text-[10px] text-neg">{previewError}</p>
          )}
        </div>

        {/* ── Preview results ── */}
        {previewPapers.length > 0 && (
          <div className="flex flex-col">
            {/* Select bar */}
            <div className="flex items-center justify-between px-5 py-2 bg-bg2 border-b border-border">
              <button
                onClick={toggleAll}
                className="font-mono text-[9px] text-text-dim hover:text-text transition-colors"
              >
                {selected.size === previewPapers.length ? "Deselect All" : "Select All"}
              </button>
              <span className="font-mono text-[9px] text-text-dim">
                {selected.size} / {previewPapers.length} selected
              </span>
              <button
                onClick={handleDownload}
                disabled={selected.size === 0 || downloading}
                className={[
                  "px-3 py-1 font-mono text-[10px] rounded-sm border transition-colors",
                  selected.size > 0 && !downloading
                    ? "border-accent2 text-accent2 hover:bg-accent2/10"
                    : "border-border text-text-dim opacity-50 cursor-not-allowed",
                ].join(" ")}
              >
                {downloading ? "Downloading…" : `Download ${selected.size}`}
              </button>
            </div>

            {/* Paper list */}
            {previewPapers.map((p, idx) => (
              <label
                key={`${p.arxiv_id}-${idx}`}
                className="flex items-start gap-3 px-5 py-3 border-b border-border/40 hover:bg-bg2 cursor-pointer"
              >
                <input
                  type="checkbox"
                  checked={selected.has(p.arxiv_id)}
                  onChange={() => toggleSelect(p.arxiv_id)}
                  className="accent-accent mt-0.5 flex-shrink-0 w-3.5 h-3.5"
                />
                <div className="flex flex-col gap-0.5 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-mono text-[9px] text-accent shrink-0">{p.arxiv_id}</span>
                    <span className="font-mono text-[9px] text-accent2 shrink-0">{p.category}</span>
                    {p.published && (
                      <span className="font-mono text-[9px] text-text-dim shrink-0">
                        {p.published.slice(0, 10)}
                      </span>
                    )}
                  </div>
                  <p className="text-[11px] text-text leading-snug line-clamp-2">{p.title}</p>
                  {p.authors.length > 0 && (
                    <p className="text-[10px] text-text-dim truncate">
                      {p.authors.slice(0, 3).join(", ")}{p.authors.length > 3 ? " +…" : ""}
                    </p>
                  )}
                </div>
              </label>
            ))}
          </div>
        )}

        {/* ── Download log ── */}
        {log.length > 0 && (
          <div className="flex flex-col border-t border-border">
            <div className="px-5 py-2 bg-bg2 flex items-center justify-between">
              <span className="font-mono text-[9px] text-text-dim uppercase tracking-widest">Progress Log</span>
              {!downloading && (
                <button
                  onClick={() => setLog([])}
                  className="font-mono text-[9px] text-text-dim hover:text-text transition-colors"
                >
                  Clear
                </button>
              )}
            </div>
            <div
              ref={logRef}
              className="max-h-48 overflow-y-auto px-5 py-3 flex flex-col gap-1 bg-bg font-mono text-[10px]"
            >
              {log.map((line, i) => (
                <span
                  key={i}
                  className={
                    line.kind === "ok"    ? "text-accent" :
                    line.kind === "error" ? "text-neg" :
                    line.kind === "step"  ? "text-accent2" :
                    line.kind === "done"  ? "text-accent font-bold" :
                    "text-text-dim"
                  }
                >
                  {line.text}
                </span>
              ))}
              {downloading && (
                <span className="text-accent2 animate-pulse">…</span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
