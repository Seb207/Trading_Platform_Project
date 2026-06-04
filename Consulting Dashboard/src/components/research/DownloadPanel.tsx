"use client";

import { useState, useRef } from "react";

const BASE_URL = "http://localhost:8000";

// ── arXiv category list ─────────────────────────────────────────────────
// ── 로컬 DB에 존재하는 카테고리 (편수 순) + arXiv 추가 검색용 ──────────
const ARXIV_CATEGORIES = [
  { value: "",           label: "All categories" },
  // ── 로컬 DB 보유 ──────────────────────────────
  { value: "q-fin.RM",  label: "q-fin.RM · Risk Management         (111편)" },
  { value: "q-fin.MF",  label: "q-fin.MF · Mathematical Finance    (105편)" },
  { value: "q-fin.CP",  label: "q-fin.CP · Computational Finance    (96편)" },
  { value: "q-fin.ST",  label: "q-fin.ST · Statistical Finance      (90편)" },
  { value: "q-fin.PM",  label: "q-fin.PM · Portfolio Management     (84편)" },
  { value: "q-fin.TR",  label: "q-fin.TR · Trading & Microstructure (62편)" },
  { value: "q-fin.GN",  label: "q-fin.GN · General Finance          (46편)" },
  { value: "q-fin.PR",  label: "q-fin.PR · Pricing of Securities    (36편)" },
  // ── arXiv 검색용 추가 카테고리 ───────────────
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
  const [maxResults, setMaxResults] = useState(10);
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
      setPreviewError("쿼리 또는 카테고리를 입력하세요.");
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
          max_results: maxResults,
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
      if ((data.results ?? []).length === 0) setPreviewError("검색 결과가 없습니다.");
      else {
        // Select all by default
        setSelected(new Set(data.results.map((p: PreviewPaper) => p.arxiv_id)));
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Preview failed";
      if (msg.includes("429") || msg.includes("503") || msg.includes("502")) {
        setPreviewError("arXiv API 일시 제한 — 잠시 후 다시 시도하세요. (보통 10~30초)");
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
    setLog([{ kind: "info", text: `${ids.length}개 논문 다운로드 시작…` }]);

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
      appendLog({ kind: "error", text: "백엔드에 연결할 수 없습니다." });
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
            appendLog({ kind: "info", text: `[${ev.index}/${ev.total}] 다운로드 중: ${ev.arxiv_id}` });
          } else if (ev.type === "paper_done") {
            appendLog({ kind: "ok", text: `  ✓ ${ev.arxiv_id} (${ev.format})` });
          } else if (ev.type === "paper_error") {
            appendLog({ kind: "error", text: `  ✗ ${ev.arxiv_id}: ${ev.message}` });
          } else if (ev.type === "step") {
            appendLog({ kind: "step", text: `⟳ ${ev.message}` });
          } else if (ev.type === "step_done") {
            const n = ev.name;
            if (n === "backfill")       appendLog({ kind: "ok", text: `  ✓ 메타데이터 업데이트 완료 (${ev.backfilled ?? 0}건)` });
            if (n === "abstract_index") appendLog({ kind: "ok", text: `  ✓ 초록 인덱스 완료 (총 ${ev.total_in_collection ?? ev.indexed ?? 0}편)` });
            if (n === "section_index")  appendLog({ kind: "ok", text: `  ✓ 섹션 인덱스 완료 (${ev.sections_indexed ?? 0} 섹션)` });
          } else if (ev.type === "step_error") {
            appendLog({ kind: "error", text: `  ✗ ${ev.name}: ${ev.message}` });
          } else if (ev.type === "done") {
            appendLog({
              kind: "done",
              text: `완료 — ${ev.downloaded}편 저장, ${ev.failed}편 실패`,
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
          arXiv API → 로컬 폴더 → ChromaDB
        </span>
      </div>

      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto flex flex-col gap-0">

        {/* ── Search form ── */}
        <div className="px-5 py-4 flex flex-col gap-3 border-b border-border">
          {/* Query */}
          <div>
            <label className="font-mono text-[9px] text-text-dim tracking-widest uppercase mb-1 block">
              검색 쿼리 (선택)
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
              카테고리
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
                시작일
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
                종료일
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
                최대 결과 수
              </label>
              <input
                type="number"
                min={1}
                max={50}
                value={maxResults}
                onChange={(e) => setMaxResults(Math.max(1, Math.min(50, parseInt(e.target.value) || 1)))}
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
              <span className="font-mono text-[10px] text-text-mid">자동 임베딩 (ChromaDB)</span>
            </label>
          </div>

          {/* Preview button */}
          <button
            onClick={handlePreview}
            disabled={previewing}
            className="w-full py-2 border border-accent text-accent font-mono text-[11px] rounded-sm hover:bg-accent/10 transition-colors disabled:opacity-50"
          >
            {previewing ? "검색 중…" : "arXiv 검색 미리보기"}
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
                {selected.size === previewPapers.length ? "전체 해제" : "전체 선택"}
              </button>
              <span className="font-mono text-[9px] text-text-dim">
                {selected.size} / {previewPapers.length} 선택
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
                {downloading ? "다운로드 중…" : `${selected.size}편 다운로드`}
              </button>
            </div>

            {/* Paper list */}
            {previewPapers.map((p) => (
              <label
                key={p.arxiv_id}
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
