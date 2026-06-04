"use client";

import { useCallback, useEffect, useState } from "react";
import SplitPanel from "@/components/layout/SplitPanel";
import SearchBar from "@/components/research/SearchBar";
import PaperTable from "@/components/research/PaperTable";
import Badge from "@/components/ui/Badge";
import ChatPanel from "@/components/chat/ChatPanel";
import { listPapers, searchAbstract, searchSections, searchArxiv } from "@/lib/api";
import type { Paper, SearchMode } from "@/lib/types";

// ── Status badges ───────────────────────────────────────────────────────
function StatusBadges() {
  const [counts, setCounts] = useState({ papers: 0, sections: 0 });
  useEffect(() => {
    fetch("http://localhost:8000/api/status")
      .then((r) => r.json())
      .then((d) => setCounts({ papers: d.papers_count, sections: d.sections_count }))
      .catch(() => {});
  }, []);
  return (
    <div className="flex items-center gap-2">
      <Badge variant="green">{counts.papers || "—"} papers</Badge>
      <Badge variant="cyan">
        {counts.sections ? counts.sections.toLocaleString() : "—"} sections
      </Badge>
    </div>
  );
}

// ── Empty state ─────────────────────────────────────────────────────────
type EmptyReason = "section-needs-query" | "no-results" | null;

function EmptyState({ reason }: { reason: EmptyReason }) {
  if (!reason) return null;
  return (
    <div className="flex flex-col items-center justify-center flex-1 gap-3 text-center px-6">
      {reason === "section-needs-query" && (
        <>
          <span className="font-mono text-[10px] text-text-dim tracking-widest uppercase">Section Search</span>
          <p className="text-text-dim text-[12px] leading-relaxed max-w-[260px]">
            섹션 검색은 쿼리가 필요합니다.<br />
            검색창에 키워드를 입력하세요.<br />
            <span className="text-text-mid text-[11px] mt-1 block font-mono">
              예: &quot;BAB factor portfolio construction&quot;
            </span>
          </p>
        </>
      )}
      {reason === "no-results" && (
        <>
          <span className="font-mono text-[10px] text-text-dim tracking-widest uppercase">No Results</span>
          <p className="text-text-dim text-[12px]">다른 쿼리나 필터를 시도해 보세요.</p>
        </>
      )}
    </div>
  );
}

// ── Left panel ─────────────────────────────────────────────────────────
function PaperLibrary({
  selected,
  onSelect,
}: {
  selected: Paper | null;
  onSelect: (p: Paper) => void;
}) {
  const [papers,      setPapers]      = useState<Paper[]>([]);
  const [loading,     setLoading]     = useState(false);
  const [error,       setError]       = useState<string | null>(null);
  const [emptyReason, setEmptyReason] = useState<EmptyReason>(null);
  const [hasLoaded,   setHasLoaded]   = useState(false);

  // Initial full list
  useEffect(() => {
    setLoading(true);
    listPapers("", 200)
      .then((r) => { setPapers(r.papers); setHasLoaded(true); })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const handleSearch = useCallback(
    async (query: string, mode: SearchMode, category: string) => {
      // Section needs query — SearchBar guards this but double-check
      if (mode === "section" && !query) {
        setEmptyReason("section-needs-query");
        setPapers([]);
        return;
      }

      setLoading(true);
      setError(null);
      setEmptyReason(null);

      try {
        if (mode === "abstract") {
          if (query) {
            const r = await searchAbstract(query, 30, category);
            setPapers(r.results);
            if (r.results.length === 0) setEmptyReason("no-results");
          } else {
            // No query → list (filtered by category)
            const r = await listPapers(category, 200);
            setPapers(r.papers);
            if (r.papers.length === 0) setEmptyReason("no-results");
          }

        } else if (mode === "section") {
          const r = await searchSections({ query, category, nResults: 30 });
          const asPapers: Paper[] = r.results.map((s) => ({
            arxiv_id:        s.arxiv_id,
            title:           `[${s.section_name}]  ${s.paper_title}`,
            category:        s.category,
            published:       "",
            abstract:        s.preview,
            similarity_score: s.similarity_score,
            relative_path:   s.relative_path,
          }));
          setPapers(asPapers);
          if (asPapers.length === 0) setEmptyReason("no-results");

        } else if (mode === "arxiv") {
          const r = await searchArxiv({ query, category, maxResults: 20 });
          setPapers(r.results);
          if (r.results.length === 0) setEmptyReason("no-results");
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Search failed";
        // Friendlier message for rate limiting
        setError(
          msg.includes("429")
            ? "arXiv API 일시 제한 (429) — 잠시 후 다시 시도하세요."
            : msg,
        );
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  const showEmpty = !loading && emptyReason !== null;
  const showTable = !loading && emptyReason === null && papers.length > 0;
  const showSkeleton = loading && papers.length === 0;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-border flex-shrink-0">
        <span className="font-mono text-[11px] text-text font-medium tracking-wide uppercase">
          Paper Library
        </span>
        <StatusBadges />
      </div>

      {/* Search + filters */}
      <div className="flex-shrink-0">
        <SearchBar onSearch={handleSearch} loading={loading} />
      </div>

      {/* Error banner */}
      {error && (
        <div className="flex-shrink-0 px-5 py-2 bg-neg/10 border-b border-neg/30 flex items-center gap-2">
          <span className="font-mono text-[10px] text-neg">{error}</span>
        </div>
      )}

      {/* Loading skeleton */}
      {showSkeleton && (
        <div className="flex-1 flex flex-col gap-0">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="flex gap-4 px-5 py-4 border-b border-border/40 animate-pulse">
              <div className="w-20 h-3 rounded bg-bg3" />
              <div className="flex-1 flex flex-col gap-1.5">
                <div className="h-3 rounded bg-bg3 w-4/5" />
                <div className="h-3 rounded bg-bg3 w-3/5" />
              </div>
              <div className="w-14 h-4 rounded bg-bg3" />
              <div className="w-10 h-3 rounded bg-bg3" />
            </div>
          ))}
        </div>
      )}

      {/* Empty / hint state */}
      {showEmpty && <EmptyState reason={emptyReason} />}

      {/* Paper table */}
      {showTable && (
        <PaperTable
          papers={papers}
          selectedId={selected?.arxiv_id}
          onSelect={onSelect}
        />
      )}

      {/* Initial load complete but list empty */}
      {!loading && hasLoaded && papers.length === 0 && !emptyReason && !error && (
        <EmptyState reason="no-results" />
      )}
    </div>
  );
}

// ── Page ───────────────────────────────────────────────────────────────
export default function ResearchPage() {
  const [selected, setSelected] = useState<Paper | null>(null);
  return (
    <SplitPanel
      left={<PaperLibrary selected={selected} onSelect={setSelected} />}
      right={<ChatPanel selectedPaper={selected} />}
    />
  );
}
