"use client";

import { useCallback, useState } from "react";
import FilterChip from "@/components/ui/FilterChip";
import ToggleGroup from "@/components/ui/ToggleGroup";
import type { SearchMode } from "@/lib/types";

// All categories present in the local DB (sorted by paper count desc)
const CATEGORIES = [
  "q-fin.RM",  // 111 papers
  "q-fin.MF",  // 105 papers
  "q-fin.CP",  //  96 papers
  "q-fin.ST",  //  90 papers
  "q-fin.PM",  //  84 papers
  "q-fin.TR",  //  62 papers
  "q-fin.GN",  //  46 papers
  "q-fin.PR",  //  36 papers
];

// arXiv is intentionally excluded — it's a separate explicit-only trigger
const LOCAL_MODES: { value: Exclude<SearchMode, "arxiv">; label: string }[] = [
  { value: "abstract", label: "Abstract" },
  { value: "section",  label: "Section"  },
];

const PLACEHOLDERS: Record<SearchMode, string> = {
  abstract: "Search local papers by topic... (e.g. 'BAB factor low volatility')",
  section:  "Search within paper sections... (e.g. 'portfolio construction methodology')",
  arxiv:    "Search arXiv live... (e.g. 'cross-sectional momentum 2024')",
};

interface SearchBarProps {
  onSearch: (query: string, mode: SearchMode, category: string) => void;
  loading?: boolean;
}

export default function SearchBar({ onSearch, loading = false }: SearchBarProps) {
  const [query,        setQuery]        = useState("");
  const [localMode,    setLocalMode]    = useState<"abstract" | "section">("abstract");
  const [arxivActive,  setArxivActive]  = useState(false);
  const [category,     setCategory]     = useState("");

  // Active mode: arXiv takes precedence when toggled on
  const mode: SearchMode = arxivActive ? "arxiv" : localMode;

  // Section requires a non-empty query
  const shouldSkip = (q: string, m: SearchMode) => m === "section" && !q.trim();

  // ── Core trigger — always explicit, never called from state-change handlers ──
  const trigger = useCallback(
    (overrides: { query?: string; mode?: SearchMode; category?: string } = {}) => {
      const q   = overrides.query    ?? query;
      const m   = overrides.mode     ?? mode;
      const cat = overrides.category ?? category;
      if (shouldSkip(q, m)) return;
      onSearch(q.trim(), m, cat);
    },
    [query, mode, category, onSearch],
  );

  // ── Local mode toggle (Abstract / Section) ─────────────────────────────
  // Switches local mode + auto-triggers local search, but first exits arXiv mode
  const handleLocalMode = useCallback(
    (m: "abstract" | "section") => {
      setArxivActive(false);
      setLocalMode(m);
      if (!shouldSkip(query, m)) {
        onSearch(query.trim(), m, category);
      }
    },
    [query, category, onSearch],
  );

  // ── arXiv toggle button ────────────────────────────────────────────────
  // Only changes mode state — never auto-calls arXiv API
  const handleArxivToggle = useCallback(() => {
    if (arxivActive) {
      // Turning OFF: revert to local mode, re-run local search
      setArxivActive(false);
      if (!shouldSkip(query, localMode)) {
        onSearch(query.trim(), localMode, category);
      }
    } else {
      // Turning ON: activate arXiv mode, but DO NOT fire API call yet
      setArxivActive(true);
    }
  }, [arxivActive, localMode, query, category, onSearch]);

  // ── Category chip ──────────────────────────────────────────────────────
  // arXiv mode: update state only — no auto-trigger
  // Local mode: update state + auto-trigger
  const handleCategory = useCallback(
    (cat: string) => {
      const next = cat === category ? "" : cat;
      setCategory(next);
      if (!arxivActive && !shouldSkip(query, localMode)) {
        onSearch(query.trim(), localMode, next);
      }
    },
    [arxivActive, category, localMode, query, onSearch],
  );

  const handleAll = useCallback(() => {
    setCategory("");
    if (!arxivActive && !shouldSkip(query, localMode)) {
      onSearch(query.trim(), localMode, "");
    }
  }, [arxivActive, localMode, query, onSearch]);

  // ── Explicit search (button / Enter) ──────────────────────────────────
  const handleSearch = useCallback(() => trigger(), [trigger]);
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleSearch();
  };

  return (
    <div className="flex flex-col gap-2.5 px-5 py-3.5 border-b border-border">

      {/* ── Search input row ── */}
      <div className="flex gap-2.5">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={PLACEHOLDERS[mode]}
          className={[
            "flex-1 bg-bg3 border rounded-sm",
            "px-3.5 py-2 font-mono text-[12px] text-text",
            "placeholder:text-text-dim outline-none transition-colors duration-150",
            arxivActive ? "border-accent2 focus:border-accent2" : "border-border focus:border-accent",
          ].join(" ")}
        />
        <button
          onClick={handleSearch}
          disabled={loading}
          className={[
            "px-4 py-2 rounded-sm font-mono text-[11px] font-semibold tracking-wide uppercase",
            "border transition-all duration-150",
            loading ? "opacity-40 cursor-not-allowed" : "cursor-pointer",
            arxivActive
              ? "border-accent2 text-accent2 hover:bg-accent2/10"
              : "border-accent  text-accent  hover:bg-accent/10",
          ].join(" ")}
        >
          {loading ? "···" : "Search"}
        </button>
      </div>

      {/* ── Filter row: category chips + mode controls ── */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="font-mono text-[10px] text-text-dim tracking-wide">FILTER:</span>

        <FilterChip label="All" active={category === ""} onClick={handleAll} />
        {CATEGORIES.map((cat) => (
          <FilterChip
            key={cat}
            label={cat}
            active={category === cat}
            onClick={() => handleCategory(cat)}
          />
        ))}

        <div className="ml-auto flex items-center gap-2">
          {/* Local mode toggle — Abstract / Section */}
          <ToggleGroup
            options={LOCAL_MODES}
            value={arxivActive ? ("" as "abstract") : localMode}
            onChange={handleLocalMode}
          />

          {/* arXiv — separate explicit-only button */}
          <button
            onClick={handleArxivToggle}
            title="arXiv live search — only fires on explicit Search click"
            className={[
              "px-2.5 py-1 rounded-sm font-mono text-[10px] border transition-all duration-150",
              arxivActive
                ? "border-accent2 text-accent2 bg-accent2/10"
                : "border-border text-text-dim hover:border-accent2/50 hover:text-accent2",
            ].join(" ")}
          >
            arXiv ↗
          </button>
        </div>
      </div>

      {/* ── Mode hints ── */}
      {mode === "section" && !query.trim() && (
        <div className="flex items-center gap-2 py-0.5">
          <span className="text-[10px] text-text-dim font-mono">
            ℹ Section search requires a query — type a keyword above and press Search.
          </span>
        </div>
      )}
      {arxivActive && (
        <div className="flex items-center gap-2 py-0.5">
          <span className="text-[10px] text-accent2/70 font-mono">
            ↗ arXiv mode — API called only on Search button or Enter.
          </span>
        </div>
      )}
    </div>
  );
}
