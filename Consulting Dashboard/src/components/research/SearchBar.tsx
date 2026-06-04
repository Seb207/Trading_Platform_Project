"use client";

import { useCallback, useState } from "react";
import FilterChip from "@/components/ui/FilterChip";
import ToggleGroup from "@/components/ui/ToggleGroup";
import type { SearchMode } from "@/lib/types";

// All categories present in the local DB (sorted by paper count desc)
const CATEGORIES = [
  "q-fin.RM",  // 111편
  "q-fin.MF",  // 105편
  "q-fin.CP",  //  96편
  "q-fin.ST",  //  90편
  "q-fin.PM",  //  84편
  "q-fin.TR",  //  62편
  "q-fin.GN",  //  46편
  "q-fin.PR",  //  36편
];

const MODE_OPTIONS: { value: SearchMode; label: string }[] = [
  { value: "abstract", label: "Abstract" },
  { value: "section",  label: "Section"  },
  { value: "arxiv",    label: "arXiv"    },
];

const PLACEHOLDERS: Record<SearchMode, string> = {
  abstract: "Search local papers by topic... (e.g. 'BAB factor low volatility')",
  section:  "Search within paper sections... (e.g. 'portfolio construction methodology')",
  arxiv:    "Search arXiv live... (e.g. 'cross-sectional momentum 2024')",
};

const MODE_HINTS: Record<SearchMode, string | null> = {
  abstract: null,
  section:  "Section search requires a query — type a term above and press Search.",
  arxiv:    null,
};

interface SearchBarProps {
  onSearch: (query: string, mode: SearchMode, category: string) => void;
  loading?: boolean;
}

export default function SearchBar({ onSearch, loading = false }: SearchBarProps) {
  const [query,    setQuery]    = useState("");
  const [mode,     setMode]     = useState<SearchMode>("abstract");
  const [category, setCategory] = useState("");

  // Section mode needs a query — don't fire without one
  const shouldSkip = (q: string, m: SearchMode) => m === "section" && !q.trim();

  const trigger = useCallback(
    (overrides: { query?: string; mode?: SearchMode; category?: string } = {}) => {
      const q   = overrides.query    ?? query;
      const m   = overrides.mode     ?? mode;
      const cat = overrides.category ?? category;
      if (shouldSkip(q, m)) return;   // Section requires query
      onSearch(q.trim(), m, cat);
    },
    [query, mode, category, onSearch],
  );

  const handleCategory = useCallback(
    (cat: string) => {
      const next = cat === category ? "" : cat;
      setCategory(next);
      trigger({ category: next });
    },
    [category, trigger],
  );

  const handleAll = useCallback(() => {
    setCategory("");
    trigger({ category: "" });
  }, [trigger]);

  const handleMode = useCallback(
    (m: SearchMode) => {
      setMode(m);
      // Don't auto-trigger if switching to section with no query
      if (!shouldSkip(query, m)) {
        trigger({ mode: m });
      }
    },
    [query, trigger],
  );

  const handleSearch = useCallback(() => trigger(), [trigger]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleSearch();
  };

  const hint = !query.trim() ? MODE_HINTS[mode] : null;

  return (
    <div className="flex flex-col gap-2.5 px-5 py-3.5 border-b border-border">
      {/* Search input */}
      <div className="flex gap-2.5">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={PLACEHOLDERS[mode]}
          className={[
            "flex-1 bg-bg3 border border-border rounded-sm",
            "px-3.5 py-2 font-mono text-[12px] text-text",
            "placeholder:text-text-dim outline-none",
            "focus:border-accent transition-colors duration-150",
          ].join(" ")}
        />
        <button
          onClick={handleSearch}
          disabled={loading}
          className={[
            "px-4 py-2 rounded-sm font-mono text-[11px] font-semibold tracking-wide uppercase",
            "border border-accent text-accent transition-all duration-150",
            loading
              ? "opacity-40 cursor-not-allowed"
              : "hover:bg-accent/10 cursor-pointer",
          ].join(" ")}
        >
          {loading ? "···" : "Search"}
        </button>
      </div>

      {/* Category chips + mode toggle */}
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
        <div className="ml-auto">
          <ToggleGroup options={MODE_OPTIONS} value={mode} onChange={handleMode} />
        </div>
      </div>

      {/* Mode-specific hint (only when relevant) */}
      {hint && (
        <div className="flex items-center gap-2 py-0.5">
          <span className="text-[10px] text-text-dim font-mono">ℹ {hint}</span>
        </div>
      )}
    </div>
  );
}
