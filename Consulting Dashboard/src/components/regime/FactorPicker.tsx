"use client";

import { useState } from "react";
import type { RegimeFactor } from "@/lib/types";

interface FactorPickerProps {
  factors: RegimeFactor[];
  themes: string[];
  selectedFactors: string[];
  onChange: (factors: string[]) => void;
  loading?: boolean;
  error?: string | null;
  onRetry?: () => void;
}

export default function FactorPicker({
  factors,
  themes,
  selectedFactors,
  onChange,
  loading = false,
  error = null,
  onRetry,
}: FactorPickerProps) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const factorsForTheme = (theme: string) => factors.filter((f) => f.theme === theme);

  const toggleExpanded = (theme: string) => {
    const next = new Set(expanded);
    next.has(theme) ? next.delete(theme) : next.add(theme);
    setExpanded(next);
  };

  const toggleFactor = (key: string) => {
    onChange(
      selectedFactors.includes(key)
        ? selectedFactors.filter((k) => k !== key)
        : [...selectedFactors, key],
    );
  };

  // Clicking a theme's header selects/deselects every factor in that theme
  // at once — a shortcut alongside per-factor checkboxes below it.
  const themeSelectionState = (theme: string): "all" | "some" | "none" => {
    const keys = factorsForTheme(theme).map((f) => f.key);
    const selectedCount = keys.filter((k) => selectedFactors.includes(k)).length;
    if (selectedCount === 0) return "none";
    if (selectedCount === keys.length) return "all";
    return "some";
  };

  const toggleTheme = (theme: string) => {
    const keys = factorsForTheme(theme).map((f) => f.key);
    const state = themeSelectionState(theme);
    if (state === "all") {
      onChange(selectedFactors.filter((k) => !keys.includes(k)));
    } else {
      onChange([...new Set([...selectedFactors, ...keys])]);
    }
  };

  if (factors.length === 0) {
    if (error) {
      return (
        <div className="flex items-center gap-2">
          <span className="font-mono text-[10px] text-neg">
            Couldn&apos;t load factors: {error}
          </span>
          {onRetry && (
            <button
              onClick={onRetry}
              className="font-mono text-[10px] text-accent2 underline hover:text-accent"
            >
              retry
            </button>
          )}
        </div>
      );
    }
    return (
      <span className="font-mono text-[10px] text-text-dim">
        {loading ? "Loading factors… (retrying if the backend is still starting up)" : "No factors available"}
      </span>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] text-text-dim uppercase tracking-wide">
          Factors {selectedFactors.length === 0 && "(none selected = all factors)"}
          {selectedFactors.length > 0 && ` — ${selectedFactors.length} selected`}
        </span>
        {selectedFactors.length > 0 && (
          <button
            onClick={() => onChange([])}
            className="font-mono text-[10px] text-text-dim hover:text-accent2 underline"
          >
            clear
          </button>
        )}
      </div>

      <div className="flex flex-col gap-1 border border-border rounded-sm divide-y divide-border max-h-[320px] overflow-y-auto">
        {themes.map((theme) => {
          const themeFactors = factorsForTheme(theme);
          const isExpanded = expanded.has(theme);
          const selState = themeSelectionState(theme);
          return (
            <div key={theme} className="bg-bg3">
              <div className="flex items-center gap-2 px-2 py-1.5">
                <button
                  onClick={() => toggleExpanded(theme)}
                  className="font-mono text-[9px] text-text-dim hover:text-text w-3 flex-shrink-0"
                  title={isExpanded ? "Collapse" : "Expand to see individual factors"}
                >
                  {isExpanded ? "▾" : "▸"}
                </button>
                <button
                  onClick={() => toggleTheme(theme)}
                  className={[
                    "flex-1 text-left font-mono text-[10px] tracking-wide px-1.5 py-0.5 rounded-sm border",
                    selState === "all"
                      ? "border-accent2 text-accent2 bg-accent2/5"
                      : selState === "some"
                        ? "border-accent2/50 text-accent2/80"
                        : "border-transparent text-text-dim hover:text-text",
                  ].join(" ")}
                >
                  {theme} ({themeFactors.length})
                  {selState === "some" && ` — ${themeFactors.filter((f) => selectedFactors.includes(f.key)).length} selected`}
                </button>
              </div>

              {isExpanded && (
                <div className="flex flex-col gap-0.5 pb-2 pl-7 pr-2">
                  {themeFactors.map((f) => (
                    <label
                      key={f.key}
                      className="flex items-center gap-2 py-0.5 cursor-pointer group"
                      title={f.start && f.end ? `${f.start} → ${f.end}` : "No coverage data"}
                    >
                      <input
                        type="checkbox"
                        checked={selectedFactors.includes(f.key)}
                        onChange={() => toggleFactor(f.key)}
                        className="accent-[#00cfff]"
                      />
                      <span className="font-mono text-[10px] text-text group-hover:text-accent2">
                        {f.name}
                      </span>
                      {f.start && (
                        <span className="font-mono text-[9px] text-text-dim ml-auto">
                          {f.start.slice(0, 4)}–{f.end?.slice(0, 4)}
                        </span>
                      )}
                    </label>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
