"use client";

import type { RegimeAnalog } from "@/lib/types";

interface SimilarityRankingProps {
  analogs: RegimeAnalog[];
}

// analogs already arrive ranked ascending by distance (find_analogs_from_zscore
// selects candidates in that order) — this just makes the ranking visible.
export default function SimilarityRanking({ analogs }: SimilarityRankingProps) {
  if (analogs.length === 0) return null;

  const maxDistance = Math.max(...analogs.map((a) => a.distance));

  return (
    <div className="flex flex-col gap-2">
      <span className="font-mono text-[10px] text-text-dim uppercase tracking-wide">
        Ranked by similarity — closest historical analog first
      </span>
      <div className="flex flex-col gap-1.5">
        {analogs.map((a, i) => {
          // Relative similarity within this result set (100% = closest of the
          // k returned, not an absolute probability or score).
          const pct = maxDistance > 0 ? Math.max(4, 100 * (1 - a.distance / maxDistance)) : 100;
          return (
            <div key={a.date} className="flex items-center gap-3">
              <span className="font-mono text-[10px] text-text-dim w-6 flex-shrink-0">
                #{i + 1}
              </span>
              <span className="font-mono text-[11px] text-text w-24 flex-shrink-0">{a.date}</span>
              <div className="flex-1 h-4 bg-bg3 rounded-sm overflow-hidden">
                <div
                  className="h-full bg-accent/70 rounded-sm"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="font-mono text-[10px] text-text-dim w-20 flex-shrink-0 text-right">
                dist={a.distance.toFixed(2)}
              </span>
            </div>
          );
        })}
      </div>
      <p className="font-mono text-[9px] text-text-dim">
        Bar length is relative similarity within this result set only (100% = closest match
        returned) — not an absolute probability.
      </p>
    </div>
  );
}
