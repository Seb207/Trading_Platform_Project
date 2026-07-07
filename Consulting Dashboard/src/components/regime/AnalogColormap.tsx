"use client";

import { Fragment, useState } from "react";
import type { RegimeAnalog, RegimeFactor } from "@/lib/types";

interface AnalogColormapProps {
  analogs: RegimeAnalog[];
  vectorDimensions: string[];
  groups: Record<string, string[]>;
  factors: RegimeFactor[];
}

const TRANSFORM_LABELS: Record<string, string> = {
  level: "level",
  yoy: "YoY %",
  mom: "1M chg",
  log_level: "level (log)",
  ma4: "4W avg",
};

// Green (accent, #00ff88) intensity scaled by contribution share (0–1 of squared distance).
function cellStyle(value: number | undefined, dim = 1): React.CSSProperties {
  const v = Math.max(0, Math.min(1, value ?? 0));
  return {
    backgroundColor: `rgba(0, 255, 136, ${(0.06 + v * 0.55) * dim})`,
    color: v > 0.35 * dim ? "#0a0a0a" : "#e8e8e8",
  };
}

/** "cpi_headline__yoy" -> "CPI (headline) — YoY %" using the factor's real name. */
function prettySingle(col: string, factorsByKey: Map<string, RegimeFactor>): string {
  const sep = col.lastIndexOf("__");
  if (sep === -1) return col;
  const key = col.slice(0, sep);
  const transform = col.slice(sep + 2);
  const factor = factorsByKey.get(key);
  const label = TRANSFORM_LABELS[transform] ?? transform;
  return factor ? `${factor.name} — ${label}` : col;
}

export default function AnalogColormap({
  analogs,
  vectorDimensions,
  groups,
  factors,
}: AnalogColormapProps) {
  const [expandedDims, setExpandedDims] = useState<Set<string>>(new Set());

  if (analogs.length === 0) return null;

  const factorsByKey = new Map(factors.map((f) => [f.key, f]));

  const toggleExpanded = (dim: string) => {
    const next = new Set(expandedDims);
    next.has(dim) ? next.delete(dim) : next.add(dim);
    setExpandedDims(next);
  };

  // Order dimensions by their max contribution across analogs (most influential first)
  const orderedDims = [...vectorDimensions].sort((a, b) => {
    const maxA = Math.max(...analogs.map((an) => an.contributions[a] ?? 0));
    const maxB = Math.max(...analogs.map((an) => an.contributions[b] ?? 0));
    return maxB - maxA;
  });

  return (
    <div className="flex flex-col gap-2">
      <span className="font-mono text-[10px] text-text-dim uppercase tracking-wide">
        Factor contribution to similarity
      </span>
      <div className="overflow-x-auto">
        <table className="border-collapse w-full">
          <thead>
            <tr>
              <th className="px-3 py-2 text-left font-mono text-[10px] text-text-dim uppercase border-b border-border sticky left-0 bg-bg2">
                Factor
              </th>
              {analogs.map((a) => (
                <th
                  key={a.date}
                  className="px-3 py-2 text-left font-mono text-[10px] text-text-dim uppercase border-b border-border whitespace-nowrap"
                >
                  {a.date}
                  <div className="text-[9px] text-text-dim normal-case">
                    dist={a.distance.toFixed(2)}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {orderedDims.map((dim) => {
              const members = groups[dim];
              const isMerged = !!members;
              const isExpanded = expandedDims.has(dim);

              return (
                <Fragment key={dim}>
                  <tr className="border-b border-border/40">
                    <td className="px-3 py-1.5 font-mono text-[10px] text-text sticky left-0 bg-bg2 whitespace-nowrap">
                      {isMerged ? (
                        <button
                          onClick={() => toggleExpanded(dim)}
                          className="flex items-center gap-1.5 hover:text-accent2"
                          title="Click to see the individual factors merged into this dimension"
                        >
                          <span className="text-text-dim text-[9px]">
                            {isExpanded ? "▾" : "▸"}
                          </span>
                          <span>{members.length} factors merged</span>
                        </button>
                      ) : (
                        prettySingle(dim, factorsByKey)
                      )}
                    </td>
                    {analogs.map((a) => (
                      <td
                        key={a.date}
                        className="px-3 py-1.5 font-mono text-[10px] text-center"
                        style={cellStyle(a.contributions[dim])}
                      >
                        {((a.contributions[dim] ?? 0) * 100).toFixed(0)}%
                      </td>
                    ))}
                  </tr>

                  {isMerged &&
                    isExpanded &&
                    members.map((member) => (
                      <tr key={`${dim}::${member}`} className="border-b border-border/20 bg-bg3/40">
                        <td className="pl-9 pr-3 py-1 font-mono text-[9px] text-text-dim sticky left-0 bg-bg3/40 whitespace-nowrap">
                          {prettySingle(member, factorsByKey)}
                        </td>
                        {analogs.map((a) => {
                          const share = a.member_contributions?.[dim]?.[member];
                          return (
                            <td
                              key={a.date}
                              className="px-3 py-1 font-mono text-[9px] text-center"
                              style={cellStyle(share, 0.6)}
                              title="Share within this merged group — informational, not a decomposition of the group's own contribution above"
                            >
                              {share !== undefined ? `${(share * 100).toFixed(0)}%` : "—"}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="font-mono text-[9px] text-text-dim leading-relaxed">
        Expanded rows show each merged factor&apos;s own share within its group — informational, not
        a decomposition of the group&apos;s percentage above (the group score is based on the
        averaged z-score, not summed per-factor differences).
      </p>
    </div>
  );
}
