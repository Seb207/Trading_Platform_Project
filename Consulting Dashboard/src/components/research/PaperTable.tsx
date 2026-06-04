"use client";

import CategoryTag from "@/components/ui/CategoryTag";
import type { Paper } from "@/lib/types";

interface PaperTableProps {
  papers: Paper[];
  selectedId?: string;
  onSelect?: (paper: Paper) => void;
}

function ScoreBar({ score }: { score?: number }) {
  if (score === undefined) return <span className="text-text-dim font-mono text-[11px]">—</span>;
  const pct = Math.round(score * 100);
  return (
    <div className="flex items-center gap-2">
      <span className="font-mono text-[11px] text-accent min-w-[36px]">
        {score.toFixed(3)}
      </span>
      <div className="flex-1 h-[3px] bg-border rounded-full">
        <div
          className="h-full bg-accent rounded-full"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function formatDate(dateStr: string): string {
  // "2026-03-02T12:52:50Z" → "2026-03"
  return dateStr.slice(0, 7);
}

// Highlight query terms in title (simple bold wrap)
function HighlightTitle({ title }: { title: string }) {
  return (
    <span className="text-[12px] text-text leading-[1.45]">{title}</span>
  );
}

export default function PaperTable({
  papers,
  selectedId,
  onSelect,
}: PaperTableProps) {
  if (papers.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center flex-1 gap-2 text-text-dim">
        <span className="font-mono text-[11px] tracking-widest uppercase">No Results</span>
        <span className="text-[12px]">Try a different query or category filter</span>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <table className="w-full border-collapse">
        {/* Header */}
        <thead className="sticky top-0 z-10 bg-bg2">
          <tr>
            {["ID", "Title", "Category", "Score", "Date"].map((col) => (
              <th
                key={col}
                className="px-5 py-2.5 text-left font-mono text-[10px] text-text-dim tracking-[1px] uppercase border-b border-border font-normal"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>

        {/* Body */}
        <tbody>
          {papers.map((paper) => {
            const isSelected = paper.arxiv_id === selectedId;
            return (
              <tr
                key={paper.arxiv_id}
                onClick={() => onSelect?.(paper)}
                className={[
                  "border-b border-border/40 cursor-pointer transition-colors duration-75 group",
                  isSelected
                    ? "bg-accent/5 border-l-2 border-l-accent"
                    : "hover:bg-bg3",
                ].join(" ")}
              >
                {/* arXiv ID */}
                <td className="px-5 py-3 align-top whitespace-nowrap">
                  <span className="font-mono text-[10px] text-accent2">
                    {paper.arxiv_id}
                  </span>
                </td>

                {/* Title */}
                <td className="px-5 py-3 align-top">
                  <HighlightTitle title={paper.title} />
                </td>

                {/* Category */}
                <td className="px-5 py-3 align-top whitespace-nowrap">
                  <CategoryTag category={paper.category} />
                </td>

                {/* Score */}
                <td className="px-5 py-3 align-top w-[120px]">
                  <ScoreBar score={paper.similarity_score} />
                </td>

                {/* Date */}
                <td className="px-5 py-3 align-top whitespace-nowrap">
                  <span className="font-mono text-[10px] text-text-dim">
                    {formatDate(paper.published)}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
