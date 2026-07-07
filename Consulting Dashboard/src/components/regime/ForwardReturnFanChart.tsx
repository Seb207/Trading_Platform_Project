"use client";

import { useState } from "react";
import {
  Bar,
  BarChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceDot,
} from "recharts";
import ToggleGroup from "@/components/ui/ToggleGroup";
import type { RegimeAnalog } from "@/lib/types";

interface ForwardReturnFanChartProps {
  analogs: RegimeAnalog[];
}

const ANALOG_COLORS = ["#00cfff", "#ff4fa3", "#ffd700", "#ff8a3d", "#a78bfa"];
const HORIZON_ORDER = ["1m", "3m", "6m", "12m"];
type SortMode = "similarity" | "time";

// Inverse-distance weighting: closer analogs (smaller distance) get more
// weight in the average — never zero weight (unlike a linear 1 - d/maxD
// scheme, which would zero out the single furthest analog and break when
// there's only one result).
function similarityWeightedAverage(analogs: RegimeAnalog[], horizon: string): number | null {
  let weightedSum = 0;
  let weightTotal = 0;
  for (const a of analogs) {
    const r = a.forward_returns[horizon];
    if (r === null || r === undefined) continue;
    const weight = 1 / (a.distance + 1e-6);
    weightedSum += weight * r;
    weightTotal += weight;
  }
  return weightTotal > 0 ? (weightedSum / weightTotal) * 100 : null;
}

export default function ForwardReturnFanChart({ analogs }: ForwardReturnFanChartProps) {
  const [sortMode, setSortMode] = useState<SortMode>("similarity");

  if (analogs.length === 0) return null;

  // analogs arrives similarity-ordered (ascending distance) from the API —
  // fix each date's color to that original order so it stays stable when
  // switching to time order below, rather than reshuffling on toggle.
  const colorByDate = new Map(
    analogs.map((a, i) => [a.date, ANALOG_COLORS[i % ANALOG_COLORS.length]]),
  );

  const orderedAnalogs =
    sortMode === "time" ? [...analogs].sort((a, b) => a.date.localeCompare(b.date)) : analogs;

  const data = HORIZON_ORDER.map((h) => {
    const row: Record<string, number | null | string> = { horizon: h };
    orderedAnalogs.forEach((a) => {
      const r = a.forward_returns[h];
      row[a.date] = r === null || r === undefined ? null : r * 100;
    });
    return row;
  });

  const weightedAverages = HORIZON_ORDER.map((h) => ({
    horizon: h,
    avg: similarityWeightedAverage(analogs, h),
  }));

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <span className="font-mono text-[10px] text-text-dim uppercase tracking-wide">
          Forward SPY return from each analog date — what actually happened next
        </span>
        <ToggleGroup<SortMode>
          options={[
            { value: "similarity", label: "sort: similarity" },
            { value: "time", label: "sort: time" },
          ]}
          value={sortMode}
          onChange={setSortMode}
        />
      </div>

      <div className="flex items-center gap-4 flex-wrap">
        <span className="font-mono text-[9px] text-text-dim uppercase tracking-wide">
          Similarity-weighted avg:
        </span>
        {weightedAverages.map(({ horizon, avg }) => (
          <span key={horizon} className="font-mono text-[11px]">
            <span className="text-text-dim">{horizon}</span>{" "}
            <span
              className={avg === null ? "text-text-dim" : avg >= 0 ? "text-accent" : "text-neg"}
            >
              {avg === null ? "—" : `${avg >= 0 ? "+" : ""}${avg.toFixed(2)}%`}
            </span>
          </span>
        ))}
      </div>

      <div style={{ width: "100%", height: 260 }}>
        {/* key={sortMode} forces a clean remount on toggle — recharts does not
            reliably reorder an existing BarChart's series/legend when only the
            Bar children's array order changes across renders (observed: the
            legend and rendered bars kept the previous order after switching
            sort modes back and forth). A full remount sidesteps that. */}
        <ResponsiveContainer key={sortMode}>
          <BarChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
            <CartesianGrid stroke="#1f1f1f" strokeDasharray="2 4" />
            <XAxis
              dataKey="horizon"
              tick={{ fill: "#888888", fontSize: 10, fontFamily: "monospace" }}
              stroke="#1f1f1f"
            />
            <YAxis
              tick={{ fill: "#888888", fontSize: 10, fontFamily: "monospace" }}
              stroke="#1f1f1f"
              unit="%"
            />
            <ReferenceLine y={0} stroke="#1f1f1f" />
            <Tooltip
              contentStyle={{
                background: "#111111",
                border: "1px solid #1f1f1f",
                fontFamily: "monospace",
                fontSize: 11,
              }}
              formatter={(value) => `${Number(value).toFixed(2)}%`}
            />
            <Legend wrapperStyle={{ fontFamily: "monospace", fontSize: 10 }} />
            {orderedAnalogs.map((a) => (
              <Bar key={a.date} dataKey={a.date} fill={colorByDate.get(a.date)} opacity={0.85} />
            ))}
            {weightedAverages.map(({ horizon, avg }) =>
              avg === null ? null : (
                <ReferenceDot
                  key={horizon}
                  x={horizon}
                  y={avg}
                  r={4}
                  fill="#00ff88"
                  stroke="#0a0a0a"
                  strokeWidth={1}
                  ifOverflow="extendDomain"
                />
              ),
            )}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="font-mono text-[9px] text-text-dim leading-relaxed">
        Interpretation is on you — this shows what happened after these specific historical
        periods, not a forecast for today. The green dot is the inverse-distance-weighted average
        across the analogs shown (closer matches count more). See &quot;Check robustness&quot;
        below for a statistical read on how much weight to put on that.
      </p>
    </div>
  );
}
