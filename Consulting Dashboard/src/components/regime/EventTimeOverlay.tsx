"use client";

import { useEffect, useRef, useState } from "react";
import {
  Line,
  LineChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import type { RegimeAnalog, EventPoint } from "@/lib/types";

interface EventTimeOverlayProps {
  queryEventSeries: EventPoint[];
  analogs: RegimeAnalog[];
}

const ANALOG_COLORS = ["#00cfff", "#ff4fa3", "#ffd700", "#ff8a3d", "#a78bfa"];

// Default initial view — full data can span ±104 weeks (see regime.py), but
// most of the interesting action for a "what happened next" read is closer
// in. Scroll/pinch inside the chart, or the min/max inputs, to see more.
const DEFAULT_VIEW_WEEKS = 26;
const MIN_SPAN_WEEKS = 4;
// Scale factor applied to the visible span per wheel/pinch tick.
const ZOOM_STEP = 0.88;

function clampToRange(n: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, n));
}

export default function EventTimeOverlay({ queryEventSeries, analogs }: EventTimeOverlayProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  const offsets = new Set<number>();
  queryEventSeries.forEach((p) => offsets.add(p.offset_weeks));
  analogs.forEach((a) => a.event_series.forEach((p) => offsets.add(p.offset_weeks)));
  const sortedOffsets = [...offsets].sort((a, b) => a - b);

  const fullMin = sortedOffsets[0] ?? -DEFAULT_VIEW_WEEKS;
  const fullMax = sortedOffsets[sortedOffsets.length - 1] ?? DEFAULT_VIEW_WEEKS;

  const defaultDomain = (): [number, number] => [
    Math.max(fullMin, -DEFAULT_VIEW_WEEKS),
    Math.min(fullMax, DEFAULT_VIEW_WEEKS),
  ];

  const [domain, setDomain] = useState<[number, number]>(defaultDomain());
  const [minInput, setMinInput] = useState(String(domain[0]));
  const [maxInput, setMaxInput] = useState(String(domain[1]));

  // Wheel handler closes over `domain` — keep a ref so it always reads the
  // latest value without needing to re-attach the (non-passive) listener on
  // every zoom tick.
  const domainRef = useRef(domain);
  domainRef.current = domain;

  // New query result → reset the zoom to the default view for the new data,
  // rather than keeping a range that may no longer make sense.
  useEffect(() => {
    const next = defaultDomain();
    setDomain(next);
    setMinInput(String(next[0]));
    setMaxInput(String(next[1]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queryEventSeries]);

  // Scroll-wheel / trackpad pinch zoom, centered on the cursor position.
  // Attached as a native (non-React) listener with {passive: false} — React's
  // synthetic onWheel is passive by default, so e.preventDefault() inside it
  // is silently ignored and the page would scroll underneath the chart too.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const [lo, hi] = domainRef.current;
      const span = hi - lo;

      const fracX = clampToRange((e.clientX - rect.left) / rect.width, 0, 1);
      const centerValue = lo + fracX * span;

      const zoomIn = e.deltaY < 0; // scroll up / pinch out = zoom in
      const scale = zoomIn ? ZOOM_STEP : 1 / ZOOM_STEP;
      const newSpan = clampToRange(span * scale, MIN_SPAN_WEEKS, fullMax - fullMin);

      let newLo = centerValue - (centerValue - lo) * (newSpan / span);
      let newHi = newLo + newSpan;
      if (newLo < fullMin) {
        newLo = fullMin;
        newHi = newLo + newSpan;
      }
      if (newHi > fullMax) {
        newHi = fullMax;
        newLo = newHi - newSpan;
      }
      newLo = Math.round(clampToRange(newLo, fullMin, fullMax));
      newHi = Math.round(clampToRange(newHi, fullMin, fullMax));
      if (newHi <= newLo) return;

      setDomain([newLo, newHi]);
      setMinInput(String(newLo));
      setMaxInput(String(newHi));
    };

    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [fullMin, fullMax]);

  if (analogs.length === 0) return null;

  const queryMap = new Map(queryEventSeries.map((p) => [p.offset_weeks, p.return_pct]));
  const analogMaps = analogs.map((a) => new Map(a.event_series.map((p) => [p.offset_weeks, p.return_pct])));

  const data = sortedOffsets.map((offset) => {
    const row: Record<string, number | null> = { offset };
    row["query"] = queryMap.get(offset) ?? null;
    analogs.forEach((a, i) => {
      row[a.date] = analogMaps[i].get(offset) ?? null;
    });
    return row;
  });

  const applySet = () => {
    let lo = Number(minInput);
    let hi = Number(maxInput);
    if (!Number.isFinite(lo)) lo = fullMin;
    if (!Number.isFinite(hi)) hi = fullMax;
    lo = clampToRange(lo, fullMin, fullMax - MIN_SPAN_WEEKS);
    hi = clampToRange(hi, lo + MIN_SPAN_WEEKS, fullMax);
    setDomain([lo, hi]);
    setMinInput(String(lo));
    setMaxInput(String(hi));
  };
  const resetZoom = () => {
    const next = defaultDomain();
    setDomain(next);
    setMinInput(String(next[0]));
    setMaxInput(String(next[1]));
  };
  const zoomOutFull = () => {
    setDomain([fullMin, fullMax]);
    setMinInput(String(fullMin));
    setMaxInput(String(fullMax));
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <span className="font-mono text-[10px] text-text-dim uppercase tracking-wide">
          Event-time overlay — SPY % return, offset weeks from each date (0 = query / analog date)
        </span>
        <div className="flex items-center gap-2">
          <label className="font-mono text-[9px] text-text-dim">x-min</label>
          <input
            type="number"
            value={minInput}
            onChange={(e) => setMinInput(e.target.value)}
            className="w-14 bg-bg3 border border-border rounded-sm px-1.5 py-0.5 font-mono text-[10px] text-text focus:border-accent outline-none"
          />
          <label className="font-mono text-[9px] text-text-dim">x-max</label>
          <input
            type="number"
            value={maxInput}
            onChange={(e) => setMaxInput(e.target.value)}
            className="w-14 bg-bg3 border border-border rounded-sm px-1.5 py-0.5 font-mono text-[10px] text-text focus:border-accent outline-none"
          />
          <button
            onClick={applySet}
            className="font-mono text-[9px] uppercase px-2 py-0.5 rounded-sm border border-accent2 text-accent2 hover:bg-accent2/10"
          >
            set
          </button>
          <button
            onClick={resetZoom}
            className="font-mono text-[9px] text-text-dim hover:text-accent2 underline"
          >
            reset
          </button>
          <button
            onClick={zoomOutFull}
            className="font-mono text-[9px] text-text-dim hover:text-accent2 underline"
          >
            zoom out full ({fullMin}…{fullMax})
          </button>
        </div>
      </div>
      <div ref={containerRef} style={{ width: "100%", height: 320 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
            <CartesianGrid stroke="#1f1f1f" strokeDasharray="2 4" />
            <XAxis
              dataKey="offset"
              type="number"
              domain={domain}
              allowDataOverflow
              tick={{ fill: "#888888", fontSize: 10, fontFamily: "monospace" }}
              stroke="#1f1f1f"
              label={{ value: "weeks", position: "insideBottomRight", fill: "#555555", fontSize: 10 }}
            />
            <YAxis
              tick={{ fill: "#888888", fontSize: 10, fontFamily: "monospace" }}
              stroke="#1f1f1f"
              unit="%"
            />
            <ReferenceLine x={0} stroke="#555555" strokeDasharray="3 3" />
            <ReferenceLine y={0} stroke="#1f1f1f" />
            <Tooltip
              contentStyle={{
                background: "#111111",
                border: "1px solid #1f1f1f",
                fontFamily: "monospace",
                fontSize: 11,
              }}
              labelFormatter={(v) => `offset: ${v}wk`}
            />
            <Legend wrapperStyle={{ fontFamily: "monospace", fontSize: 10 }} />
            <Line
              type="monotone"
              dataKey="query"
              name="query (today)"
              stroke="#00ff88"
              strokeWidth={2.5}
              dot={false}
              connectNulls
            />
            {analogs.map((a, i) => (
              <Line
                key={a.date}
                type="monotone"
                dataKey={a.date}
                stroke={ANALOG_COLORS[i % ANALOG_COLORS.length]}
                strokeWidth={1.25}
                dot={false}
                connectNulls
                opacity={0.85}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="font-mono text-[9px] text-text-dim">
        Scroll or pinch inside the chart to zoom (centered on the cursor) — or type exact week
        bounds above and click Set.
      </p>
    </div>
  );
}
