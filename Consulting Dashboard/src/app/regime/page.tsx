"use client";

import { useEffect, useState } from "react";
import FactorPicker from "@/components/regime/FactorPicker";
import SimilarityRanking from "@/components/regime/SimilarityRanking";
import AnalogColormap from "@/components/regime/AnalogColormap";
import EventTimeOverlay from "@/components/regime/EventTimeOverlay";
import ForwardReturnFanChart from "@/components/regime/ForwardReturnFanChart";
import ValidationPanel from "@/components/regime/ValidationPanel";
import { fetchRegimeAnalogs, fetchRegimeFactors } from "@/lib/api";
import type { RegimeAnalogsResponse, RegimeFactor } from "@/lib/types";

function todayIso(): string {
  return new Date().toISOString().slice(0, 10);
}

const K_MIN = 1;
const K_MAX = 20;
const K_DEFAULT = 5;

function clampK(raw: string): number {
  const n = Number(raw);
  if (!Number.isFinite(n)) return K_DEFAULT;
  return Math.min(K_MAX, Math.max(K_MIN, Math.round(n)));
}

export default function RegimePage() {
  const [asOf, setAsOf] = useState(todayIso());
  const [allFactors, setAllFactors] = useState<RegimeFactor[]>([]);
  const [allThemes, setAllThemes] = useState<string[]>([]);
  const [selectedFactors, setSelectedFactors] = useState<string[]>([]);
  // Kept as a raw string so the field can be freely cleared/edited while
  // typing — coercing straight to a number on every keystroke forces the
  // input back to "0" the instant it's empty, which then sticks in front
  // of whatever the user types next. k (below) is the clamped number
  // actually used for queries.
  const [kInput, setKInput] = useState(String(K_DEFAULT));
  const k = clampK(kInput);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RegimeAnalogsResponse | null>(null);

  const [factorsLoading, setFactorsLoading] = useState(true);
  const [factorsError, setFactorsError] = useState<string | null>(null);
  const [factorsReloadKey, setFactorsReloadKey] = useState(0);

  // Auto-retries for ~60s: if the page loads before the backend has finished
  // starting up, the very first fetch gets a connection error with no
  // built-in retry — and previously that left the factor picker stuck on
  // "Loading factors…" forever, even after the backend came up, since a
  // plain useEffect(..., []) only runs once. Retrying covers the common
  // "both dev servers started together" race without any user action.
  // 60s (not 15s) because the backend's cold start imports pandas/numpy/
  // scipy plus uvicorn's --reload watcher spawning a subprocess — measured
  // at 1-3s typically, but slow enough on a cold cache to blow past a
  // shorter window (observed directly: a 15s/10-attempt budget expired
  // while the backend was still mid-import). retryFactors() below is the
  // manual fallback for anything slower than that.
  useEffect(() => {
    let cancelled = false;
    let attempt = 0;
    const maxAttempts = 30;
    const retryDelayMs = 2000;

    setFactorsLoading(true);
    setFactorsError(null);

    const attemptLoad = () => {
      fetchRegimeFactors()
        .then((r) => {
          if (cancelled) return;
          setAllFactors(r.factors);
          setAllThemes(r.themes);
          setFactorsLoading(false);
        })
        .catch((e) => {
          if (cancelled) return;
          attempt += 1;
          if (attempt < maxAttempts) {
            setTimeout(attemptLoad, retryDelayMs);
          } else {
            setFactorsLoading(false);
            setFactorsError(
              e instanceof Error ? e.message : "Could not reach the backend",
            );
          }
        });
    };
    attemptLoad();

    return () => {
      cancelled = true;
    };
  }, [factorsReloadKey]);

  const retryFactors = () => setFactorsReloadKey((k) => k + 1);

  const handleFindAnalogs = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await fetchRegimeAnalogs({
        asOf,
        factors: selectedFactors.length > 0 ? selectedFactors : null,
        k,
      });
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to find analogs");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col flex-1 w-full min-w-0 h-full overflow-y-auto">
      {/* Header */}
      <div className="flex-shrink-0 flex items-center gap-3 px-6 py-4 border-b border-border">
        <span className="font-mono text-accent3 text-xs tracking-widest uppercase">
          Market Regime
        </span>
        <span className="text-text-dim text-[11px]">
          Find historically similar periods — interpretation is yours, not the tool&apos;s.
        </span>
      </div>

      {/* Query controls */}
      <div className="flex-shrink-0 flex flex-col gap-4 px-6 py-4 border-b border-border bg-bg2">
        <div className="flex items-end gap-4 flex-wrap">
          <div className="flex flex-col gap-1">
            <label className="font-mono text-[10px] text-text-dim uppercase tracking-wide">
              As of
            </label>
            <input
              type="date"
              value={asOf}
              onChange={(e) => setAsOf(e.target.value)}
              className="bg-bg3 border border-border rounded-sm px-2 py-1.5 font-mono text-[12px] text-text focus:border-accent outline-none"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="font-mono text-[10px] text-text-dim uppercase tracking-wide">
              k (analogs)
            </label>
            <input
              type="number"
              min={K_MIN}
              max={K_MAX}
              value={kInput}
              onChange={(e) => setKInput(e.target.value)}
              onBlur={() => setKInput(String(k))}
              className="w-16 bg-bg3 border border-border rounded-sm px-2 py-1.5 font-mono text-[12px] text-text focus:border-accent outline-none"
            />
          </div>
          <button
            onClick={handleFindAnalogs}
            disabled={loading}
            className="font-mono text-[11px] uppercase tracking-wide px-4 py-2 rounded-sm border border-accent text-accent hover:bg-accent/10 transition-colors disabled:opacity-40"
          >
            {loading ? "Searching…" : "Find analogs"}
          </button>
        </div>

        <FactorPicker
          factors={allFactors}
          themes={allThemes}
          selectedFactors={selectedFactors}
          onChange={setSelectedFactors}
          loading={factorsLoading}
          error={factorsError}
          onRetry={retryFactors}
        />
      </div>

      {/* Results */}
      <div className="flex-1 flex flex-col gap-6 px-6 py-6">
        {error && (
          <div className="px-4 py-2 bg-neg/10 border border-neg/30 rounded-sm">
            <span className="font-mono text-[11px] text-neg">{error}</span>
          </div>
        )}

        {!result && !error && !loading && (
          <div className="flex flex-col items-center justify-center flex-1 gap-2 text-center py-16">
            <span className="font-mono text-[10px] text-text-dim tracking-widest uppercase">
              No query yet
            </span>
            <p className="text-text-dim text-[12px] max-w-[320px]">
              Pick a date and (optionally) narrow to specific factors, then click{" "}
              <span className="text-text">Find analogs</span>.
            </p>
          </div>
        )}

        {result && (
          <>
            <div className="flex items-center gap-3">
              <span className="font-mono text-[11px] text-text-dim">
                Query snapped to <span className="text-accent">{result.query_date}</span>
                {" · "}
                {result.vector_dimensions.length} vector dimensions from{" "}
                {result.factors_used.length} factors
              </span>
            </div>

            <SimilarityRanking analogs={result.analogs} />

            <AnalogColormap
              analogs={result.analogs}
              vectorDimensions={result.vector_dimensions}
              groups={result.groups}
              factors={allFactors}
            />

            <EventTimeOverlay
              queryEventSeries={result.query_event_series}
              analogs={result.analogs}
            />

            <ForwardReturnFanChart analogs={result.analogs} />

            <ValidationPanel
              factors={selectedFactors.length > 0 ? selectedFactors : null}
              themes={null}
              k={k}
            />
          </>
        )}
      </div>
    </div>
  );
}
