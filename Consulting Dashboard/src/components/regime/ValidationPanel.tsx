"use client";

import { useState } from "react";
import { fetchRegimeValidation } from "@/lib/api";
import type { RegimeValidateResponse } from "@/lib/types";

interface ValidationPanelProps {
  factors?: string[] | null;
  themes?: string[] | null;
  k: number;
}

const HORIZON_ORDER = ["1m", "3m", "6m", "12m"];

function pct(v: number | undefined): string {
  return v === undefined ? "—" : `${(v * 100).toFixed(2)}%`;
}

export default function ValidationPanel({ factors, themes, k }: ValidationPanelProps) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RegimeValidateResponse | null>(null);
  const [mode, setMode] = useState<"validate" | "sweep">("validate");

  const run = async (m: "validate" | "sweep") => {
    setLoading(true);
    setError(null);
    setMode(m);
    try {
      const r = await fetchRegimeValidation({ factors, themes, k, mode: m, queryIntervalWeeks: 4 });
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Validation failed");
    } finally {
      setLoading(false);
    }
  };

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="self-start font-mono text-[10px] uppercase tracking-wide px-3 py-1.5 rounded-sm border border-border text-text-dim hover:border-accent2 hover:text-accent2 transition-colors"
      >
        Check statistical robustness (optional)
      </button>
    );
  }

  return (
    <div className="flex flex-col gap-3 border border-border rounded-sm p-4 bg-bg3">
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] text-text-dim uppercase tracking-wide">
          Robustness check — opt-in, walk-forward backtest across many historical query dates
        </span>
        <button
          onClick={() => setOpen(false)}
          className="font-mono text-[10px] text-text-dim hover:text-text"
        >
          ✕ close
        </button>
      </div>

      <div className="flex gap-2">
        <button
          onClick={() => run("validate")}
          disabled={loading}
          className={[
            "font-mono text-[10px] uppercase px-3 py-1.5 rounded-sm border transition-colors",
            mode === "validate" && result
              ? "border-accent text-accent"
              : "border-border text-text-dim hover:border-accent2 hover:text-accent2",
          ].join(" ")}
        >
          {loading && mode === "validate" ? "Running…" : "Run KS test"}
        </button>
        <button
          onClick={() => run("sweep")}
          disabled={loading}
          className={[
            "font-mono text-[10px] uppercase px-3 py-1.5 rounded-sm border transition-colors",
            mode === "sweep" && result
              ? "border-accent text-accent"
              : "border-border text-text-dim hover:border-accent2 hover:text-accent2",
          ].join(" ")}
        >
          {loading && mode === "sweep" ? "Running…" : "Run sensitivity sweep"}
        </button>
      </div>

      {loading && (
        <span className="font-mono text-[10px] text-text-dim">
          This re-runs find_analogs across hundreds of historical dates — may take up to a minute…
        </span>
      )}

      {error && <span className="font-mono text-[10px] text-neg">{error}</span>}

      {result && result.mode === "validate" && result.horizons && (
        <div className="flex flex-col gap-2">
          <span className="font-mono text-[10px] text-text-dim">
            {result.n_query_dates} query dates, k={result.k}. Trust the &quot;unique&quot;
            columns — the plain pooled test overstates significance (adjacent queries reuse
            the same analogs).
          </span>
          <table className="border-collapse w-full">
            <thead>
              <tr>
                {["Horizon", "n (unique)", "cond. mean", "uncond. mean", "KS p-value", "Significant?"].map(
                  (col) => (
                    <th
                      key={col}
                      className="px-2 py-1.5 text-left font-mono text-[9px] text-text-dim uppercase border-b border-border"
                    >
                      {col}
                    </th>
                  ),
                )}
              </tr>
            </thead>
            <tbody>
              {HORIZON_ORDER.filter((h) => result.horizons?.[h]).map((h) => {
                const s = result.horizons![h];
                if (s.note) {
                  return (
                    <tr key={h} className="border-b border-border/40">
                      <td className="px-2 py-1.5 font-mono text-[10px]">{h}</td>
                      <td colSpan={5} className="px-2 py-1.5 font-mono text-[10px] text-text-dim">
                        {s.note}
                      </td>
                    </tr>
                  );
                }
                return (
                  <tr key={h} className="border-b border-border/40">
                    <td className="px-2 py-1.5 font-mono text-[10px]">{h}</td>
                    <td className="px-2 py-1.5 font-mono text-[10px]">{s.n_conditioned_unique}</td>
                    <td className="px-2 py-1.5 font-mono text-[10px]">{pct(s.conditioned_unique_mean)}</td>
                    <td className="px-2 py-1.5 font-mono text-[10px]">{pct(s.unconditional_mean)}</td>
                    <td className="px-2 py-1.5 font-mono text-[10px]">{s.ks_pvalue_unique?.toFixed(4)}</td>
                    <td className="px-2 py-1.5 font-mono text-[10px]">
                      {s.significant_at_5pct_unique ? (
                        <span className="text-accent">yes (p&lt;0.05)</span>
                      ) : (
                        <span className="text-text-dim">no</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {result && result.mode === "sweep" && result.rows && (
        <div className="flex flex-col gap-2">
          <span className="font-mono text-[10px] text-text-dim">
            k × exclusion-window grid — is significance a property of one setting, or robust
            across configurations?
          </span>
          <table className="border-collapse w-full">
            <thead>
              <tr>
                {["k", "exclude wks", "n query", "n unique", "KS p-value", "Significant?"].map((col) => (
                  <th
                    key={col}
                    className="px-2 py-1.5 text-left font-mono text-[9px] text-text-dim uppercase border-b border-border"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.rows.map((row, i) => (
                <tr key={i} className="border-b border-border/40">
                  <td className="px-2 py-1.5 font-mono text-[10px]">{row.k}</td>
                  <td className="px-2 py-1.5 font-mono text-[10px]">{row.exclude_weeks}</td>
                  <td className="px-2 py-1.5 font-mono text-[10px]">{row.n_query_dates}</td>
                  <td className="px-2 py-1.5 font-mono text-[10px]">{row.n_unique_analogs ?? "—"}</td>
                  <td className="px-2 py-1.5 font-mono text-[10px]">
                    {row.ks_pvalue_unique?.toFixed(4) ?? "—"}
                  </td>
                  <td className="px-2 py-1.5 font-mono text-[10px]">
                    {row.significant ? (
                      <span className="text-accent">yes</span>
                    ) : (
                      <span className="text-text-dim">no</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
