"use client";

import { useEffect, useRef, useState } from "react";
import { useLLM } from "@/context/LLMContext";
import type { LLMConfig, LLMProvider } from "@/lib/types";

const CLAUDE_MODELS = [
  "claude-opus-4-5",
  "claude-sonnet-4-5",
  "claude-haiku-4-5",
];

interface ModelSelectorProps {
  config: LLMConfig;
  onChange: (config: LLMConfig) => void;
}

/* ── Reusable custom dropdown ── */
interface DropdownProps {
  value: string;
  options: string[];
  onSelect: (v: string) => void;
  accent?: "accent" | "accent2";
  placeholder?: string;
}

function Dropdown({ value, options, onSelect, accent = "accent", placeholder = "—" }: DropdownProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const accentColor  = accent === "accent" ? "#00ff88" : "#00cfff";
  const accentClass  = accent === "accent"
    ? "border-accent  text-accent  focus:border-accent"
    : "border-accent2 text-accent2 focus:border-accent2";
  const hoverBg      = accent === "accent" ? "hover:bg-accent/8" : "hover:bg-accent2/8";
  const activeBg     = accent === "accent" ? "bg-accent/10"      : "bg-accent2/10";
  const activeText   = accent === "accent" ? "text-accent"       : "text-accent2";

  return (
    <div ref={ref} className="relative w-full">
      {/* Trigger */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={[
          "w-full flex items-center justify-between gap-2",
          "bg-bg3 border rounded-sm px-3 py-1.5",
          "font-mono text-[11px] text-text",
          "transition-colors duration-100 outline-none",
          open ? accentClass : "border-border hover:border-border2",
        ].join(" ")}
      >
        <span className={value ? "text-text" : "text-text-dim"}>
          {value || placeholder}
        </span>
        {/* Chevron */}
        <svg
          width="10" height="10" viewBox="0 0 10 10" fill="none"
          style={{
            transform: open ? "rotate(180deg)" : "rotate(0deg)",
            transition: "transform 150ms",
            color: open ? accentColor : "#555",
            flexShrink: 0,
          }}
        >
          <path d="M2 3.5L5 6.5L8 3.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>

      {/* Options list */}
      {open && options.length > 0 && (
        <div
          className="absolute z-50 w-full mt-1 bg-bg2 border border-border2 rounded-sm overflow-hidden"
          style={{ boxShadow: `0 4px 20px rgba(0,0,0,0.6), 0 0 0 1px rgba(${accent === "accent" ? "0,255,136" : "0,207,255"},0.08)` }}
        >
          {options.map((opt) => (
            <button
              key={opt}
              type="button"
              onClick={() => { onSelect(opt); setOpen(false); }}
              className={[
                "w-full text-left px-3 py-1.5 font-mono text-[11px]",
                "border-b border-border last:border-b-0",
                "transition-colors duration-75",
                opt === value
                  ? `${activeBg} ${activeText}`
                  : `text-text-mid ${hoverBg} hover:text-text`,
              ].join(" ")}
            >
              {opt}
              {opt === value && (
                <span className="float-right text-[9px] opacity-60">✓</span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Main component ── */
export default function ModelSelector({ config, onChange }: ModelSelectorProps) {
  const { setConfig: setGlobalConfig } = useLLM();
  const [expanded,      setExpanded]      = useState(false);
  const [apiKey,        setApiKey]        = useState(config.apiKey ?? "");
  const [ollamaUrl,     setOllamaUrl]     = useState(config.ollamaUrl ?? "http://localhost:11434");
  const [ollamaModels,  setOllamaModels]  = useState<string[]>([]);
  const [ollamaLoading, setOllamaLoading] = useState(false);
  const [ollamaError,   setOllamaError]   = useState<string | null>(null);

  // Fetch installed Ollama models whenever provider = ollama or URL changes
  useEffect(() => {
    if (config.provider !== "ollama") return;

    setOllamaLoading(true);
    setOllamaError(null);

    const url = ollamaUrl.replace(/\/$/, "");
    fetch(`${url}/api/tags`, { signal: AbortSignal.timeout(4000) })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => {
        const names: string[] = (data.models ?? []).map(
          (m: { name: string }) => m.name,
        );
        setOllamaModels(names);
        if (names.length > 0 && !names.includes(config.model)) {
          onChange({ ...config, model: names[0] });
        }
      })
      .catch(() => {
        setOllamaError("Ollama not running — start with: ollama serve");
        setOllamaModels([]);
      })
      .finally(() => setOllamaLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.provider, ollamaUrl]);

  const setProvider = (provider: LLMProvider) => {
    const model = provider === "claude" ? CLAUDE_MODELS[0] : (ollamaModels[0] ?? "");
    const updated = { ...config, provider, model };
    onChange(updated);
    setGlobalConfig(updated);
  };

  const saveAndCollapse = () => {
    const updated = { ...config, apiKey, ollamaUrl };
    onChange(updated);
    setGlobalConfig(updated);
    setExpanded(false);
  };

  const refreshOllama = () => {
    setOllamaModels([]);
    setOllamaError(null);
    setOllamaLoading(true);
    const url = ollamaUrl.replace(/\/$/, "");
    fetch(`${url}/api/tags`, { signal: AbortSignal.timeout(4000) })
      .then((r) => r.json())
      .then((data) => {
        const names: string[] = (data.models ?? []).map(
          (m: { name: string }) => m.name,
        );
        setOllamaModels(names);
        if (names.length > 0) onChange({ ...config, ollamaUrl, model: names[0] });
      })
      .catch(() => setOllamaError("Ollama not running"))
      .finally(() => setOllamaLoading(false));
  };

  return (
    <div className="border-b border-border">
      {/* Collapsed bar */}
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center gap-2.5 px-4 py-2.5 hover:bg-bg3 transition-colors duration-100 text-left"
      >
        <span className="font-mono text-[9px] text-text-dim tracking-widest uppercase">
          Model
        </span>

        {/* Provider pill */}
        <span
          className={[
            "px-2 py-0.5 rounded-full font-mono text-[9px] border",
            config.provider === "claude"
              ? "border-accent/40  text-accent  bg-accent/8"
              : "border-accent2/40 text-accent2 bg-accent2/8",
          ].join(" ")}
        >
          {config.provider === "claude" ? "Claude" : "Ollama"}
        </span>

        <span className="font-mono text-[10px] text-text-mid truncate max-w-[120px]">
          {config.model || (ollamaLoading ? "loading…" : "—")}
        </span>

        {/* Status dot — reflects actual connection state */}
        {(() => {
          if (config.provider === "claude") {
            const ready = !!config.apiKey && config.apiKey.startsWith("sk-ant-");
            return (
              <span
                className="ml-auto w-1.5 h-1.5 rounded-full flex-shrink-0"
                title={ready ? "API key configured" : "API key not set"}
                style={
                  ready
                    ? { background: "#00ff88", boxShadow: "0 0 4px #00ff88" }
                    : { background: "#555", boxShadow: "none" }
                }
              />
            );
          } else {
            if (ollamaLoading) {
              return (
                <span
                  className="ml-auto w-1.5 h-1.5 rounded-full flex-shrink-0 animate-pulse"
                  title="Connecting to Ollama…"
                  style={{ background: "#f5a623", boxShadow: "0 0 4px #f5a623" }}
                />
              );
            }
            if (ollamaError || ollamaModels.length === 0) {
              return (
                <span
                  className="ml-auto w-1.5 h-1.5 rounded-full flex-shrink-0"
                  title={ollamaError ?? "No models installed"}
                  style={{ background: "#ff4444", boxShadow: "0 0 4px #ff4444" }}
                />
              );
            }
            return (
              <span
                className="ml-auto w-1.5 h-1.5 rounded-full flex-shrink-0"
                title="Ollama connected"
                style={{ background: "#00ff88", boxShadow: "0 0 4px #00ff88" }}
              />
            );
          }
        })()}

        <span className="text-text-dim text-[10px]">{expanded ? "▲" : "▼"}</span>
      </button>

      {/* Expanded settings */}
      {expanded && (
        <div className="px-4 pb-4 pt-2 flex flex-col gap-3 bg-bg2 border-t border-border">

          {/* Provider toggle */}
          <div>
            <p className="font-mono text-[9px] text-text-dim tracking-widest mb-1.5 uppercase">Provider</p>
            <div className="flex gap-2">
              {(["claude", "ollama"] as LLMProvider[]).map((p) => (
                <button
                  key={p}
                  onClick={() => setProvider(p)}
                  className={[
                    "px-3 py-1 rounded-sm font-mono text-[10px] border transition-all duration-100 capitalize",
                    config.provider === p
                      ? "border-accent text-accent bg-accent/8"
                      : "border-border text-text-dim hover:border-border2 hover:text-text",
                  ].join(" ")}
                >
                  {p === "claude" ? "Claude" : "Ollama (local)"}
                </button>
              ))}
            </div>
          </div>

          {/* ── Claude section ── */}
          {config.provider === "claude" && (
            <>
              <div>
                <p className="font-mono text-[9px] text-text-dim tracking-widest mb-1.5 uppercase">Model</p>
                <Dropdown
                  value={config.model}
                  options={CLAUDE_MODELS}
                  onSelect={(m) => onChange({ ...config, model: m })}
                  accent="accent"
                />
              </div>

              <div>
                <p className="font-mono text-[9px] text-text-dim tracking-widest mb-1.5 uppercase">API Key</p>
                <div className="flex gap-2">
                  <input
                    type="password"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder="sk-ant-••••••••"
                    className="flex-1 bg-bg3 border border-border rounded-sm px-3 py-1.5 font-mono text-[11px] text-text placeholder:text-text-dim outline-none focus:border-accent"
                  />
                  <button
                    onClick={saveAndCollapse}
                    className="px-3 py-1.5 border border-accent text-accent font-mono text-[10px] rounded-sm hover:bg-accent/10 transition-colors"
                  >
                    Save
                  </button>
                </div>
              </div>
            </>
          )}

          {/* ── Ollama section ── */}
          {config.provider === "ollama" && (
            <>
              <div>
                <p className="font-mono text-[9px] text-text-dim tracking-widest mb-1.5 uppercase">
                  Ollama URL
                </p>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={ollamaUrl}
                    onChange={(e) => setOllamaUrl(e.target.value)}
                    className="flex-1 bg-bg3 border border-border rounded-sm px-3 py-1.5 font-mono text-[11px] text-text outline-none focus:border-accent2"
                  />
                  <button
                    onClick={refreshOllama}
                    className="px-3 py-1.5 border border-accent2 text-accent2 font-mono text-[10px] rounded-sm hover:bg-accent2/10 transition-colors whitespace-nowrap"
                  >
                    ↺ Refresh
                  </button>
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <p className="font-mono text-[9px] text-text-dim tracking-widest uppercase">
                    Installed Models
                  </p>
                  {ollamaLoading && (
                    <span className="font-mono text-[9px] text-accent2 animate-pulse">loading…</span>
                  )}
                  {!ollamaLoading && !ollamaError && ollamaModels.length > 0 && (
                    <span className="font-mono text-[9px] text-text-dim">{ollamaModels.length} models</span>
                  )}
                </div>

                {/* Error state */}
                {ollamaError && (
                  <div className="bg-neg/10 border border-neg/30 rounded-sm px-3 py-2">
                    <p className="font-mono text-[10px] text-neg">{ollamaError}</p>
                    <p className="font-mono text-[9px] text-text-dim mt-0.5">
                      ollama serve — then click ↺ Refresh
                    </p>
                  </div>
                )}

                {/* Empty state */}
                {!ollamaLoading && !ollamaError && ollamaModels.length === 0 && (
                  <div className="bg-bg3 border border-border rounded-sm px-3 py-2">
                    <p className="font-mono text-[10px] text-text-dim">No models installed.</p>
                    <p className="font-mono text-[9px] text-text-dim mt-0.5">ollama pull gemma3:27b</p>
                  </div>
                )}

                {/* Custom model dropdown */}
                {ollamaModels.length > 0 && (
                  <Dropdown
                    value={config.model}
                    options={ollamaModels}
                    onSelect={(m) => onChange({ ...config, model: m })}
                    accent="accent2"
                  />
                )}
              </div>

              <button
                onClick={saveAndCollapse}
                className="px-3 py-1.5 border border-accent2 text-accent2 font-mono text-[10px] rounded-sm hover:bg-accent2/10 transition-colors self-end"
              >
                Save
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}
