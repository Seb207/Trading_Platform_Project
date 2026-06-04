"use client";

import { useEffect, useState } from "react";
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
        // Auto-select first model if current model isn't in the list
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
    setGlobalConfig(updated);   // sync TopBar
  };

  const saveAndCollapse = () => {
    const updated = { ...config, apiKey, ollamaUrl };
    onChange(updated);
    setGlobalConfig(updated);   // sync TopBar
    setExpanded(false);
  };

  const models = config.provider === "claude" ? CLAUDE_MODELS : ollamaModels;

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

        {/* Status dot */}
        <span
          className="ml-auto w-1.5 h-1.5 rounded-full flex-shrink-0"
          style={{ background: "#00ff88", boxShadow: "0 0 4px #00ff88" }}
        />
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
                <select
                  value={config.model}
                  onChange={(e) => onChange({ ...config, model: e.target.value })}
                  className="w-full bg-bg3 border border-border rounded-sm px-3 py-1.5 font-mono text-[11px] text-text outline-none focus:border-accent"
                >
                  {CLAUDE_MODELS.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
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
              {/* Ollama server URL */}
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
                    onClick={() => {
                      // Manually trigger re-fetch by toggling a temp state
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
                    }}
                    className="px-3 py-1.5 border border-accent2 text-accent2 font-mono text-[10px] rounded-sm hover:bg-accent2/10 transition-colors whitespace-nowrap"
                  >
                    ↺ Refresh
                  </button>
                </div>
              </div>

              {/* Model selector — local models only */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <p className="font-mono text-[9px] text-text-dim tracking-widest uppercase">
                    Installed Models
                  </p>
                  {ollamaLoading && (
                    <span className="font-mono text-[9px] text-accent2 animate-pulse">
                      loading…
                    </span>
                  )}
                  {!ollamaLoading && !ollamaError && ollamaModels.length > 0 && (
                    <span className="font-mono text-[9px] text-text-dim">
                      {ollamaModels.length} models
                    </span>
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
                    <p className="font-mono text-[10px] text-text-dim">
                      No models installed.
                    </p>
                    <p className="font-mono text-[9px] text-text-dim mt-0.5">
                      ollama pull gemma3:27b
                    </p>
                  </div>
                )}

                {/* Model list */}
                {ollamaModels.length > 0 && (
                  <select
                    value={config.model}
                    onChange={(e) => onChange({ ...config, model: e.target.value })}
                    className="w-full bg-bg3 border border-border rounded-sm px-3 py-1.5 font-mono text-[11px] text-text outline-none focus:border-accent2"
                  >
                    {ollamaModels.map((m) => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                  </select>
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
