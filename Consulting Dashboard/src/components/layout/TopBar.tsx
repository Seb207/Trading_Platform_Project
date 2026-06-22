"use client";

import { useLLM } from "@/context/LLMContext";

export default function TopBar() {
  const { config } = useLLM();
  const providerLabel =
    config.provider === "claude" ? "Claude"
    : config.provider === "ollama" ? "Ollama"
    : "OpenRouter";
  const label = `${providerLabel}: ${config.model}`;

  const isReady =
    config.provider === "claude"
      ? !!config.apiKey && config.apiKey.startsWith("sk-ant-")
      : config.provider === "openrouter"
      ? !!config.openRouterApiKey
      : !!config.model; // Ollama: trust model is set

  return (
    <header className="flex items-center justify-between px-6 h-[52px] bg-bg border-b border-border flex-shrink-0">
      <div className="font-display text-[16px] font-bold text-accent tracking-[1px]">
        QUANT<span className="text-text-dim font-light">/</span>RESEARCH
      </div>
      <div className="flex items-center gap-3">
        <span className="font-mono text-[11px] text-text-dim tracking-wide">
          LLM: {label}
        </span>
        <span
          className="w-2 h-2 rounded-full flex-shrink-0"
          title={isReady ? "Ready" : config.provider === "claude" ? "API key not set" : "No model selected"}
          style={
            isReady
              ? { background: "#00ff88", boxShadow: "0 0 6px #00ff88" }
              : { background: "#555", boxShadow: "none" }
          }
        />
      </div>
    </header>
  );
}
