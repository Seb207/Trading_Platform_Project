"use client";

import { useLLM } from "@/context/LLMContext";

export default function TopBar() {
  const { config } = useLLM();
  const label = `${config.provider === "claude" ? "Claude" : "Ollama"}: ${config.model}`;

  return (
    <header className="flex items-center justify-between px-6 h-[52px] bg-bg border-b border-border flex-shrink-0">
      <div className="font-mono text-[15px] font-bold text-accent tracking-[2px]">
        QUANT<span className="text-text-dim">/</span>RESEARCH
      </div>
      <div className="flex items-center gap-3">
        <span className="font-mono text-[11px] text-text-dim tracking-wide">
          LLM: {label}
        </span>
        <span
          className="w-2 h-2 rounded-full bg-accent flex-shrink-0"
          style={{ boxShadow: "0 0 6px #00ff88" }}
        />
      </div>
    </header>
  );
}
