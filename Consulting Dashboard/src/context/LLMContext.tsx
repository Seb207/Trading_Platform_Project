"use client";

import { createContext, useContext, useState } from "react";
import type { LLMConfig } from "@/lib/types";

interface LLMContextValue {
  config: LLMConfig;
  setConfig: (c: LLMConfig) => void;
}

const LLMContext = createContext<LLMContextValue>({
  config:    { provider: "claude", model: "claude-opus-4-5" },
  setConfig: () => {},
});

export function LLMProvider({ children }: { children: React.ReactNode }) {
  const [config, setConfig] = useState<LLMConfig>({
    provider: "claude",
    model:    "claude-opus-4-5",
  });
  return (
    <LLMContext.Provider value={{ config, setConfig }}>
      {children}
    </LLMContext.Provider>
  );
}

export const useLLM = () => useContext(LLMContext);
