"use client";

import { createContext, useContext, useRef, useState } from "react";
import type { ChatMessage as ChatMessageType, LLMConfig } from "@/lib/types";

const BASE_URL = "http://localhost:8000";

interface PaperContextArg {
  arxiv_id: string;
  relative_path: string;
}

interface SendMessageParams {
  content: string;
  config: LLMConfig;
  paperContext: PaperContextArg | null;
  contentLevel: "abstract" | "full";
  task: string;
}

interface ChatContextValue {
  messages: ChatMessageType[];
  streaming: boolean;
  sendMessage: (params: SendMessageParams) => Promise<void>;
}

const ChatContext = createContext<ChatContextValue>({
  messages: [],
  streaming: false,
  sendMessage: async () => {},
});

// ── SSE streaming helper ───────────────────────────────────────────────
// Lives here (not in ChatPanel) so the in-flight request and its callbacks
// are owned by this provider — mounted once in the root layout, it survives
// route navigation. ChatPanel itself unmounts whenever the user leaves
// /research; if this logic lived there, navigating away mid-answer would
// orphan the fetch (it'd still finish server-side, but its result would
// vanish since there'd be no live component state left to write into).
async function streamChat(
  config: LLMConfig,
  messages: { role: string; content: string }[],
  paperContext: PaperContextArg | null,
  contentLevel: "abstract" | "full",
  task: string,
  onChunk:     (chunk: string) => void,
  onVerifying: () => void,
  onVerified:  () => void,
  onRevised:   (content: string, issues: string[]) => void,
  onDone:      (paperRefs: string[]) => void,
  onError:     (msg: string) => void,
) {
  const body = {
    provider:   config.provider,
    model:      config.model,
    api_key:    config.provider === "openrouter"
      ? (config.openRouterApiKey ?? "")
      : (config.apiKey ?? ""),
    ollama_url: config.ollamaUrl ?? "http://localhost:11434",
    messages,
    paper_context: paperContext
      ? { arxiv_id: paperContext.arxiv_id, relative_path: paperContext.relative_path, content_level: contentLevel }
      : null,
    task,
    critic_api_key: config.openRouterApiKey ?? "",
  };

  let response: Response;
  try {
    response = await fetch(`${BASE_URL}/api/chat`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body),
    });
  } catch {
    onError("Backend unreachable — is FastAPI running on port 8000?");
    return;
  }

  if (!response.ok || !response.body) {
    onError(`API error ${response.status}`);
    return;
  }

  const reader  = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer    = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";   // keep incomplete last line

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      try {
        const data = JSON.parse(line.slice(6));
        if (data.type === "chunk")     onChunk(data.content);
        if (data.type === "verifying") onVerifying();
        if (data.type === "verified")  onVerified();
        if (data.type === "revised")   onRevised(data.content, data.issues ?? []);
        if (data.type === "done")      onDone(data.paper_refs ?? []);
        if (data.type === "error")     onError(data.message);
      } catch { /* ignore malformed SSE */ }
    }
  }
}

// ── Provider ─────────────────────────────────────────────────────────────
export function ChatProvider({ children }: { children: React.ReactNode }) {
  const [messages,  setMessages]  = useState<ChatMessageType[]>([]);
  const [streaming, setStreaming] = useState(false);
  const streamingIdRef = useRef<string | null>(null);

  const sendMessage = async ({ content, config, paperContext, contentLevel, task }: SendMessageParams) => {
    if (streaming) return;

    const userMsg: ChatMessageType = {
      id:        crypto.randomUUID(),
      role:      "user",
      content,
      timestamp: new Date(),
      paperRefs: paperContext ? [paperContext.arxiv_id] : [],
    };
    setMessages((prev) => [...prev, userMsg]);
    setStreaming(true);

    const assistantId = crypto.randomUUID();
    streamingIdRef.current = assistantId;
    const assistantMsg: ChatMessageType = {
      id:        assistantId,
      role:      "assistant",
      content:   "",
      timestamp: new Date(),
      modelName: config.model,
    };
    setMessages((prev) => [...prev, assistantMsg]);

    // Build history from a snapshot taken before this send (React state
    // updates above are async, so `messages` in this closure is still the
    // pre-send list — exactly what we want for the request history).
    const history = [...messages, userMsg].map((m) => ({
      role:    m.role,
      content: m.content,
    }));

    await streamChat(
      config,
      history,
      paperContext,
      contentLevel,
      task,
      (chunk) => {
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + chunk } : m)),
        );
      },
      () => {
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, verification: "verifying" } : m)),
        );
      },
      () => {
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, verification: "verified" } : m)),
        );
      },
      (revisedContent, issues) => {
        setMessages((prev) => [
          ...prev.map((m) =>
            m.id === assistantId ? { ...m, verification: "revised" as const } : m,
          ),
          {
            id:             crypto.randomUUID(),
            role:           "assistant" as const,
            content:        revisedContent,
            timestamp:      new Date(),
            modelName:      config.model,
            revisionOf:     assistantId,
            revisionIssues: issues,
          },
        ]);
      },
      (paperRefs) => {
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, paperRefs } : m)),
        );
        setStreaming(false);
        streamingIdRef.current = null;
      },
      (errMsg) => {
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, content: `⚠️ ${errMsg}` } : m)),
        );
        setStreaming(false);
        streamingIdRef.current = null;
      },
    );
  };

  return (
    <ChatContext.Provider value={{ messages, streaming, sendMessage }}>
      {children}
    </ChatContext.Provider>
  );
}

export const useChat = () => useContext(ChatContext);
