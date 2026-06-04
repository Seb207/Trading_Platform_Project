"use client";

import { useEffect, useRef, useState } from "react";
import ModelSelector from "./ModelSelector";
import ChatMessage from "./ChatMessage";
import ChatInput from "./ChatInput";
import { analyzePaper } from "@/lib/api";
import type { AnalyzeResult } from "@/lib/api";
import type { ChatMessage as ChatMessageType, LLMConfig, Paper } from "@/lib/types";

type ChatTab = "chat" | "viewer" | "strategy";

const TABS: { id: ChatTab; label: string }[] = [
  { id: "chat",     label: "LLM Chat"     },
  { id: "viewer",   label: "Paper Viewer" },
  { id: "strategy", label: "Strategy"     },
];

const BASE_URL = "http://localhost:8000";

// ── SSE streaming helper ───────────────────────────────────────────────
async function streamChat(
  config: LLMConfig,
  messages: { role: string; content: string }[],
  paperContext: { arxiv_id: string; relative_path: string } | null,
  onChunk: (chunk: string) => void,
  onDone:  (paperRefs: string[]) => void,
  onError: (msg: string) => void,
) {
  const body = {
    provider:   config.provider,
    model:      config.model,
    api_key:    config.apiKey ?? "",
    ollama_url: config.ollamaUrl ?? "http://localhost:11434",
    messages,
    paper_context: paperContext
      ? { arxiv_id: paperContext.arxiv_id, relative_path: paperContext.relative_path, content_level: "abstract" }
      : null,
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
        if (data.type === "chunk") onChunk(data.content);
        if (data.type === "done")  onDone(data.paper_refs ?? []);
        if (data.type === "error") onError(data.message);
      } catch { /* ignore malformed SSE */ }
    }
  }
}

// ── Component ──────────────────────────────────────────────────────────
interface ChatPanelProps {
  selectedPaper: Paper | null;
}

export default function ChatPanel({ selectedPaper }: ChatPanelProps) {
  const [activeTab,    setActiveTab]    = useState<ChatTab>("chat");
  const [messages,     setMessages]     = useState<ChatMessageType[]>([]);
  const [streaming,    setStreaming]    = useState(false);
  const [config,       setConfig]       = useState<LLMConfig>({
    provider: "claude",
    model:    "claude-opus-4-5",
  });
  const [paperDetail,  setPaperDetail]  = useState<AnalyzeResult | null>(null);
  const [viewerLoading, setViewerLoading] = useState(false);
  const bottomRef      = useRef<HTMLDivElement>(null);
  const streamingIdRef = useRef<string | null>(null); // ID of the assistant message being built

  // Auto-scroll on new content
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Fetch paper detail when viewer tab is opened or paper changes
  useEffect(() => {
    if (activeTab !== "viewer" || !selectedPaper?.relative_path) return;
    setViewerLoading(true);
    analyzePaper(selectedPaper.relative_path)
      .then(setPaperDetail)
      .catch(() => setPaperDetail(null))
      .finally(() => setViewerLoading(false));
  }, [activeTab, selectedPaper?.relative_path]);

  const handleSend = async (content: string) => {
    if (streaming) return;

    // Add user message
    const userMsg: ChatMessageType = {
      id:        crypto.randomUUID(),
      role:      "user",
      content,
      timestamp: new Date(),
      paperRefs: selectedPaper ? [selectedPaper.arxiv_id] : [],
    };
    setMessages((prev) => [...prev, userMsg]);
    setStreaming(true);

    // Create empty assistant message (will be filled by stream)
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

    // Build history for API (exclude the empty assistant message)
    const history = [...messages, userMsg].map((m) => ({
      role:    m.role,
      content: m.content,
    }));

    await streamChat(
      config,
      history,
      selectedPaper
        ? { arxiv_id: selectedPaper.arxiv_id, relative_path: selectedPaper.relative_path ?? "" }
        : null,
      // onChunk — append to streaming message
      (chunk) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, content: m.content + chunk } : m,
          ),
        );
      },
      // onDone
      (paperRefs) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, paperRefs } : m,
          ),
        );
        setStreaming(false);
        streamingIdRef.current = null;
      },
      // onError
      (errMsg) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: `⚠️ ${errMsg}` }
              : m,
          ),
        );
        setStreaming(false);
        streamingIdRef.current = null;
      },
    );
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Tab bar */}
      <div className="flex border-b border-border flex-shrink-0">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={[
              "flex-1 py-3 font-mono text-[10px] tracking-wide uppercase transition-colors duration-100",
              "border-b-2",
              activeTab === tab.id
                ? "text-accent border-accent"
                : "text-text-dim border-transparent hover:text-text",
            ].join(" ")}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* ── LLM Chat tab ── */}
      {activeTab === "chat" && (
        <>
          <div className="flex-shrink-0">
            <ModelSelector config={config} onChange={setConfig} />
          </div>

          {/* Paper context badge */}
          {selectedPaper && (
            <div className="flex-shrink-0 flex items-center gap-2 px-4 py-2 border-b border-border bg-accent/5">
              <span className="font-mono text-[9px] text-text-dim uppercase tracking-wide">Context:</span>
              <span className="font-mono text-[9px] text-accent">{selectedPaper.arxiv_id}</span>
              <span className="text-[10px] text-text-mid truncate">
                {selectedPaper.title.slice(0, 38)}…
              </span>
            </div>
          )}

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-5">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full gap-3 text-center">
                <span className="font-mono text-[10px] text-text-dim tracking-widest uppercase">Ready</span>
                <p className="text-text-dim text-[11px] leading-relaxed max-w-[220px]">
                  {selectedPaper
                    ? `논문 ${selectedPaper.arxiv_id} 선택됨.\n질문하거나 전략 코드를 요청하세요.`
                    : "좌측에서 논문을 선택 후 질문하세요."}
                </p>
              </div>
            )}

            {messages.map((msg) => (
              <ChatMessage key={msg.id} message={msg} />
            ))}

            {/* Streaming cursor */}
            {streaming && streamingIdRef.current && (
              <div className="flex items-center gap-1.5">
                {[0, 1, 2].map((i) => (
                  <span
                    key={i}
                    className="w-1 h-1 rounded-full bg-accent animate-pulse"
                    style={{ animationDelay: `${i * 150}ms` }}
                  />
                ))}
              </div>
            )}

            <div ref={bottomRef} />
          </div>

          <ChatInput onSend={handleSend} disabled={streaming} />
        </>
      )}

      {/* ── Paper Viewer tab ── */}
      {activeTab === "viewer" && (
        <div className="flex flex-col flex-1 overflow-y-auto px-5 py-4 gap-4">
          {!selectedPaper && (
            <div className="flex flex-col items-center justify-center flex-1 gap-2">
              <span className="font-mono text-[10px] text-text-dim uppercase tracking-widest">No Paper Selected</span>
              <p className="text-text-dim text-[11px]">Click a paper in the table to view it here.</p>
            </div>
          )}

          {selectedPaper && viewerLoading && (
            <div className="flex flex-col gap-3 animate-pulse">
              <div className="h-3 bg-bg3 rounded w-1/3" />
              <div className="h-4 bg-bg3 rounded w-4/5" />
              <div className="h-4 bg-bg3 rounded w-3/5" />
              <div className="mt-4 h-3 bg-bg3 rounded w-full" />
              <div className="h-3 bg-bg3 rounded w-full" />
              <div className="h-3 bg-bg3 rounded w-2/3" />
            </div>
          )}

          {selectedPaper && !viewerLoading && (
            <>
              {/* Header */}
              <div>
                <span className="font-mono text-[9px] text-accent2 tracking-widest uppercase">
                  {selectedPaper.arxiv_id} · {selectedPaper.category}
                </span>
                <h2 className="text-[13px] text-text font-medium leading-snug mt-1">
                  {selectedPaper.title}
                </h2>
              </div>

              {/* Abstract */}
              <div className="border-t border-border pt-3">
                <span className="font-mono text-[9px] text-text-dim tracking-widest uppercase mb-2 block">Abstract</span>
                <p className="text-[11px] text-text-mid leading-relaxed">
                  {selectedPaper.abstract || "Abstract not available."}
                </p>
              </div>

              {/* Section map */}
              {paperDetail && paperDetail.section_map.length > 0 && (
                <div className="border-t border-border pt-3">
                  <span className="font-mono text-[9px] text-text-dim tracking-widest uppercase mb-2 block">
                    Sections ({paperDetail.section_count})
                  </span>
                  <div className="flex flex-col gap-1">
                    {paperDetail.section_map.map((s, i) => (
                      <div key={i} className="flex items-center justify-between gap-3 py-1 border-b border-border/40">
                        <span className="text-[11px] text-text truncate">{s.title}</span>
                        <span className="font-mono text-[9px] text-text-dim flex-shrink-0">
                          {(s.char_count / 1000).toFixed(1)}k
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* No .md available */}
              {!paperDetail && (
                <p className="text-[11px] text-text-dim">
                  Section map unavailable — PDF-only paper or not yet downloaded as Markdown.
                </p>
              )}
            </>
          )}
        </div>
      )}

      {/* ── Strategy tab ── */}
      {activeTab === "strategy" && (() => {
        // Extract all code blocks from assistant messages
        const codeBlocks: { code: string; msgId: string; idx: number }[] = [];
        messages.forEach((m) => {
          if (m.role !== "assistant") return;
          const regex = /```(?:\w*)\n?([\s\S]*?)```/g;
          let match;
          let idx = 0;
          while ((match = regex.exec(m.content)) !== null) {
            codeBlocks.push({ code: match[1].trim(), msgId: m.id, idx: idx++ });
          }
        });

        return (
          <div className="flex flex-col flex-1 overflow-y-auto px-4 py-4 gap-4">
            {codeBlocks.length === 0 ? (
              <div className="flex flex-col items-center justify-center flex-1 gap-2">
                <span className="font-mono text-[10px] text-text-dim uppercase tracking-widest">No Code Yet</span>
                <p className="text-text-dim text-[11px] text-center max-w-[200px]">
                  Ask the LLM to generate strategy code in the Chat tab.
                </p>
              </div>
            ) : (
              codeBlocks.map((cb) => (
                <div key={`${cb.msgId}-${cb.idx}`} className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-[9px] text-accent tracking-widest uppercase">
                      Strategy Code
                    </span>
                    <button
                      onClick={() => navigator.clipboard.writeText(cb.code)}
                      className="font-mono text-[9px] text-text-dim hover:text-text border border-border rounded-sm px-2 py-0.5 transition-colors"
                    >
                      Copy
                    </button>
                  </div>
                  <pre className="bg-bg border border-border rounded-sm p-3 font-mono text-[10px] text-accent overflow-x-auto whitespace-pre leading-relaxed">
                    {cb.code}
                  </pre>
                </div>
              ))
            )}
          </div>
        );
      })()}
    </div>
  );
}
