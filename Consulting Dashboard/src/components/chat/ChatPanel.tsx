"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ModelSelector from "./ModelSelector";
import ChatMessage from "./ChatMessage";
import ChatInput from "./ChatInput";
import MarkdownRenderer from "@/components/research/MarkdownRenderer";
import { analyzePaper } from "@/lib/api";
import type { AnalyzeResult } from "@/lib/api";
import type { ChatMessage as ChatMessageType, LLMConfig, Paper } from "@/lib/types";

type ChatTab = "chat" | "viewer" | "strategy";

// ── Section parser ─────────────────────────────────────────────────────
interface ParsedSection {
  title:   string;
  level:   number;   // 1 = #, 2 = ##, 3 = ###
  content: string;
  chars:   number;
}

function parseSections(fullContent: string): ParsedSection[] {
  // Split on markdown headings (# / ## / ###)
  const lines   = fullContent.split("\n");
  const sections: ParsedSection[] = [];
  let cur: ParsedSection | null   = null;

  for (const line of lines) {
    const m = line.match(/^(#{1,3})\s+(.+)/);
    if (m) {
      if (cur) {
        cur.content = cur.content.trimEnd();
        cur.chars   = cur.content.length;
        sections.push(cur);
      }
      cur = { title: m[2].trim(), level: m[1].length, content: "", chars: 0 };
    } else if (cur) {
      cur.content += line + "\n";
    }
  }
  if (cur) {
    cur.content = cur.content.trimEnd();
    cur.chars   = cur.content.length;
    sections.push(cur);
  }
  return sections;
}

// ── Section accordion item ─────────────────────────────────────────────
function SectionItem({
  section,
  index,
  isOpen,
  onToggle,
}: {
  section:  ParsedSection;
  index:    number;
  isOpen:   boolean;
  onToggle: (i: number) => void;
}) {
  const sizeLabel =
    section.chars > 1000
      ? `${(section.chars / 1000).toFixed(1)}k`
      : `${section.chars}`;

  return (
    <div className="border-b border-border/50 last:border-b-0">
      {/* Header row — always rendered */}
      <button
        type="button"
        onClick={() => onToggle(index)}
        className="w-full flex items-center gap-2 py-2 px-0 text-left group transition-colors duration-75 hover:bg-bg3/40"
      >
        {/* Level indent */}
        {section.level > 1 && (
          <span
            className="flex-shrink-0 w-px self-stretch bg-border/60"
            style={{ marginLeft: (section.level - 1) * 10 }}
          />
        )}

        {/* Chevron */}
        <svg
          width="10" height="10" viewBox="0 0 10 10" fill="none"
          className="flex-shrink-0 transition-transform duration-150"
          style={{ transform: isOpen ? "rotate(90deg)" : "rotate(0deg)" }}
        >
          <path d="M3 2L7 5L3 8" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"
            className={isOpen ? "text-accent" : "text-text-dim group-hover:text-text-mid"} />
        </svg>

        {/* Title */}
        <span
          className={[
            "flex-1 font-mono truncate transition-colors duration-75",
            section.level === 1 ? "text-[11px] text-text" : "text-[10px] text-text-mid",
            isOpen ? "text-accent" : "group-hover:text-text",
          ].join(" ")}
        >
          {section.title}
        </span>

        {/* Size badge */}
        <span className="flex-shrink-0 font-mono text-[9px] text-text-dim">{sizeLabel}</span>
      </button>

      {/* Content — only mounted when open (memory-efficient) */}
      {isOpen && section.content && (
        <div className="pb-3 pl-4 pr-2">
          <MarkdownRenderer content={section.content} />
        </div>
      )}
    </div>
  );
}

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
  contentLevel: "abstract" | "full",
  task: string,
  onChunk: (chunk: string) => void,
  onDone:  (paperRefs: string[]) => void,
  onError: (msg: string) => void,
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
  const [paperDetail,   setPaperDetail]   = useState<AnalyzeResult | null>(null);
  const [viewerLoading, setViewerLoading] = useState(false);
  const [openSections,  setOpenSections]  = useState<Set<number>>(new Set());
  const [contentLevel,  setContentLevel]  = useState<"abstract" | "full">("full");

  // Task modes (prompts/tasks/*). "" = no mode → no task prompt injected.
  const [task,      setTask]      = useState<string>("");
  const [taskList,  setTaskList]  = useState<{ id: string; label: string }[]>([]);

  // Fetch available task-mode prompts once on mount.
  useEffect(() => {
    fetch(`${BASE_URL}/api/chat/tasks`)
      .then((r) => r.json())
      .then((d) => setTaskList(d.tasks ?? []))
      .catch(() => setTaskList([]));
  }, []);

  // Reset accordion when paper changes
  useEffect(() => { setOpenSections(new Set()); }, [selectedPaper?.arxiv_id]);

  // Parse sections from full_content — memoized so re-renders don't re-parse
  const parsedSections = useMemo(
    () => (paperDetail?.full_content ? parseSections(paperDetail.full_content) : []),
    [paperDetail?.full_content],
  );

  const toggleSection = useCallback((i: number) => {
    setOpenSections((prev) => {
      const next = new Set(prev);
      next.has(i) ? next.delete(i) : next.add(i);
      return next;
    });
  }, []);

  const expandAll  = useCallback(() => setOpenSections(new Set(parsedSections.map((_, i) => i))), [parsedSections]);
  const collapseAll = useCallback(() => setOpenSections(new Set()), []);
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
      contentLevel,
      task,
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

          {/* Task-mode selector — "None" = no task prompt injected */}
          <div className="flex-shrink-0 flex items-center gap-2 px-4 py-2 border-b border-border bg-bg2">
            <span className="font-mono text-[9px] text-text-dim uppercase tracking-wide flex-shrink-0">Mode:</span>
            <div className="flex items-center gap-1 flex-wrap">
              <button
                onClick={() => setTask("")}
                title="No task prompt — answer normally"
                className={[
                  "px-2 py-0.5 rounded-sm font-mono text-[9px] border transition-colors",
                  task === ""
                    ? "border-accent text-accent bg-accent/10"
                    : "border-border text-text-dim hover:border-border2 hover:text-text",
                ].join(" ")}
              >
                None
              </button>
              {taskList.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setTask(t.id)}
                  title={`Inject the "${t.label}" task prompt`}
                  className={[
                    "px-2 py-0.5 rounded-sm font-mono text-[9px] border transition-colors",
                    task === t.id
                      ? "border-accent text-accent bg-accent/10"
                      : "border-border text-text-dim hover:border-border2 hover:text-text",
                  ].join(" ")}
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>

          {/* Paper context badge + grounding level toggle */}
          {selectedPaper && (
            <div className="flex-shrink-0 flex items-center gap-2 px-4 py-2 border-b border-border bg-accent/5">
              <span className="font-mono text-[9px] text-text-dim uppercase tracking-wide">Context:</span>
              <span className="font-mono text-[9px] text-accent">{selectedPaper.arxiv_id}</span>
              <span className="text-[10px] text-text-mid truncate flex-1">
                {selectedPaper.title.slice(0, 30)}…
              </span>

              {/* Abstract / Full-paper grounding toggle */}
              <div className="flex items-center gap-0.5 flex-shrink-0 border border-border rounded-sm overflow-hidden">
                {(["abstract", "full"] as const).map((lvl) => (
                  <button
                    key={lvl}
                    onClick={() => setContentLevel(lvl)}
                    title={
                      lvl === "full"
                        ? "Inject full paper text into the LLM (higher token cost, deeper answers)"
                        : "Inject only the abstract (cheaper, quick questions)"
                    }
                    className={[
                      "px-2 py-0.5 font-mono text-[8px] uppercase tracking-wide transition-colors",
                      contentLevel === lvl
                        ? "bg-accent/15 text-accent"
                        : "text-text-dim hover:text-text",
                    ].join(" ")}
                  >
                    {lvl === "full" ? "Full" : "Abstract"}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-5">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full gap-3 text-center">
                <span className="font-mono text-[10px] text-text-dim tracking-widest uppercase">Ready</span>
                <p className="text-text-dim text-[11px] leading-relaxed max-w-[220px]">
                  {selectedPaper
                    ? `Paper ${selectedPaper.arxiv_id} selected.\nAsk questions or request strategy code.`
                    : "Select a paper from the list on the left to start."}
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
        <div className="flex flex-col flex-1 overflow-hidden">
          {/* No paper selected */}
          {!selectedPaper && (
            <div className="flex flex-col items-center justify-center flex-1 gap-2">
              <span className="font-mono text-[10px] text-text-dim uppercase tracking-widest">No Paper Selected</span>
              <p className="text-text-dim text-[11px]">Click a paper in the table to view it here.</p>
            </div>
          )}

          {/* Loading skeleton */}
          {selectedPaper && viewerLoading && (
            <div className="flex flex-col gap-3 px-5 py-4 animate-pulse">
              <div className="h-3 bg-bg3 rounded w-1/3" />
              <div className="h-4 bg-bg3 rounded w-4/5" />
              <div className="h-4 bg-bg3 rounded w-3/5" />
              <div className="mt-4 h-3 bg-bg3 rounded w-full" />
              <div className="h-3 bg-bg3 rounded w-full" />
              <div className="h-3 bg-bg3 rounded w-2/3" />
            </div>
          )}

          {selectedPaper && !viewerLoading && (
            <div className="flex flex-col flex-1 overflow-y-auto">
              {/* Sticky header */}
              <div className="flex-shrink-0 px-5 pt-4 pb-3 border-b border-border bg-bg">
                <span className="font-mono text-[9px] text-accent2 tracking-widest uppercase">
                  {selectedPaper.arxiv_id} · {selectedPaper.category}
                </span>
                <h2 className="text-[13px] text-text font-medium leading-snug mt-1">
                  {selectedPaper.title}
                </h2>

                {/* Abstract — always visible in header area */}
                {selectedPaper.abstract && (
                  <p className="mt-2 text-[10px] text-text-mid leading-relaxed line-clamp-3">
                    {selectedPaper.abstract}
                  </p>
                )}
              </div>

              {/* Sections accordion */}
              {parsedSections.length > 0 && (
                <div className="flex flex-col flex-1 overflow-y-auto">
                  {/* Toolbar */}
                  <div className="flex items-center justify-between px-5 py-1.5 border-b border-border/50 bg-bg2 flex-shrink-0">
                    <span className="font-mono text-[9px] text-text-dim tracking-widest uppercase">
                      {parsedSections.length} sections · {openSections.size} open
                    </span>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        onClick={expandAll}
                        className="font-mono text-[9px] text-text-dim hover:text-accent transition-colors"
                      >
                        expand all
                      </button>
                      <span className="text-text-dim text-[9px]">/</span>
                      <button
                        type="button"
                        onClick={collapseAll}
                        className="font-mono text-[9px] text-text-dim hover:text-accent transition-colors"
                      >
                        collapse all
                      </button>
                    </div>
                  </div>

                  {/* Section list */}
                  <div className="px-5 py-1 overflow-y-auto">
                    {parsedSections.map((section, i) => (
                      <SectionItem
                        key={i}
                        section={section}
                        index={i}
                        isOpen={openSections.has(i)}
                        onToggle={toggleSection}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* No markdown available */}
              {!paperDetail && (
                <p className="px-5 py-4 text-[11px] text-text-dim">
                  Section content unavailable — PDF-only paper or not yet downloaded as Markdown.
                </p>
              )}

              {/* Markdown fetched but no sections parsed */}
              {paperDetail && parsedSections.length === 0 && (
                <div className="px-5 py-4">
                  <span className="font-mono text-[9px] text-text-dim tracking-widest uppercase mb-2 block">Full Content</span>
                  <MarkdownRenderer content={paperDetail.full_content} />
                </div>
              )}
            </div>
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
