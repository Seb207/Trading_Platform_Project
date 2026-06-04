import type { ChatMessage as ChatMessageType } from "@/lib/types";

// ── Simple markdown-ish renderer (no external deps) ───────────────────
function renderContent(text: string) {
  const parts: React.ReactNode[] = [];
  // Split on ``` code blocks
  const codeBlockRegex = /```(\w*)\n?([\s\S]*?)```/g;
  let lastIndex = 0;
  let match;

  while ((match = codeBlockRegex.exec(text)) !== null) {
    // Text before code block
    if (match.index > lastIndex) {
      parts.push(
        <span key={`t-${lastIndex}`}>
          {renderInline(text.slice(lastIndex, match.index))}
        </span>
      );
    }
    // Code block
    parts.push(
      <pre
        key={`code-${match.index}`}
        className="mt-2 mb-1 p-3 bg-bg border border-border rounded-sm font-mono text-[10px] text-accent overflow-x-auto whitespace-pre"
      >
        {match[2].trim()}
      </pre>
    );
    lastIndex = match.index + match[0].length;
  }

  // Remaining text
  if (lastIndex < text.length) {
    parts.push(
      <span key={`t-${lastIndex}`}>
        {renderInline(text.slice(lastIndex))}
      </span>
    );
  }

  return parts.length ? parts : renderInline(text);
}

function renderInline(text: string): React.ReactNode {
  // Handle inline `code`, **bold**, and line breaks
  const parts: React.ReactNode[] = [];
  const regex = /(`[^`]+`|\*\*[^*]+\*\*)/g;
  let last = 0;
  let m;

  while ((m = regex.exec(text)) !== null) {
    if (m.index > last) {
      parts.push(...renderLineBreaks(text.slice(last, m.index), `lb-${last}`));
    }
    if (m[0].startsWith("`")) {
      parts.push(
        <code
          key={`ic-${m.index}`}
          className="font-mono text-[10px] text-accent bg-accent/10 px-1 py-0.5 rounded-sm"
        >
          {m[0].slice(1, -1)}
        </code>
      );
    } else {
      parts.push(
        <strong key={`b-${m.index}`} className="font-semibold text-text">
          {m[0].slice(2, -2)}
        </strong>
      );
    }
    last = m.index + m[0].length;
  }

  if (last < text.length) {
    parts.push(...renderLineBreaks(text.slice(last), `lb-${last}`));
  }

  return parts.length === 1 ? parts[0] : <>{parts}</>;
}

function renderLineBreaks(text: string, keyPrefix: string): React.ReactNode[] {
  return text.split("\n").flatMap((line, i, arr) =>
    i < arr.length - 1
      ? [line, <br key={`${keyPrefix}-br-${i}`} />]
      : [line]
  );
}

// ── Component ──────────────────────────────────────────────────────────
interface ChatMessageProps {
  message: ChatMessageType;
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <div className="flex flex-col gap-1.5">
      {/* Role + timestamp */}
      <div className="flex items-center gap-2">
        <span
          className={[
            "font-mono text-[9px] tracking-widest uppercase font-medium",
            isUser ? "text-accent2" : "text-accent",
          ].join(" ")}
        >
          {isUser ? "You" : (message.modelName ?? "Research LLM")}
        </span>
        <span className="font-mono text-[9px] text-text-dim">
          {message.timestamp.toLocaleTimeString("en-US", {
            hour: "2-digit",
            minute: "2-digit",
            hour12: false,
          })}
        </span>
      </div>

      {/* Bubble */}
      <div
        className={[
          "px-3.5 py-2.5 rounded-sm text-[12px] leading-relaxed",
          "border text-text",
          isUser
            ? "bg-accent2/5 border-accent2/15"
            : "bg-bg3 border-border",
        ].join(" ")}
      >
        {renderContent(message.content)}
      </div>

      {/* Paper refs */}
      {message.paperRefs && message.paperRefs.length > 0 && (
        <div className="flex gap-1.5 flex-wrap mt-0.5">
          {message.paperRefs.map((id) => (
            <span
              key={id}
              className="font-mono text-[9px] text-accent2 border border-accent2/30 bg-accent2/5 px-1.5 py-0.5 rounded-sm"
            >
              {id}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
