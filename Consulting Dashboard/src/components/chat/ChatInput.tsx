"use client";

import { useRef, useState } from "react";

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export default function ChatInput({
  onSend,
  disabled = false,
  placeholder = "Ask about papers, request strategy code...",
}: ChatInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Auto-grow textarea
  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  };

  return (
    <div className="px-4 py-3 border-t border-border flex-shrink-0">
      <div
        className={[
          "flex items-end gap-2 bg-bg3 border rounded-sm transition-colors duration-150",
          disabled ? "border-border opacity-50" : "border-border focus-within:border-accent",
        ].join(" ")}
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder={placeholder}
          rows={2}
          className="flex-1 bg-transparent resize-none outline-none px-3 py-2.5 text-[12px] text-text placeholder:text-text-dim font-sans leading-relaxed"
          style={{ minHeight: "52px", maxHeight: "160px" }}
        />
        <button
          onClick={handleSend}
          disabled={disabled || !value.trim()}
          className={[
            "m-2 w-8 h-8 rounded-sm flex-shrink-0 flex items-center justify-center transition-all duration-150",
            value.trim() && !disabled
              ? "bg-accent cursor-pointer hover:bg-accent/80"
              : "bg-bg4 cursor-not-allowed opacity-40",
          ].join(" ")}
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="#000">
            <path d="M2 21l21-9L2 3v7l15 2-15 2z" />
          </svg>
        </button>
      </div>
      <p className="font-mono text-[9px] text-text-dim mt-1.5 text-right">
        Enter to send · Shift+Enter for newline
      </p>
    </div>
  );
}
