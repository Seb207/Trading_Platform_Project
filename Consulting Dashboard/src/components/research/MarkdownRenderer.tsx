"use client";

/**
 * MarkdownRenderer
 * Renders markdown text with LaTeX math support.
 *
 * Inline math  : $...$
 * Display math : $$...$$  or  \[...\]
 *
 * Uses react-markdown + remark-math + rehype-katex.
 * Only imported/rendered when actually needed (open accordion section),
 * so the KaTeX bundle is code-split per-section — memory-friendly.
 */

import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import type { Components } from "react-markdown";

// ── Custom renderers matched to dashboard design tokens ─────────────────
const components: Components = {
  // Paragraphs
  p: ({ children }) => (
    <p className="text-[11px] text-text-mid leading-relaxed mb-2 last:mb-0">
      {children}
    </p>
  ),

  // Headings (sub-headings inside a section)
  h1: ({ children }) => (
    <h1 className="font-mono text-[12px] text-text font-semibold mt-3 mb-1">{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 className="font-mono text-[11px] text-text font-semibold mt-3 mb-1">{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 className="font-mono text-[10px] text-text-mid font-semibold mt-2 mb-1">{children}</h3>
  ),

  // Inline code
  code: ({ children, className }) => {
    // Block code (has a language class like "language-python")
    const isBlock = !!className;
    if (isBlock) {
      return (
        <code className="block bg-bg border border-border rounded-sm px-3 py-2 font-mono text-[10px] text-accent overflow-x-auto whitespace-pre leading-relaxed my-2">
          {children}
        </code>
      );
    }
    return (
      <code className="bg-bg3 border border-border/60 rounded px-1 py-0.5 font-mono text-[10px] text-accent2">
        {children}
      </code>
    );
  },

  // Pre (wraps block code)
  pre: ({ children }) => (
    <pre className="overflow-x-auto my-2">{children}</pre>
  ),

  // Lists
  ul: ({ children }) => (
    <ul className="list-disc list-inside text-[11px] text-text-mid leading-relaxed mb-2 space-y-0.5 pl-2">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="list-decimal list-inside text-[11px] text-text-mid leading-relaxed mb-2 space-y-0.5 pl-2">
      {children}
    </ol>
  ),
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,

  // Blockquote
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-accent/30 pl-3 my-2 text-[10px] text-text-dim italic">
      {children}
    </blockquote>
  ),

  // Horizontal rule
  hr: () => <hr className="border-border my-3" />,

  // Strong / em
  strong: ({ children }) => (
    <strong className="text-text font-semibold">{children}</strong>
  ),
  em: ({ children }) => (
    <em className="text-text-mid italic">{children}</em>
  ),

  // Links (no navigation — just styled)
  a: ({ children, href }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-accent2 underline underline-offset-2 hover:text-accent transition-colors text-[11px]"
    >
      {children}
    </a>
  ),

  // Tables
  table: ({ children }) => (
    <div className="overflow-x-auto my-2">
      <table className="w-full border-collapse font-mono text-[10px] text-text-mid">
        {children}
      </table>
    </div>
  ),
  thead: ({ children }) => (
    <thead className="border-b border-border">{children}</thead>
  ),
  th: ({ children }) => (
    <th className="text-left px-2 py-1 text-text font-semibold">{children}</th>
  ),
  td: ({ children }) => (
    <td className="px-2 py-1 border-b border-border/30">{children}</td>
  ),
};

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

export default function MarkdownRenderer({ content, className }: MarkdownRendererProps) {
  return (
    <div className={className}>
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
