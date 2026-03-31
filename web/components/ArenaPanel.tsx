"use client";

import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Citation } from "@/lib/types";

interface ArenaPanelProps {
  label: string;          // "Model A" | "Model B"
  content: string;        // streaming/completed text
  isStreaming: boolean;
  error?: string | null;
  // Post-reveal props (all optional until reveal)
  revealed?: boolean;
  revealLabel?: "RAG Enhanced" | "Plain LLM" | null;
  citations?: Citation[];
}

export function ArenaPanel({
  label,
  content,
  isStreaming,
  error,
  revealed = false,
  revealLabel = null,
  citations = [],
}: ArenaPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll while streaming
  useEffect(() => {
    if (isStreaming && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [content, isStreaming]);

  const badgeColor =
    revealLabel === "RAG Enhanced"
      ? "bg-[#19e6d4]/20 text-[#19e6d4] border border-[#19e6d4]/40"
      : "bg-gray-600/20 text-gray-300 border border-gray-600/40";

  return (
    <div className="flex-1 flex flex-col bg-[#0d0d1a] rounded-lg border border-gray-700 overflow-hidden min-h-0">
      {/* Panel header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-700 bg-[#0a0a17] shrink-0">
        <h2 className="text-sm font-semibold text-gray-300">{label}</h2>
        {revealed && revealLabel && (
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${badgeColor}`}>
            {revealLabel}
          </span>
        )}
      </div>

      {/* Content area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {error ? (
          <div className="text-red-400 text-sm border border-red-800 rounded-lg p-3 bg-red-900/20">
            ⚠️ {error}
          </div>
        ) : content ? (
          <>
            <div className="prose prose-invert prose-sm max-w-none text-[#F3EFE0]">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
            </div>
            {isStreaming && (
              <span className="inline-block w-2 h-4 bg-[#19e6d4] animate-pulse rounded-sm align-middle" />
            )}
          </>
        ) : isStreaming ? (
          <div className="flex items-center gap-2 text-gray-500 text-sm">
            <span className="inline-block w-2 h-4 bg-[#19e6d4] animate-pulse rounded-sm" />
            <span>Generating…</span>
          </div>
        ) : (
          <p className="text-gray-500 text-sm italic">Awaiting response…</p>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Citations — shown after reveal if RAG side has citations */}
      {revealed && revealLabel === "RAG Enhanced" && citations.length > 0 && (
        <div className="border-t border-gray-700 px-4 py-3 bg-[#0a0a17] shrink-0">
          <p className="text-xs font-semibold text-[#19e6d4] mb-2">引用来源 Citations</p>
          <ul className="space-y-1">
            {citations.slice(0, 5).map((c, i) => (
              <li key={i} className="text-xs text-gray-400">
                [{i + 1}]{" "}
                {c.type === "text"
                  ? c.source
                  : c.type === "graph"
                  ? c.fact
                  : ""}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
