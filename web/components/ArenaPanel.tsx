"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useI18n } from "@/i18n/context";
import type { Citation } from "@/lib/types";
import { createMarkdownComponents, postProcessAssistantContent } from "@/lib/markdown";

interface ArenaRoundData {
  query: string;
  response: string;
}

interface ArenaPanelProps {
  label: string;
  rounds: ArenaRoundData[];
  currentQuery?: string;
  currentResponse: string;
  isStreaming: boolean;
  error?: string | null;
  revealed?: boolean;
  revealLabel?: string | null;
  citations?: Citation[];
}

export function ArenaPanel({
  label,
  rounds,
  currentQuery,
  currentResponse,
  isStreaming,
  error,
  revealed = false,
  revealLabel = null,
  citations = [],
}: ArenaPanelProps) {
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const { t } = useI18n();
  const [activeLoadingMessageIndex, setActiveLoadingMessageIndex] = useState(0);
  const [isUserScrolledUp, setIsUserScrolledUp] = useState(false);

  const processContent = useCallback((text: string) => {
    if (!text) return "";
    let cleaned = text.replace(/\[\d+\]/g, "");
    cleaned = cleaned.replace(/  +/g, " ");
    cleaned = cleaned.replace(/^\s*(?:Answer|\u7b54\u6848|\u7b54)\s*[:：]\s*/i, "");
    return postProcessAssistantContent(cleaned);
  }, []);

  const processedCurrentContent = useMemo(() => processContent(currentResponse), [currentResponse, processContent]);

  const markdownComponents = useMemo(() => createMarkdownComponents([]), []);

  const loadingMessages = useMemo(() => [
    t.arena.loadingRetrieving ?? "Retrieving...",
    t.arena.loadingGenerating ?? "Generating...",
    t.arena.loadingPreparing ?? "Preparing...",
  ], [t]);

  useEffect(() => {
    if (isStreaming && !currentResponse && !error) {
      const interval = window.setInterval(() => {
        setActiveLoadingMessageIndex((prev) => (prev + 1) % loadingMessages.length);
      }, 1400);
      return () => window.clearInterval(interval);
    }
  }, [currentResponse, error, isStreaming, loadingMessages.length]);

  const isNearBottom = useCallback(() => {
    const el = scrollContainerRef.current;
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  }, []);

  const handleScroll = useCallback(() => {
    setIsUserScrolledUp(!isNearBottom());
  }, [isNearBottom]);

  const scrollToBottom = useCallback(() => {
    setIsUserScrolledUp(false);
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    if (isStreaming && !isUserScrolledUp && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [currentResponse, isStreaming, isUserScrolledUp]);
  const badgeColor =
    revealLabel === t.arena.ragEnhanced
      ? "border border-primary/40 bg-primary/20 text-primary"
      : "border border-gray-600/40 bg-gray-600/20 text-gray-300";

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-lg border border-gray-700 bg-sidebar-dark">
      <div className="flex shrink-0 items-center justify-between border-b border-gray-700 bg-sidebar-dark px-4 py-3">
        <h2 className="text-sm font-semibold text-gray-300">{label}</h2>
        {revealed && revealLabel && (
          <span className={`rounded-full px-2 py-1 text-xs font-medium ${badgeColor}`}>{revealLabel}</span>
        )}
      </div>

      <div ref={scrollContainerRef} onScroll={handleScroll} className="relative flex-1 space-y-2 overflow-y-auto p-4">
        {rounds.map((round, i) => (
          <div key={i} className="space-y-4">
            <div className="flex justify-end">
              <div className="max-w-[85%] rounded-2xl rounded-tr-sm bg-primary/15 px-4 py-2.5 text-sm text-parchment">
                {round.query}
              </div>
            </div>
            <div className="flex justify-start">
              <div className="max-w-[95%] relative overflow-hidden rounded-2xl rounded-tl-sm border border-[#e3dac3] bg-parchment p-5 shadow-sm opacity-90">
                <div className="prose prose-sm max-w-none font-serif text-parchment-text">
                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                    {processContent(round.response)}
                  </ReactMarkdown>
                </div>
              </div>
            </div>
          </div>
        ))}

        {(currentQuery || currentResponse || isStreaming || error) && (
          <div className="space-y-4 mt-6 pt-4 border-t border-white/5">
            {currentQuery && (
              <div className="flex justify-end">
                <div className="max-w-[85%] rounded-2xl rounded-tr-sm bg-primary/15 px-4 py-2.5 text-sm text-parchment">
                  {currentQuery}
                </div>
              </div>
            )}

            {error ? (
              <div className="flex justify-start">
                <div className="max-w-[95%] rounded-lg border border-red-800 bg-red-900/20 p-3 text-sm text-red-400">
                  ⚠️ {error}
                </div>
              </div>
            ) : currentResponse ? (
              <div className="flex justify-start">
                <div className="max-w-[95%] relative overflow-hidden rounded-2xl rounded-tl-sm border border-[#e3dac3] bg-parchment p-5 shadow-lg">
                  <div className="prose prose-sm max-w-none font-serif text-parchment-text">
                    <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                      {processedCurrentContent}
                    </ReactMarkdown>
                  </div>
                  {isStreaming && <span className="mt-2 inline-block h-4 w-2 animate-pulse rounded-sm bg-primary align-middle" />}
                </div>
              </div>
            ) : isStreaming ? (
              <div className="flex justify-start">
                <div className="flex min-h-[60px] max-w-[95%] flex-col items-center justify-center rounded-2xl rounded-tl-sm border border-primary/10 bg-background-dark/40 px-6 py-4 text-center">
                  <div className="mb-2 flex items-center gap-2">
                    <span className="h-2 w-2 animate-bounce rounded-full bg-primary [animation-delay:-0.3s]" />
                    <span className="h-2 w-2 animate-bounce rounded-full bg-primary [animation-delay:-0.15s]" />
                    <span className="h-2 w-2 animate-bounce rounded-full bg-primary" />
                  </div>
                  <p className="text-xs font-medium text-parchment/60">{loadingMessages[activeLoadingMessageIndex]}</p>
                </div>
              </div>
            ) : (
              <div className="flex justify-start">
                <p className="text-sm italic text-gray-500 px-2 py-1">{t.arena.awaitingResponse}</p>
              </div>
            )}
          </div>
        )}
        <div ref={bottomRef} />
        {isUserScrolledUp && isStreaming && (
          <button
            onClick={scrollToBottom}
            className="sticky bottom-2 left-1/2 -translate-x-1/2 rounded-full bg-sidebar-dark/90 border border-white/10 px-3 py-1.5 text-xs text-parchment/70 hover:text-primary shadow-lg backdrop-blur-sm transition-all z-10"
            aria-label={t.common.scrollToBottom}
          >
            ↓
          </button>
        )}
      </div>

      {revealed && revealLabel === t.arena.ragEnhanced && citations.length > 0 && (
        <div className="shrink-0 border-t border-gray-700 bg-sidebar-dark px-4 py-3">
          <p className="mb-2 text-xs font-semibold text-primary">{t.citation.citations}</p>
          <ul className="space-y-1">
            {citations.slice(0, 5).map((c, i) => (
              <li key={i} className="text-xs text-gray-400">
                [{i + 1}] {c.type === "text" ? c.source : c.type === "graph" ? c.fact : ""}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
