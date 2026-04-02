"use client";

import Link from "next/link";
import { ExternalLink } from "lucide-react";
import type { Citation, GraphCitation, TextCitation } from "@/lib/types";
import { useState } from "react";

type VoteOption = "a" | "b" | "tie";

interface ArenaRoundVote {
  roundNumber: number;
  query: string;
  responseA: string;
  responseB: string;
  positionMapping: Record<string, string>; // e.g. {"a": "rag", "b": "plain"}
  vote: VoteOption;
  comment?: string | null;
  citationsA?: Citation[];
  citationsB?: Citation[];
}

interface ArenaRevealProps {
  votes: ArenaRoundVote[];
  onReset: () => void;
}

function resolveWinner(vote: ArenaRoundVote): "rag" | "plain" | "tie" {
  if (vote.vote === "tie") return "tie";
  const voted = vote.vote; // "a" or "b"
  return (vote.positionMapping[voted] as "rag" | "plain") ?? "tie";
}

function getRagCitations(vote: ArenaRoundVote): Citation[] {
  const ragPanel = Object.entries(vote.positionMapping).find(([, v]) => v === "rag")?.[0];
  if (ragPanel === "a") return vote.citationsA ?? [];
  if (ragPanel === "b") return vote.citationsB ?? [];
  return [];
}

function TextCitationDetail({ citation }: { citation: TextCitation }) {
  return (
    <div className="space-y-2">
      <p className="text-xs font-semibold uppercase tracking-wider text-primary/80">
        Passage Content
      </p>
      <div className="rounded-md border border-primary/10 bg-sidebar-dark/30 p-3">
        <p className="border-l-2 border-primary/40 pl-3 text-sm leading-relaxed whitespace-pre-wrap text-parchment/90">
          {citation.content}
        </p>
      </div>
      {citation.chunk_id ? (
        <Link
          href={`/source/${encodeURIComponent(citation.chunk_id)}`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-primary/70 hover:text-primary inline-flex items-center gap-1 mt-2"
        >
          View full paragraph →
        </Link>
      ) : null}
      <span className="inline-flex items-center rounded bg-primary/10 px-1.5 py-0.5 text-[10px] text-primary/70">
        Rel: {citation.relevance_percent.toFixed(1)}%
      </span>
    </div>
  );
}

function GraphCitationDetail({ citation }: { citation: GraphCitation }) {
  return (
    <div className="space-y-2">
      <p className="text-xs font-semibold uppercase tracking-wider text-primary/80">
        Knowledge Graph Fact
      </p>
      <div className="rounded-md border border-primary/10 bg-sidebar-dark/30 p-3">
        <p className="text-sm leading-relaxed text-parchment/90">{citation.fact}</p>
      </div>
      <span className="inline-flex items-center rounded bg-primary/10 px-1.5 py-0.5 text-[10px] text-primary/70">
        {citation.depth}-hop
      </span>
    </div>
  );
}

export function ArenaReveal({ votes, onReset }: ArenaRevealProps) {
  const [expandedCitation, setExpandedCitation] = useState<{ round: number; index: number } | null>(null);
  const [showAllCitations, setShowAllCitations] = useState<Set<number>>(() => new Set<number>());
  const ragWins = votes.filter((v) => resolveWinner(v) === "rag").length;
  const plainWins = votes.filter((v) => resolveWinner(v) === "plain").length;
  const ties = votes.filter((v) => resolveWinner(v) === "tie").length;
  const total = votes.length;

  return (
    <div className="fixed inset-0 z-50 bg-background-dark/95 backdrop-blur-sm overflow-y-auto">
      <div className="max-w-3xl mx-auto px-4 py-8 space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-2xl font-bold text-primary">评测结果揭晓</h1>
          <p className="text-gray-400 text-sm">共 {total} 轮对话 · 盲评已结束</p>
        </div>

        {/* Summary stats */}
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="bg-primary/10 border border-primary/30 rounded-xl p-4">
            <div className="text-3xl font-bold text-primary">{ragWins}</div>
            <div className="text-xs text-gray-400 mt-1">RAG 系统胜</div>
          </div>
          <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-4">
            <div className="text-3xl font-bold text-gray-300">{ties}</div>
            <div className="text-xs text-gray-400 mt-1">平局</div>
          </div>
          <div className="bg-gray-700/20 border border-gray-700 rounded-xl p-4">
            <div className="text-3xl font-bold text-gray-400">{plainWins}</div>
            <div className="text-xs text-gray-400 mt-1">普通 LLM 胜</div>
          </div>
        </div>

        {/* Per-round breakdown */}
        <div className="space-y-4">
          <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">逐轮详情</h2>
          {votes.map((vote) => {
            const winner = resolveWinner(vote);
            const ragCitations = getRagCitations(vote);
            const isShowingAllCitations = showAllCitations.has(vote.roundNumber);
            const displayedCitations = isShowingAllCitations ? ragCitations : ragCitations.slice(0, 8);
            const moreCount = ragCitations.length - displayedCitations.length;
            const ragPanel = Object.entries(vote.positionMapping).find(([, v]) => v === "rag")?.[0]?.toUpperCase();
            const plainPanel = Object.entries(vote.positionMapping).find(([, v]) => v === "plain")?.[0]?.toUpperCase();

            return (
              <div
                key={vote.roundNumber}
                className="bg-sidebar-dark border border-gray-700 rounded-xl p-4 space-y-3"
              >
                {/* Round header */}
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-gray-500 uppercase">
                    第 {vote.roundNumber} 轮
                  </span>
                  <div className="flex gap-2 text-xs">
                    <span className="bg-primary/20 text-primary border border-primary/30 px-2 py-0.5 rounded-full">
                      {ragPanel} = RAG 增强
                    </span>
                    <span className="bg-gray-700/30 text-gray-400 border border-gray-700 px-2 py-0.5 rounded-full">
                      {plainPanel} = 普通 LLM
                    </span>
                  </div>
                </div>

                {/* Query */}
                <p className="text-sm text-gray-300 font-medium">&quot;{vote.query}&quot;</p>

                {/* Vote result */}
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500">你的选择：</span>
                  {winner === "rag" && (
                    <span className="text-xs bg-primary/20 text-primary border border-primary/40 px-2 py-0.5 rounded-full font-medium">
                      ✓ RAG 增强系统
                    </span>
                  )}
                  {winner === "plain" && (
                    <span className="text-xs bg-gray-700/30 text-gray-400 border border-gray-600 px-2 py-0.5 rounded-full">
                      普通 LLM
                    </span>
                  )}
                  {winner === "tie" && (
                    <span className="text-xs bg-gray-700/30 text-gray-400 border border-gray-600 px-2 py-0.5 rounded-full">
                      平局
                    </span>
                  )}
                  {vote.comment && (
                    <span className="text-xs text-gray-500 italic ml-1">&quot;{vote.comment}&quot;</span>
                  )}
                </div>

                {/* RAG Citations */}
                {ragCitations.length > 0 && (
                  <div className="border-t border-gray-800 pt-2 space-y-1">
                    <p className="text-xs text-primary font-medium">RAG 引用来源</p>
                    <ul className="space-y-2">
                      {displayedCitations.map((c, i) => {
                        const isExpanded =
                          expandedCitation?.round === vote.roundNumber && expandedCitation?.index === i;

                        return (
                          <li key={`${vote.roundNumber}-${i}`} className="text-xs">
                            <button
                              type="button"
                              onClick={() =>
                                setExpandedCitation(isExpanded ? null : { round: vote.roundNumber, index: i })
                              }
                              aria-expanded={isExpanded}
                              className={`flex w-full items-start gap-2 text-left cursor-pointer transition-colors ${
                                isExpanded ? "text-primary" : "text-gray-400 hover:text-primary"
                              }`}
                            >
                              <span className="shrink-0 text-primary/70">[{i + 1}]</span>
                              <span className="flex-1">
                                {c.type === "text" ? c.source : c.type === "graph" ? c.fact : ""}
                              </span>
                              <span className="shrink-0 ml-1 text-primary/50">
                                {isExpanded ? "▼" : "▶"}
                              </span>
                            </button>

                            {isExpanded && (
                              <div className="mt-1 ml-4 rounded-lg border border-primary/20 bg-background-dark/50 p-3">
                                {c.type === "text" ? (
                                  <TextCitationDetail citation={c} />
                                ) : c.type === "graph" ? (
                                  <GraphCitationDetail citation={c} />
                                ) : null}
                              </div>
                            )}
                          </li>
                        );
                      })}
                    </ul>
                    {ragCitations.length > 8 && (
                      <button
                        type="button"
                        onClick={() =>
                          setShowAllCitations((prev) => {
                            const next = new Set(prev);
                            if (next.has(vote.roundNumber)) {
                              next.delete(vote.roundNumber);
                            } else {
                              next.add(vote.roundNumber);
                            }
                            return next;
                          })
                        }
                        aria-expanded={isShowingAllCitations}
                        className="pl-2 text-xs text-primary/70 hover:text-primary cursor-pointer"
                      >
                        {isShowingAllCitations ? "Show less ▼" : `Show all ${moreCount} citations ▶`}
                      </button>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Reset button */}
        <div className="text-center pt-4">
          <button
            type="button"
            onClick={onReset}
            className="px-8 py-3 bg-primary text-sidebar-dark font-semibold rounded-xl hover:bg-primary-dark transition-colors"
          >
            开始新会话
          </button>
        </div>
        <div className="mt-6 pt-4 border-t border-gray-700">
          <a
            href="https://forms.gle/Sm62ucNSKQzGGPJ76"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-primary/30 text-primary hover:bg-primary/10 transition-colors text-sm font-medium"
          >
            <ExternalLink size={16} />
            Share detailed feedback (Google Form)
          </a>
        </div>
      </div>
    </div>
  );
}
