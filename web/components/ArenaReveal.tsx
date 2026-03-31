"use client";

import type { Citation } from "@/lib/types";

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

export function ArenaReveal({ votes, onReset }: ArenaRevealProps) {
  const ragWins = votes.filter((v) => resolveWinner(v) === "rag").length;
  const plainWins = votes.filter((v) => resolveWinner(v) === "plain").length;
  const ties = votes.filter((v) => resolveWinner(v) === "tie").length;
  const total = votes.length;

  return (
    <div className="fixed inset-0 z-50 bg-[#0a0a17]/95 backdrop-blur-sm overflow-y-auto">
      <div className="max-w-3xl mx-auto px-4 py-8 space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-2xl font-bold text-[#19e6d4]">评测结果揭晓</h1>
          <p className="text-gray-400 text-sm">共 {total} 轮对话 · 盲评已结束</p>
        </div>

        {/* Summary stats */}
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="bg-[#19e6d4]/10 border border-[#19e6d4]/30 rounded-xl p-4">
            <div className="text-3xl font-bold text-[#19e6d4]">{ragWins}</div>
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
            const ragPanel = Object.entries(vote.positionMapping).find(([, v]) => v === "rag")?.[0]?.toUpperCase();
            const plainPanel = Object.entries(vote.positionMapping).find(([, v]) => v === "plain")?.[0]?.toUpperCase();

            return (
              <div
                key={vote.roundNumber}
                className="bg-[#0d0d1a] border border-gray-700 rounded-xl p-4 space-y-3"
              >
                {/* Round header */}
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-gray-500 uppercase">
                    第 {vote.roundNumber} 轮
                  </span>
                  <div className="flex gap-2 text-xs">
                    <span className="bg-[#19e6d4]/15 text-[#19e6d4] border border-[#19e6d4]/30 px-2 py-0.5 rounded-full">
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
                    <span className="text-xs bg-[#19e6d4]/20 text-[#19e6d4] border border-[#19e6d4]/40 px-2 py-0.5 rounded-full font-medium">
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
                    <p className="text-xs text-[#19e6d4] font-medium">RAG 引用来源</p>
                    <ul className="space-y-0.5">
                      {ragCitations.slice(0, 4).map((c, i) => (
                        <li key={i} className="text-xs text-gray-500">
                          [{i + 1}]{" "}
                          {c.type === "text"
                            ? `${(c as { book?: string }).book ?? ""} · ${(c as { chapter_display?: string; chapter?: string }).chapter_display ?? (c as { chapter?: string }).chapter ?? ""}`
                            : c.type === "graph"
                            ? (c as { fact?: string }).fact ?? ""
                            : ""}
                        </li>
                      ))}
                    </ul>
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
            className="px-8 py-3 bg-[#19e6d4] text-[#0d0d1a] font-semibold rounded-xl hover:bg-[#14c9b8] transition-colors"
          >
            开始新会话
          </button>
        </div>
      </div>
    </div>
  );
}
