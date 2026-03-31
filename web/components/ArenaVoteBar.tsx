"use client";

import { useState } from "react";

type VoteOption = "a" | "b" | "tie";

interface ArenaVoteBarProps {
  onVote: (vote: VoteOption, comment?: string) => void;
  onReveal: () => void;
  disabled: boolean;        // true while either stream is active
  hasVoted: boolean;        // true after user voted this round
  selectedVote?: VoteOption | null;
  totalVotes: number;       // votes cast across all rounds
  roundNumber: number;
}

export function ArenaVoteBar({
  onVote,
  onReveal,
  disabled,
  hasVoted,
  selectedVote,
  totalVotes,
  roundNumber,
}: ArenaVoteBarProps) {
  const [showComment, setShowComment] = useState(false);
  const [comment, setComment] = useState("");

  const handleVote = (vote: VoteOption) => {
    if (disabled || hasVoted) return;
    onVote(vote, comment.trim() || undefined);
    setComment("");
    setShowComment(false);
  };

  const voteButtons: { value: VoteOption; label: string; activeColor: string }[] = [
    { value: "a", label: "A 更好", activeColor: "bg-blue-600 border-blue-500 text-white" },
    { value: "b", label: "B 更好", activeColor: "bg-purple-600 border-purple-500 text-white" },
    { value: "tie", label: "平局", activeColor: "bg-gray-600 border-gray-500 text-white" },
  ];

  return (
    <div className="border-t border-gray-700 bg-[#0d0d1a] px-4 py-3 space-y-2 shrink-0">
      {/* Round indicator */}
      <div className="flex items-center justify-between text-xs text-gray-500">
        <span>第 {roundNumber} 轮 · 已投 {totalVotes} 票</span>
        {hasVoted && (
          <span className="text-[#19e6d4] font-medium">✓ 已投票，可继续提问</span>
        )}
      </div>

      {/* Vote buttons row */}
      <div className="flex items-center gap-2 flex-wrap">
        {voteButtons.map(({ value, label, activeColor }) => {
          const isSelected = selectedVote === value;
          const isInactive = hasVoted && !isSelected;
          return (
            <button
              key={value}
              type="button"
              onClick={() => handleVote(value)}
              disabled={disabled || hasVoted}
              className={`px-4 py-2 rounded-lg text-sm font-medium border transition-all ${
                isSelected
                  ? activeColor
                  : isInactive
                  ? "border-gray-700 text-gray-600 bg-transparent cursor-not-allowed opacity-40"
                  : disabled
                  ? "border-gray-700 text-gray-600 bg-transparent cursor-not-allowed opacity-40"
                  : "border-gray-600 text-gray-300 hover:border-gray-400 hover:text-white"
              }`}
            >
              {label}
            </button>
          );
        })}

        {/* Comment toggle */}
        {!hasVoted && !disabled && (
          <button
            type="button"
            onClick={() => setShowComment((v) => !v)}
            className="px-3 py-2 text-xs border border-gray-700 rounded-lg text-gray-500 hover:text-gray-300 hover:border-gray-500 transition-colors"
          >
            {showComment ? "隐藏备注" : "+ 添加备注"}
          </button>
        )}

        {/* Spacer */}
        <div className="flex-1" />

        {/* End & Reveal */}
        <button
          type="button"
          onClick={onReveal}
          disabled={totalVotes === 0}
          className="px-4 py-2 rounded-lg text-sm font-medium border border-[#19e6d4] text-[#19e6d4] hover:bg-[#19e6d4]/10 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          揭晓结果
        </button>
      </div>

      {/* Comment textarea */}
      {showComment && (
        <textarea
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          placeholder="可选：说明你的选择理由…"
          rows={2}
          className="w-full bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm text-[#F3EFE0] placeholder-gray-500 focus:outline-none focus:border-[#19e6d4] resize-none"
        />
      )}
    </div>
  );
}
