"use client";

import { useState } from "react";
import { useI18n } from "@/i18n/context";

type VoteOption = "a" | "b" | "tie";

interface ArenaVoteBarProps {
  onVote: (vote: VoteOption, comment?: string) => void;
  onReveal: () => void;
  disabled: boolean;
  hasVoted: boolean;
  selectedVote?: VoteOption | null;
  totalVotes: number;
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
  const { t } = useI18n();

  const handleVote = (vote: VoteOption) => {
    if (disabled || hasVoted) return;
    onVote(vote, comment.trim() || undefined);
    setComment("");
    setShowComment(false);
  };

  const voteButtons: { value: VoteOption; label: string; activeColor: string }[] = [
    { value: "a", label: t.arena.voteA, activeColor: "bg-blue-600 border-blue-500 text-white" },
    { value: "b", label: t.arena.voteB, activeColor: "bg-purple-600 border-purple-500 text-white" },
    { value: "tie", label: t.arena.tie, activeColor: "bg-gray-600 border-gray-500 text-white" },
  ];

  return (
    <div className="shrink-0 space-y-3 border-t border-gray-700 bg-sidebar-dark px-4 py-3">
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-gray-500">
        <span>{t.arena.roundIndicator.replace("{round}", String(roundNumber)).replace("{votes}", String(totalVotes))}</span>
        {hasVoted && <span className="font-medium text-primary">{t.arena.votedContinue}</span>}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {voteButtons.map(({ value, label, activeColor }) => {
          const isSelected = selectedVote === value;
          const isInactive = hasVoted && !isSelected;

          return (
            <button
              key={value}
              type="button"
              onClick={() => handleVote(value)}
              disabled={disabled || hasVoted}
              className={`min-h-11 rounded-lg border px-4 py-3 text-sm font-medium transition-all ${
                isSelected
                  ? activeColor
                  : isInactive || disabled
                    ? "cursor-not-allowed border-gray-700 bg-transparent text-gray-600 opacity-40"
                    : "border-gray-600 text-gray-300 hover:border-gray-400 hover:text-white"
              }`}
            >
              {label}
            </button>
          );
        })}

        {!hasVoted && !disabled && (
          <button
            type="button"
            onClick={() => setShowComment((v) => !v)}
            className="min-h-11 rounded-lg border border-gray-700 px-3 py-2 text-xs text-gray-500 transition-colors hover:border-gray-500 hover:text-gray-300"
          >
            {showComment ? t.arena.hideComment : t.arena.addComment}
          </button>
        )}

        <div className="flex-1" />

        <button
          type="button"
          onClick={onReveal}
          disabled={totalVotes === 0}
          className="min-h-11 rounded-lg border border-primary px-4 py-3 text-sm font-medium text-primary transition-colors hover:bg-primary/10 disabled:cursor-not-allowed disabled:opacity-30"
        >
          {t.arena.reveal}
        </button>
      </div>

      {showComment && (
        <textarea
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          placeholder={t.arenaReveal.commentPlaceholder}
          rows={2}
          className="w-full resize-none rounded-lg border border-gray-600 bg-gray-800 px-3 py-2 text-sm text-parchment placeholder-gray-500 focus:border-primary focus:outline-none"
        />
      )}
    </div>
  );
}
