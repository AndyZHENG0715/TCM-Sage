"use client";

import { useRef, useState } from "react";
import Link from "next/link";
import { ArenaModelSelector } from "@/components/ArenaModelSelector";
import { ArenaPanel } from "@/components/ArenaPanel";
import { ArenaReveal } from "@/components/ArenaReveal";
import { ArenaVoteBar } from "@/components/ArenaVoteBar";
import { useArena } from "@/hooks/useArena";
import { ARENA_MODEL_PRESETS, ARENA_SAMPLE_PROMPTS } from "@/lib/arenaPrompts";
import type { VoteOption } from "@/hooks/useArena";
import type { Citation } from "@/lib/types";

export default function ArenaPage() {
  const [sessionId] = useState(() => crypto.randomUUID());
  const [inputValue, setInputValue] = useState("");
  const [selectedVote, setSelectedVote] = useState<VoteOption | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const { state, sendArenaQuery, submitVote, setSelectedModel, revealAll, resetSession } =
    useArena(sessionId);

  const isStreaming = state.isStreamingA || state.isStreamingB;
  const bothDone = state.canVote && !isStreaming;

  const posMap = state.arenaConfig?.position_mapping ?? {};
  const revealLabelA: "RAG Enhanced" | "Plain LLM" | null = state.showReveal
    ? posMap["a"] === "rag"
      ? "RAG Enhanced"
      : "Plain LLM"
    : null;
  const revealLabelB: "RAG Enhanced" | "Plain LLM" | null = state.showReveal
    ? posMap["b"] === "rag"
      ? "RAG Enhanced"
      : "Plain LLM"
    : null;

  const citationsA = (state.metadataA?.citations ?? []) as Citation[];
  const citationsB = (state.metadataB?.citations ?? []) as Citation[];

  const handleSubmit = () => {
    const question = inputValue.trim();
    if (!question || isStreaming) return;

    setInputValue("");
    setSelectedVote(null);
    void sendArenaQuery(question);
  };

  const handleVote = async (vote: VoteOption, comment?: string) => {
    setSelectedVote(vote);
    await submitVote(vote, comment);
    inputRef.current?.focus();
  };

  const handleReset = () => {
    resetSession();
    setInputValue("");
    setSelectedVote(null);
  };

  return (
    <div className="min-h-screen bg-background-dark text-parchment flex flex-col">
      <header className="flex items-center gap-4 px-6 py-4 border-b border-white/5 bg-background-dark/80 backdrop-blur-md z-10 shrink-0">
        <Link href="/" className="shrink-0 p-1.5 text-gray-400 hover:text-primary transition-colors rounded-lg hover:bg-primary/10" title="返回主页">
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m15 18-6-6 6-6"/></svg>
        </Link>
        <h1 className="text-lg font-semibold text-primary shrink-0">TCM Arena</h1>
        <div className="flex-1">
          <ArenaModelSelector
            models={ARENA_MODEL_PRESETS}
            selected={state.selectedModel}
            onSelect={setSelectedModel}
            disabled={isStreaming}
          />
        </div>
        <button
          type="button"
          onClick={handleReset}
          className="shrink-0 px-3 py-1 text-sm border border-gray-600 rounded-lg text-gray-300 hover:border-gray-400 hover:text-white transition-colors"
        >
          New Session
        </button>
      </header>

      <main className="flex flex-1 flex-col md:flex-row gap-4 p-4 min-h-0 overflow-hidden">
        <ArenaPanel
          label="Model A"
          content={state.responseA}
          isStreaming={state.isStreamingA}
          error={state.errorA}
          revealed={state.showReveal}
          revealLabel={revealLabelA}
          citations={citationsA}
        />
        <ArenaPanel
          label="Model B"
          content={state.responseB}
          isStreaming={state.isStreamingB}
          error={state.errorB}
          revealed={state.showReveal}
          revealLabel={revealLabelB}
          citations={citationsB}
        />
      </main>

      {(bothDone || state.votes.length > 0) && !state.showReveal && (
        <ArenaVoteBar
          onVote={(vote, comment) => {
            void handleVote(vote, comment);
          }}
          onReveal={revealAll}
          disabled={isStreaming || !bothDone}
          hasVoted={state.hasVotedThisRound}
          selectedVote={selectedVote}
          totalVotes={state.votes.length}
          roundNumber={state.roundNumber}
        />
      )}

      {!state.showReveal && (
        <footer className="px-4 py-3 border-t border-gray-700 bg-sidebar-dark space-y-2">
          <div className="flex gap-2 flex-wrap">
            {ARENA_SAMPLE_PROMPTS.slice(0, 5).map((prompt) => (
              <button
                key={prompt}
                type="button"
                onClick={() => setInputValue(prompt)}
                disabled={isStreaming}
                className="text-xs px-2 py-1 rounded-full bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white transition-colors disabled:opacity-40"
              >
                {prompt.length > 20 ? prompt.slice(0, 20) + "…" : prompt}
              </button>
            ))}
          </div>

          <div className="flex gap-2">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(event) => setInputValue(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  handleSubmit();
                }
              }}
              placeholder="向两个模型提问…"
              disabled={isStreaming}
              className="flex-1 bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm text-parchment placeholder-gray-500 focus:outline-none focus:border-primary disabled:opacity-50"
            />
            <button
              type="button"
              onClick={handleSubmit}
              disabled={!inputValue.trim() || isStreaming}
              className="px-4 py-2 bg-primary text-sidebar-dark font-semibold text-sm rounded-lg disabled:opacity-40 disabled:cursor-not-allowed hover:bg-primary-dark transition-colors"
            >
              提交
            </button>
          </div>
        </footer>
      )}

      {state.showReveal && <ArenaReveal votes={state.votes} onReset={handleReset} />}
    </div>
  );
}
