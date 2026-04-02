"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { ArenaModelSelector } from "@/components/ArenaModelSelector";
import { ArenaPanel } from "@/components/ArenaPanel";
import { ArenaReveal } from "@/components/ArenaReveal";
import { ArenaVoteBar } from "@/components/ArenaVoteBar";
import { useArena } from "@/hooks/useArena";
import { ARENA_SAMPLE_PROMPTS } from "@/lib/arenaPrompts";
import { useSettings } from "@/hooks/useSettings";
import type { VoteOption } from "@/hooks/useArena";
import type { Citation } from "@/lib/types";

export default function ArenaPage() {
  const { settings, isLoaded } = useSettings();
  const [sessionId] = useState(() =>
    typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID()
      : Math.random().toString(36).slice(2) + Date.now().toString(36)
  );
  const [inputValue, setInputValue] = useState("");
  const [selectedVote, setSelectedVote] = useState<VoteOption | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [expandedHistory, setExpandedHistory] = useState<Set<number>>(new Set());
  const [showGuide, setShowGuide] = useState(false);

  useEffect(() => {
    const seen = localStorage.getItem("arena-guide-seen");
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (!seen) setShowGuide(true);
  }, []);

  const dismissGuide = () => {
    setShowGuide(false);
    localStorage.setItem("arena-guide-seen", "1");
  };
  const { state, sendArenaQuery, submitVote, setSelectedModel, revealAll, resetSession } =
    useArena(sessionId);
  const didSyncInitialModel = useRef(false);

  useEffect(() => {
    if (!isLoaded || didSyncInitialModel.current) return;
    setSelectedModel(settings.arenaModels.plus);
    didSyncInitialModel.current = true;
  }, [isLoaded, settings.arenaModels.plus, setSelectedModel]);

  const arenaModelPresets = useMemo(
    () => [
      { label: "Flash", value: settings.arenaModels.flash, description: "轻量快速" },
      { label: "Plus", value: settings.arenaModels.plus, description: "均衡性价比" },
      { label: "Max", value: settings.arenaModels.max, description: "旗舰性能" },
    ],
    [settings.arenaModels.flash, settings.arenaModels.plus, settings.arenaModels.max]
  );

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

  if (!isLoaded) {
    return (
      <div className="flex h-screen items-center justify-center bg-background-dark text-parchment">
        <div className="animate-pulse text-sm text-gray-400">Loading Arena settings...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background-dark text-parchment flex flex-col">
      <header className="flex items-center gap-4 px-6 py-4 border-b border-white/5 bg-background-dark/80 backdrop-blur-md z-10 shrink-0">
        <Link href="/" className="flex items-center gap-2 text-gray-400 hover:text-parchment transition-colors group">
          <ArrowLeft size={18} className="group-hover:-translate-x-1 transition-transform" />
          <span className="text-sm font-medium">Back to TCM-Sage</span>
        </Link>
        <h1 className="text-lg font-semibold text-primary shrink-0">TCM Arena</h1>
        <div className="flex-1">
          <ArenaModelSelector
            models={arenaModelPresets}
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
      {state.votes.length > 0 && !state.showReveal && (
        <div className="max-h-48 overflow-y-auto p-4 space-y-2 border-b border-gray-800 shrink-0">
          {state.votes.map((vote) => {
            const isExpanded = expandedHistory.has(vote.roundNumber);
            const labelA = vote.positionMapping?.["a"] === "rag" ? "RAG Enhanced" : "Plain LLM";
            const labelB = vote.positionMapping?.["b"] === "rag" ? "RAG Enhanced" : "Plain LLM";

            return (
              <div
                key={vote.roundNumber}
                className="bg-sidebar-dark border border-gray-700 rounded-lg p-3 cursor-pointer hover:border-gray-600 transition-colors"
                onClick={() => {
                  setExpandedHistory(prev => {
                    const next = new Set(prev);
                    if (next.has(vote.roundNumber)) next.delete(vote.roundNumber);
                    else next.add(vote.roundNumber);
                    return next;
                  });
                }}
              >
                <div className="flex justify-between items-center mb-1">
                  <div className="text-xs font-semibold text-gray-500 uppercase">
                    Round {vote.roundNumber}: <span className="normal-case">{vote.query}</span>
                  </div>
                  <div className="text-xs text-gray-500">
                    {isExpanded ? "Collapse ▲" : "Expand ▼"}
                  </div>
                </div>
                {!isExpanded ? (
                  <div className="flex flex-col gap-1">
                    <div className="text-sm text-gray-400 truncate">
                      <span className="font-semibold text-gray-500">Model A:</span> {vote.responseA.slice(0, 100)}...
                    </div>
                    <div className="text-sm text-gray-400 truncate">
                      <span className="font-semibold text-gray-500">Model B:</span> {vote.responseB.slice(0, 100)}...
                    </div>
                  </div>
                ) : (
                  <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4 border-t border-gray-700 pt-3 cursor-text" onClick={(e) => e.stopPropagation()}>
                    <div>
                      <div className="text-xs font-semibold text-gray-500 mb-1">Model A ({labelA})</div>
                      <div className="text-sm text-parchment whitespace-pre-wrap">{vote.responseA}</div>
                    </div>
                    <div>
                      <div className="text-xs font-semibold text-gray-500 mb-1">Model B ({labelB})</div>
                      <div className="text-sm text-parchment whitespace-pre-wrap">{vote.responseB}</div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

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

      {showGuide && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
              <div className="bg-sidebar-dark border border-gray-700 rounded-2xl p-8 max-w-md mx-4 space-y-4 shadow-2xl">
                  <h2 className="text-xl font-serif font-bold text-parchment">Welcome to Arena</h2>
                  <div className="space-y-3 text-sm text-gray-300 leading-relaxed">
                      <p>Arena is a <strong className="text-parchment">blind evaluation</strong> system for comparing our RAG-enhanced responses against a standard LLM.</p>
                      <ol className="list-decimal list-inside space-y-2">
                          <li>Both panels use the <strong className="text-parchment">same AI model</strong> — one receives retrieved classical text context, the other does not.</li>
                          <li>Read both responses and <strong className="text-parchment">vote for the one you find better</strong> — you won&apos;t know which is which until after voting.</li>
                          <li>After voting, the <strong className="text-parchment">reveal panel</strong> shows which was RAG-enhanced, along with citations used.</li>
                          <li>You can continue asking <strong className="text-parchment">follow-up questions</strong> for multi-round evaluation.</li>
                      </ol>
                      <p className="text-gray-400 text-xs">Your votes help us improve the system. Thank you for participating!</p>
                  </div>
                  <button
                      onClick={dismissGuide}
                      className="w-full py-2.5 rounded-lg bg-primary text-background-dark font-semibold hover:bg-primary/90 transition-colors"
                  >
                      Got it, let&apos;s start
                  </button>
              </div>
          </div>
      )}
    </div>
  );
}
