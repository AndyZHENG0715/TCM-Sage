"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft } from "lucide-react";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { ArenaModelSelector } from "@/components/ArenaModelSelector";
import { ArenaPanel } from "@/components/ArenaPanel";
import { ArenaReveal } from "@/components/ArenaReveal";
import { ArenaVoteBar } from "@/components/ArenaVoteBar";
import { useI18n } from "@/i18n/context";
import { useArena } from "@/hooks/useArena";
import { useSettings } from "@/hooks/useSettings";
import type { VoteOption } from "@/hooks/useArena";
import type { Citation } from "@/lib/types";
import { ARENA_SAMPLE_PROMPTS } from "@/lib/arenaPrompts";

export default function ArenaPage() {
  const { t } = useI18n();
  const { settings, isLoaded } = useSettings();
  const [sessionId] = useState(() =>
    typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID()
      : Math.random().toString(36).slice(2) + Date.now().toString(36)
  );
  const [inputValue, setInputValue] = useState("");
  const [currentQuery, setCurrentQuery] = useState("");
  const [selectedVote, setSelectedVote] = useState<VoteOption | null>(null);
  const [showGuide, setShowGuide] = useState(() => {
    if (typeof window === "undefined") return false;
    return !localStorage.getItem("arena-guide-seen");
  });
  const inputRef = useRef<HTMLInputElement>(null);
  const didSyncInitialModel = useRef(false);

  const dismissGuide = () => {
    setShowGuide(false);
    localStorage.setItem("arena-guide-seen", "1");
  };

  const { state, sendArenaQuery, submitVote, setSelectedModel, revealAll, resetSession } = useArena(sessionId);

  useEffect(() => {
    if (!isLoaded || didSyncInitialModel.current) return;
    setSelectedModel(settings.arenaModels.flash);
    didSyncInitialModel.current = true;
  }, [isLoaded, setSelectedModel, settings.arenaModels.flash]);

  const arenaModelPresets = useMemo(
    () => [
      { label: t.modelPresets.flash, value: settings.arenaModels.flash, description: t.modelPresets.flashDesc },
      { label: t.modelPresets.plus, value: settings.arenaModels.plus, description: t.modelPresets.plusDesc },
      { label: t.modelPresets.max, value: settings.arenaModels.max, description: t.modelPresets.maxDesc },
    ],
    [settings.arenaModels.flash, settings.arenaModels.max, settings.arenaModels.plus, t.modelPresets]
  );

  const isStreaming = state.isStreamingA || state.isStreamingB;
  const bothDone = state.canVote && !isStreaming;
  const posMap = state.arenaConfig?.position_mapping ?? {};

  const revealLabelA = state.showReveal ? (posMap.a === "rag" ? t.arena.ragEnhanced : t.arena.plainLLM) : null;
  const revealLabelB = state.showReveal ? (posMap.b === "rag" ? t.arena.ragEnhanced : t.arena.plainLLM) : null;

  const citationsA = (state.metadataA?.citations ?? []) as Citation[];
  const citationsB = (state.metadataB?.citations ?? []) as Citation[];

  const handleSubmit = () => {
    const question = inputValue.trim();
    if (!question || isStreaming) return;
    setInputValue("");
    setSelectedVote(null);
    setCurrentQuery(question);
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
    setCurrentQuery("");
    setSelectedVote(null);
  };

  const previousRoundsA = state.votes.map((v) => ({
    query: v.query,
    response: v.responseA,
  }));

  const previousRoundsB = state.votes.map((v) => ({
    query: v.query,
    response: v.responseB,
  }));

  if (!isLoaded) {
    return (
      <div className="flex h-screen items-center justify-center bg-background-dark text-parchment">
        <div className="animate-pulse text-sm text-gray-400">{t.common.loading}</div>
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <div className="flex min-h-screen flex-col bg-background-dark text-parchment">
        <header className="z-10 flex shrink-0 flex-col gap-4 border-b border-white/5 bg-background-dark/80 px-4 py-4 backdrop-blur-md md:flex-row md:items-center md:px-6">
          <div className="flex items-center gap-4">
            <Link href="/" className="group flex min-h-11 items-center gap-2 text-gray-400 transition-colors hover:text-parchment">
              <ArrowLeft size={18} className="transition-transform group-hover:-translate-x-1" />
              <span className="text-sm font-medium">{t.common.backToApp}</span>
            </Link>
            <h1 className="shrink-0 text-lg font-semibold text-primary">{t.arena.title}</h1>
          </div>
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
            className="min-h-11 shrink-0 rounded-lg border border-gray-600 px-4 py-2 text-sm text-gray-300 transition-colors hover:border-gray-400 hover:text-white"
          >
            {t.arena.newSession}
          </button>
        </header>

        <main className="flex min-h-0 flex-1 flex-col gap-4 overflow-hidden p-4 md:flex-row">
          <ArenaPanel
            label={t.arena.modelA}
            rounds={previousRoundsA}
            currentQuery={!state.hasVotedThisRound && (isStreaming || state.responseA) ? currentQuery : undefined}
            currentResponse={!state.hasVotedThisRound ? state.responseA : ""}
            isStreaming={state.isStreamingA}
            error={state.errorA}
            revealed={state.showReveal}
            revealLabel={revealLabelA}
            citations={citationsA}
          />
          <ArenaPanel
            label={t.arena.modelB}
            rounds={previousRoundsB}
            currentQuery={!state.hasVotedThisRound && (isStreaming || state.responseB) ? currentQuery : undefined}
            currentResponse={!state.hasVotedThisRound ? state.responseB : ""}
            isStreaming={state.isStreamingB}
            error={state.errorB}
            revealed={state.showReveal}
            revealLabel={revealLabelB}
            citations={citationsB}
          />
        </main>

        {state.voteError ? (
          <div className="border-t border-primary/20 bg-background-dark px-4 py-2 text-sm text-primary">
            {state.voteError}
          </div>
        ) : null}

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
          <footer className="space-y-2 border-t border-gray-700 bg-sidebar-dark px-4 py-3 pb-[calc(env(safe-area-inset-bottom)+0.75rem)]">
            <div className="flex flex-wrap gap-2">
              {ARENA_SAMPLE_PROMPTS.slice(0, 5).map((item) => (
                <button
                  key={item.label}
                  type="button"
                  onClick={() => setInputValue(item.query)}
                  disabled={isStreaming}
                  className="min-h-11 rounded-full bg-gray-700 px-3 py-2 text-xs text-gray-300 transition-colors hover:bg-gray-600 hover:text-white disabled:opacity-40"
                >
                  {item.label}
                </button>
              ))}
            </div>

            <div className="flex flex-col gap-2 sm:flex-row">
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(event) => setInputValue(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") handleSubmit();
                }}
                placeholder={t.arena.inputPlaceholder}
                disabled={isStreaming}
                className="min-h-11 flex-1 rounded-lg border border-gray-600 bg-gray-800 px-3 py-2 text-sm text-parchment placeholder-gray-500 focus:border-primary focus:outline-none disabled:opacity-50"
              />
              <button
                type="button"
                onClick={handleSubmit}
                disabled={!inputValue.trim() || isStreaming}
                className="min-h-11 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-sidebar-dark transition-colors hover:bg-primary-dark disabled:cursor-not-allowed disabled:opacity-40"
              >
                {t.arena.submit}
              </button>
            </div>
          </footer>
        )}

        {state.showReveal && <ArenaReveal votes={state.votes} onReset={handleReset} />}

        {showGuide && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
            <div className="mx-4 space-y-4 rounded-2xl border border-gray-700 bg-sidebar-dark p-8 shadow-2xl max-w-md">
              <h2 className="text-xl font-serif font-bold text-parchment">{t.arena.guide.title}</h2>
              <div className="space-y-3 text-sm leading-relaxed text-gray-300">
                <p>{t.arena.guide.description}</p>
                <ol className="list-inside list-decimal space-y-2">
                  <li>{t.arena.guide.step1}</li>
                  <li>{t.arena.guide.step2}</li>
                  <li>{t.arena.guide.step3}</li>
                  <li>{t.arena.guide.step4}</li>
                </ol>
                <p className="text-xs text-gray-400">{t.arena.guide.footer}</p>
              </div>
              <button
                type="button"
                onClick={dismissGuide}
                className="w-full rounded-lg bg-primary py-2.5 font-semibold text-background-dark transition-colors hover:bg-primary/90"
              >
                {t.arena.guide.start}
              </button>
            </div>
          </div>
        )}
      </div>
    </ErrorBoundary>
  );
}
