"use client";

import Link from "next/link";
import { ArrowRight, ExternalLink, Network } from "lucide-react";
import { useState } from "react";
import { useI18n } from "@/i18n/context";
import type { Citation, GraphCitation, TextCitation } from "@/lib/types";

type VoteOption = "a" | "b" | "tie";

interface ArenaRoundVote {
  roundNumber: number;
  query: string;
  responseA: string;
  responseB: string;
  positionMapping: Record<string, string>;
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
  return (vote.positionMapping[vote.vote] as "rag" | "plain") ?? "tie";
}

function getRagCitations(vote: ArenaRoundVote): Citation[] {
  const ragPanel = Object.entries(vote.positionMapping).find(([, value]) => value === "rag")?.[0];
  if (ragPanel === "a") return vote.citationsA ?? [];
  if (ragPanel === "b") return vote.citationsB ?? [];
  return [];
}

function TextCitationDetail({ citation }: { citation: TextCitation }) {
  const { t } = useI18n();

  return (
    <div className="space-y-2">
      <p className="text-xs font-semibold uppercase tracking-wider text-primary/80">{t.citation.passageContent}</p>
      <div className="rounded-md border border-primary/10 bg-sidebar-dark/30 p-3">
        <p className="whitespace-pre-wrap border-l-2 border-primary/40 pl-3 text-sm leading-relaxed text-parchment/90">
          {citation.content}
        </p>
      </div>
      {citation.chunk_id ? (
        <Link
          href={`/source/${encodeURIComponent(citation.chunk_id)}`}
          target="_blank"
          rel="noopener noreferrer"
          className="mt-2 inline-flex items-center gap-1 text-xs text-primary/70 hover:text-primary"
        >
          {t.arenaReveal.viewFullParagraph}
        </Link>
      ) : null}
      <span className="inline-flex items-center rounded bg-primary/10 px-1.5 py-0.5 text-[10px] text-primary/70">
        {t.citation.relevance}: {citation.relevance_percent.toFixed(1)}%
      </span>
    </div>
  );
}

function GraphCitationDetail({ citation }: { citation: GraphCitation }) {
  const { t } = useI18n();
  const entityName = citation.fact.match(/^(.+?)\s*--/)?.[1]?.trim();

  return (
    <div className="space-y-2">
      <p className="text-xs font-semibold uppercase tracking-wider text-primary/80">{t.citation.graphFact}</p>
      <div className="rounded-md border border-primary/10 bg-sidebar-dark/30 p-3">
        <p className="text-sm leading-relaxed text-parchment/90">{citation.fact}</p>
      </div>
      <div className="flex items-center gap-3">
        <span className="inline-flex items-center rounded bg-primary/10 px-1.5 py-0.5 text-[10px] text-primary/70">
          {t.citation.depth} {citation.depth}-{t.citation.hop}
        </span>
        {entityName && (
          <Link
            href={`/kg/${encodeURIComponent(entityName)}`}
            className="inline-flex items-center gap-1 text-xs text-primary/70 hover:text-primary"
          >
            <Network size={12} />
            {t.arenaReveal.exploreFullGraph}
          </Link>
        )}
      </div>
    </div>
  );
}

export function ArenaReveal({ votes, onReset }: ArenaRevealProps) {
  const [expandedCitation, setExpandedCitation] = useState<{ round: number; index: number } | null>(null);
  const [showAllCitations, setShowAllCitations] = useState<Set<number>>(() => new Set<number>());
  const { t } = useI18n();

  const ragWins = votes.filter((vote) => resolveWinner(vote) === "rag").length;
  const plainWins = votes.filter((vote) => resolveWinner(vote) === "plain").length;
  const ties = votes.filter((vote) => resolveWinner(vote) === "tie").length;
  const total = votes.length;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-background-dark/95 backdrop-blur-sm">
      <div className="mx-auto max-w-3xl space-y-6 px-4 py-8">
        <div className="space-y-2 text-center">
          <h1 className="text-2xl font-bold text-primary">{t.arenaReveal.title}</h1>
          <p className="text-sm text-gray-400">{t.arenaReveal.subtitle.replace("{total}", String(total))}</p>
        </div>

        <div className="grid grid-cols-1 gap-4 text-center sm:grid-cols-3">
          <div className="rounded-xl border border-primary/30 bg-primary/10 p-4">
            <div className="text-3xl font-bold text-primary">{ragWins}</div>
            <div className="mt-1 text-xs text-gray-400">{t.arenaReveal.ragWins}</div>
          </div>
          <div className="rounded-xl border border-gray-700 bg-gray-800/50 p-4">
            <div className="text-3xl font-bold text-gray-300">{ties}</div>
            <div className="mt-1 text-xs text-gray-400">{t.arenaReveal.ties}</div>
          </div>
          <div className="rounded-xl border border-gray-700 bg-gray-700/20 p-4">
            <div className="text-3xl font-bold text-gray-400">{plainWins}</div>
            <div className="mt-1 text-xs text-gray-400">{t.arenaReveal.llmWins}</div>
          </div>
        </div>

        <div className="space-y-4">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-300">{t.arenaReveal.roundDetail}</h2>
          {votes.map((vote) => {
            const winner = resolveWinner(vote);
            const ragCitations = getRagCitations(vote);
            const isShowingAllCitations = showAllCitations.has(vote.roundNumber);
            const displayedCitations = isShowingAllCitations ? ragCitations : ragCitations.slice(0, 8);
            const moreCount = ragCitations.length - displayedCitations.length;
            const ragPanel = Object.entries(vote.positionMapping).find(([, value]) => value === "rag")?.[0]?.toUpperCase();
            const plainPanel = Object.entries(vote.positionMapping).find(([, value]) => value === "plain")?.[0]?.toUpperCase();

            return (
              <div key={vote.roundNumber} className="space-y-3 rounded-xl border border-gray-700 bg-sidebar-dark p-4">
                <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                  <span className="text-xs font-semibold uppercase text-gray-500">
                    {t.arenaReveal.roundLabel.replace("{round}", String(vote.roundNumber))}
                  </span>
                  <div className="flex flex-wrap gap-2 text-xs">
                    <span className="rounded-full border border-primary/30 bg-primary/20 px-2 py-0.5 text-primary">
                      {ragPanel} = {t.arena.ragEnhanced}
                    </span>
                    <span className="rounded-full border border-gray-700 bg-gray-700/30 px-2 py-0.5 text-gray-400">
                      {plainPanel} = {t.arena.plainLLM}
                    </span>
                  </div>
                </div>

                <p className="text-sm font-medium text-gray-300">&quot;{vote.query}&quot;</p>

                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs text-gray-500">{t.arenaReveal.yourChoice}</span>
                  {winner === "rag" && (
                    <span className="rounded-full border border-primary/40 bg-primary/20 px-2 py-0.5 text-xs font-medium text-primary">
                      {t.arenaReveal.voteRag}
                    </span>
                  )}
                  {winner === "plain" && (
                    <span className="rounded-full border border-gray-600 bg-gray-700/30 px-2 py-0.5 text-xs text-gray-400">
                      {t.arenaReveal.voteLlm}
                    </span>
                  )}
                  {winner === "tie" && (
                    <span className="rounded-full border border-gray-600 bg-gray-700/30 px-2 py-0.5 text-xs text-gray-400">
                      {t.arenaReveal.voteTie}
                    </span>
                  )}
                  {vote.comment && <span className="ml-1 text-xs italic text-gray-500">&quot;{vote.comment}&quot;</span>}
                </div>

                {ragCitations.length > 0 && (
                  <div className="space-y-1 border-t border-gray-800 pt-2">
                    <p className="text-xs font-medium text-primary">{t.arenaReveal.ragCitations}</p>
                    <ul className="space-y-2">
                      {displayedCitations.map((citation, index) => {
                        const isExpanded =
                          expandedCitation?.round === vote.roundNumber && expandedCitation?.index === index;

                        return (
                          <li key={`${vote.roundNumber}-${index}`} className="text-xs">
                            <button
                              type="button"
                              onClick={() =>
                                setExpandedCitation(isExpanded ? null : { round: vote.roundNumber, index })
                              }
                              aria-expanded={isExpanded}
                              className={`flex w-full cursor-pointer items-start gap-2 text-left transition-colors ${
                                isExpanded ? "text-primary" : "text-gray-400 hover:text-primary"
                              }`}
                            >
                              <span className="shrink-0 text-primary/70">[{index + 1}]</span>
                              <span className="flex-1">{citation.type === "text" ? citation.source : citation.type === "graph" ? citation.fact : ""}</span>
                              <span className="ml-1 shrink-0 text-primary/50">{isExpanded ? "▼" : "▶"}</span>
                            </button>

                            {isExpanded && (
                              <div className="mt-1 ml-4 rounded-lg border border-primary/20 bg-background-dark/50 p-3">
                                {citation.type === "text" ? (
                                  <TextCitationDetail citation={citation} />
                                ) : citation.type === "graph" ? (
                                  <GraphCitationDetail citation={citation} />
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
                            if (next.has(vote.roundNumber)) next.delete(vote.roundNumber);
                            else next.add(vote.roundNumber);
                            return next;
                          })
                        }
                        aria-expanded={isShowingAllCitations}
                        className="pl-2 text-xs text-primary/70 transition-colors hover:text-primary"
                      >
                        {isShowingAllCitations ? "▲" : `+${moreCount}`}
                      </button>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <div className="space-y-4 pt-4 text-center">
          <button
            type="button"
            onClick={onReset}
            className="rounded-xl bg-primary px-8 py-3 font-semibold text-sidebar-dark transition-colors hover:bg-primary-dark"
          >
            {t.arenaReveal.newSession}
          </button>

          <div className="rounded-2xl border border-primary/25 bg-gradient-to-r from-primary/20 to-primary/10 p-[1px] shadow-[0_0_24px_rgba(25,230,212,0.14)]">
            <div className="rounded-[15px] bg-sidebar-dark/95 px-5 py-5 text-center">
              <a
                href="https://forms.gle/Sm62ucNSKQzGGPJ76"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex animate-pulse items-center gap-2 rounded-xl bg-gradient-to-r from-primary/20 to-primary/10 px-5 py-3 text-base font-semibold text-primary shadow-[0_0_20px_rgba(25,230,212,0.12)] transition-all hover:scale-[1.02] hover:from-primary/25 hover:to-primary/15"
              >
                <ExternalLink size={18} />
                {t.arenaReveal.feedbackButton}
              </a>
              <p className="mt-3 text-sm text-gray-400">{t.arenaReveal.feedbackDescription}</p>
            </div>
          </div>

          <Link
            href="/"
            className="flex items-center justify-between gap-3 rounded-2xl border border-primary/20 bg-sidebar-dark/80 px-5 py-4 text-left transition-colors hover:bg-primary/5"
          >
            <div>
              <p className="font-medium text-parchment">{t.arenaReveal.tryMainChat}</p>
              <p className="mt-1 text-sm text-gray-400">{t.arenaReveal.tryMainChatDescription}</p>
            </div>
            <ArrowRight className="shrink-0 text-primary" size={20} />
          </Link>
        </div>
      </div>
    </div>
  );
}
