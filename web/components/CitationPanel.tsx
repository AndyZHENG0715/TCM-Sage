"use client";

import { ChunkContext, fetchChunkContext } from "@/lib/api";
import { getDisplaySourceLabel, getOcrArtifacts } from "@/lib/citations";
import { useI18n } from "@/i18n/context";
import { Citation, GraphCitation, TextCitation } from "@/lib/types";
import { BookOpen, ExternalLink, Loader2, Network, X } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { KGViewer } from "./KGViewer";

interface CitationPanelProps {
    citation: Citation | null;
    onClose: () => void;
}

function HighlightedText({
    text,
    start,
    end,
}: {
    text: string;
    start: number;
    end: number;
}) {
    const safeStart = Math.max(0, start);
    const safeEnd = Math.max(safeStart, end);
    const before = text.slice(0, safeStart);
    const highlighted = text.slice(safeStart, safeEnd);
    const after = text.slice(safeEnd);

    if (!highlighted) {
        return <>{text}</>;
    }

    return (
        <>
            {before}
            <mark className="bg-primary/20 text-parchment-text px-0.5 rounded">
                {highlighted}
            </mark>
            {after}
        </>
    );
}

function TextCitationContent({
    citation,
    onSourceLabelResolved
}: {
    citation: TextCitation;
    onSourceLabelResolved?: (label: string) => void;
}) {
    const { t } = useI18n();
    const chunkId = citation.chunk_id;
    const [context, setContext] = useState<ChunkContext | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [showFullText, setShowFullText] = useState(false);

    useEffect(() => {
        if (!chunkId || !showFullText || context) {
            return;
        }

        let isCancelled = false;

        fetchChunkContext(chunkId)
            .then((data) => {
                if (!isCancelled) {
                    setContext(data);
                }
            })
            .catch((fetchError) => {
                if (!isCancelled) {
                    setError(
                        fetchError instanceof Error
                            ? fetchError.message
                            : t.citation.loadingContext
                    );
                }
            });

        return () => {
            isCancelled = true;
        };
    }, [chunkId, context, showFullText, t.citation.loadingContext]);

    useEffect(() => {
        const label = getDisplaySourceLabel(
            citation.source,
            context?.chapter_display || context?.chapter
        );
        const display = [context?.book, label].filter(Boolean).join(" — ");
        if (display) {
            onSourceLabelResolved?.(display);
        } else {
            onSourceLabelResolved?.(label || t.citation.textSource);
        }
    }, [citation.source, context, onSourceLabelResolved, t.citation.textSource]);

    const loading = showFullText && !context && !error;

    // Determine what text to show based on toggle state
    const paragraphText = showFullText && context?.paragraph_text
        ? context.paragraph_text
        : citation.content;

    // Highlights only apply when we have the full context mapping
    const paragraphStart = showFullText ? (context?.paragraph_highlight_start ?? 0) : 0;
    const paragraphEnd = showFullText
        ? (context?.paragraph_highlight_end ?? paragraphText.length)
        : paragraphText.length;

    const ocrArtifacts = getOcrArtifacts(paragraphText);

    return (
        <div className="space-y-6">
            <div className="space-y-3">
                <div className="flex items-center justify-between">
                    <h3 className="font-sans text-sm font-semibold text-[#8c8578] uppercase">
                        {t.citation.passageContent}
                    </h3>
                    {chunkId && (
                        <button
                            onClick={() => setShowFullText(!showFullText)}
                            className="text-xs font-bold text-primary hover:underline flex items-center gap-1"
                        >
                            {showFullText ? t.citation.viewSnippet : t.citation.viewFullParagraph}
                        </button>
                    )}
                </div>
                <div className="relative pl-6">
                    <div className="absolute left-0 top-0 bottom-0 w-1 bg-primary/50 rounded-full" />
                    <p className="font-serif text-lg text-parchment-text leading-loose whitespace-pre-wrap">
                        {loading ? (
                            <span className="inline-flex items-center gap-2 text-[#8c8578]">
                                <Loader2 size={16} className="animate-spin" />
                                {t.citation.loadingContext}
                            </span>
                        ) : (
                            <HighlightedText
                                text={paragraphText}
                                start={paragraphStart}
                                end={paragraphEnd}
                            />
                        )}
                    </p>
                </div>
            </div>

            {ocrArtifacts.length > 0 && (
                <div className="rounded-lg border border-amber-300/60 bg-amber-50/70 px-4 py-3 text-sm text-amber-900">
                    {t.citation.ocrWarning} {ocrArtifacts.join(", ")}
                </div>
            )}

            {error && (
                <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                    {error}
                </div>
            )}

            <div className="flex gap-2">
                <span className="inline-flex items-center px-2 py-1 rounded bg-[#dcd3b8]/50 text-[#5c5548] text-xs font-medium">
                    {t.citation.relevance}: {citation.relevance_percent.toFixed(1)}%
                </span>
            </div>
        </div>
    );
}

function GraphCitationContent({ citation }: { citation: GraphCitation }) {
    const { t } = useI18n();
    return (
        <div className="space-y-6">
            <div className="space-y-2">
                <h3 className="font-sans text-sm font-semibold text-[#8c8578] uppercase">
                    {t.citation.factRelationship}
                </h3>
                <KGViewer citation={citation} />
            </div>

            <div className="space-y-3">
                <h3 className="font-sans text-sm font-semibold text-[#8c8578] uppercase">
                    {t.citation.traversalMetadata}
                </h3>
                <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 bg-[#f5f5f5] rounded-lg border border-gray-200">
                        <p className="text-xs text-gray-500 uppercase">{t.citation.depth}</p>
                        <p className="text-lg font-bold text-gray-800">{citation.depth}-{t.citation.hop}</p>
                    </div>
                </div>
            </div>

            {citation.source_ref && (
                <div className="space-y-3">
                    <h3 className="font-sans text-sm font-semibold text-[#8c8578] uppercase">
                        {t.citation.provenance}
                    </h3>
                    <pre className="p-3 bg-gray-100 rounded text-xs overflow-x-auto text-gray-700">
                        {JSON.stringify(citation.source_ref, null, 2)}
                    </pre>
                </div>
            )}
        </div>
    );
}

export function CitationPanel({ citation, onClose }: CitationPanelProps) {
    const { t } = useI18n();
    const [sourceLabel, setSourceLabel] = useState<string>("");

    if (!citation) {
        return null;
    }

    const isText = citation.type === "text";
    const chunkId = isText ? (citation as TextCitation).chunk_id : undefined;

    return (
        <div className="relative h-screen w-full sm:w-[400px] lg:w-[450px] bg-parchment shadow-2xl transform transition-transform duration-300 ease-in-out flex flex-col border-l border-[#dcd3b8] shrink-0">
            <div className="flex items-center justify-between p-6 border-b border-[#dcd3b8] bg-[#ebe5d5]">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-primary/10 rounded-lg text-primary-dark">
                        <BookOpen size={20} />
                    </div>
                    <div className="min-w-0 flex-1">
                        <h2 className="font-sans text-xs font-bold text-[#8c8578] uppercase tracking-wider">
                            {isText ? t.citation.textSource : t.citation.graphFact}
                        </h2>
                        <p className="font-serif font-bold text-parchment-text text-lg leading-tight truncate max-w-[250px]">
                            {sourceLabel || `Ref [${citation.number}]`}
                        </p>
                        {sourceLabel && (
                            <p className="font-sans text-[10px] text-[#8c8578] uppercase mt-0.5">
                                Ref [{citation.number}]
                            </p>
                        )}
                    </div>
                </div>
                <button
                    onClick={onClose}
                    className="p-2 text-[#8c8578] hover:text-primary-dark hover:bg-[#dcd3b8]/50 rounded-full transition-colors"
                    aria-label={t.common.close}
                >
                    <X size={20} />
                </button>
            </div>

            <div className="flex-1 overflow-y-auto p-6">
                {isText ? (
                    <TextCitationContent
                        key={chunkId || `text-${citation.number}`}
                        citation={citation as TextCitation}
                        onSourceLabelResolved={setSourceLabel}
                    />
                ) : (
                    <GraphCitationContent citation={citation as GraphCitation} />
                )}
            </div>

            <div className="p-6 border-t border-[#dcd3b8] bg-[#ebe5d5]">
                {isText && chunkId ? (
                    <Link
                        href={`/source/${chunkId}?from=chat`}
                        target="_blank"
                        className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-primary text-white font-sans font-bold rounded-lg hover:bg-primary-dark transition-colors shadow-sm"
                    >
                        <ExternalLink size={18} />
                        {t.citation.viewFullContext}
                    </Link>
                ) : (
                    <Link
                        href={`/kg/${encodeURIComponent(citation.type === "graph" ? ((citation as GraphCitation).fact.match(/^(.+?)\s*--/)?.[1]?.trim() || "graph") : "graph")}`}
                        className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-primary text-white font-sans font-bold rounded-lg hover:bg-primary-dark transition-colors shadow-sm"
                    >
                        <Network size={18} />
                        {t.citation.exploreFullGraph}
                    </Link>
                )}
            </div>
        </div>
    );
}
