"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { X, BookOpen, ExternalLink, Loader2 } from "lucide-react";
import { Citation, GraphCitation, TextCitation } from "@/lib/types";
import { ChunkContext, fetchChunkContext } from "@/lib/api";
import { getDisplaySourceLabel, getOcrArtifacts } from "@/lib/citations";
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

function TextCitationContent({ citation }: { citation: TextCitation }) {
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
                            : "Failed to load context"
                    );
                }
            });

        return () => {
            isCancelled = true;
        };
    }, [chunkId, showFullText, context]);

    const loading = showFullText && !context && !error;
    const sourceLabel = getDisplaySourceLabel(
        citation.source,
        context?.chapter_display || context?.chapter
    );
    const sourceDisplay = [context?.book, sourceLabel].filter(Boolean).join(" — ");
    
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
            <div className="space-y-2">
                <h3 className="font-sans text-sm font-semibold text-[#8c8578] uppercase">
                    Source Chapter
                </h3>
                <p className="font-serif text-2xl text-parchment-text border-b-2 border-primary/20 pb-2 inline-block">
                    {sourceDisplay || sourceLabel || "Source"}
                </p>
            </div>

            <div className="space-y-3">
                <div className="flex items-center justify-between">
                    <h3 className="font-sans text-sm font-semibold text-[#8c8578] uppercase">
                        Passage Content
                    </h3>
                    {chunkId && (
                        <button
                            onClick={() => setShowFullText(!showFullText)}
                            className="text-xs font-bold text-primary hover:underline flex items-center gap-1"
                        >
                            {showFullText ? "View Snippet" : "View Full Paragraph"}
                        </button>
                    )}
                </div>
                <div className="relative pl-6">
                    <div className="absolute left-0 top-0 bottom-0 w-1 bg-primary/50 rounded-full" />
                    <p className="font-serif text-lg text-parchment-text leading-loose whitespace-pre-wrap">
                        {loading ? (
                            <span className="inline-flex items-center gap-2 text-[#8c8578]">
                                <Loader2 size={16} className="animate-spin" />
                                Loading full paragraph context...
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
                    Possible OCR/source artifact detected in this passage: {ocrArtifacts.join(", ")}
                </div>
            )}

            {error && (
                <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                    {error}
                </div>
            )}

            <div className="flex gap-2">
                <span className="inline-flex items-center px-2 py-1 rounded bg-[#dcd3b8]/50 text-[#5c5548] text-xs font-medium">
                    Rel: {citation.relevance_percent.toFixed(1)}%
                </span>
            </div>
        </div>
    );
}

function GraphCitationContent({ citation }: { citation: GraphCitation }) {
    return (
        <div className="space-y-6">
            <div className="space-y-2">
                <h3 className="font-sans text-sm font-semibold text-[#8c8578] uppercase">
                    Fact Relationship
                </h3>
                <KGViewer citation={citation} />
            </div>

            <div className="space-y-3">
                <h3 className="font-sans text-sm font-semibold text-[#8c8578] uppercase">
                    Traversal Metadata
                </h3>
                <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 bg-[#f5f5f5] rounded-lg border border-gray-200">
                        <p className="text-xs text-gray-500 uppercase">Depth</p>
                        <p className="text-lg font-bold text-gray-800">{citation.depth}-hop</p>
                    </div>
                </div>
            </div>

            {citation.source_ref && (
                <div className="space-y-3">
                    <h3 className="font-sans text-sm font-semibold text-[#8c8578] uppercase">
                        Provenance
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
    if (!citation) {
        return null;
    }

    const isText = citation.type === "text";
    const chunkId = isText ? (citation as TextCitation).chunk_id : undefined;

    return (
        <div className="fixed inset-y-0 right-0 w-full sm:w-[400px] lg:w-[450px] bg-parchment shadow-2xl transform transition-transform duration-300 ease-in-out z-50 flex flex-col border-l border-[#dcd3b8]">
            <div className="flex items-center justify-between p-6 border-b border-[#dcd3b8] bg-[#ebe5d5]">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-primary/10 rounded-lg text-primary-dark">
                        <BookOpen size={20} />
                    </div>
                    <div>
                        <h2 className="font-sans text-xs font-bold text-[#8c8578] uppercase tracking-wider">
                            {isText ? "Classic Text Source" : "Knowledge Graph Fact"}
                        </h2>
                        <p className="font-serif font-bold text-parchment-text text-lg leading-tight">
                            Ref [{citation.number}]
                        </p>
                    </div>
                </div>
                <button
                    onClick={onClose}
                    className="p-2 text-[#8c8578] hover:text-primary-dark hover:bg-[#dcd3b8]/50 rounded-full transition-colors"
                >
                    <X size={20} />
                </button>
            </div>

            <div className="flex-1 overflow-y-auto p-6">
                {isText ? (
                    <TextCitationContent
                        key={chunkId || `text-${citation.number}`}
                        citation={citation as TextCitation}
                    />
                ) : (
                    <GraphCitationContent citation={citation as GraphCitation} />
                )}
            </div>

            <div className="p-6 border-t border-[#dcd3b8] bg-[#ebe5d5]">
                {isText && chunkId ? (
                    <Link
                        href={`/source/${chunkId}`}
                        target="_blank"
                        className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-primary text-white font-sans font-bold rounded-lg hover:bg-primary-dark transition-colors shadow-sm"
                    >
                        <ExternalLink size={18} />
                        View Full Context
                    </Link>
                ) : (
                    <button
                        disabled
                        title={isText ? "No chunk ID available" : "Graph viewer shown above"}
                        className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-primary/40 text-background-dark/60 font-sans font-bold rounded-lg cursor-not-allowed shadow-sm"
                    >
                        <ExternalLink size={18} />
                        {isText ? "View Full Context" : "View Graph"}
                    </button>
                )}
            </div>
        </div>
    );
}
