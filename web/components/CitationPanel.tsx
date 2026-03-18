"use client";

import { useState, useRef, useEffect } from "react";
import { Citation, TextCitation, GraphCitation } from "@/lib/types";
import { cleanSourceLabel } from "@/lib/citations";
import { X, BookOpen, ExternalLink, Loader2 } from "lucide-react";
import { fetchChunkContext, ChunkContext } from "@/lib/api";

interface CitationPanelProps {
    citation: Citation | null;
    onClose: () => void;
}

export function CitationPanel({ citation, onClose }: CitationPanelProps) {
    const [context, setContext] = useState<ChunkContext | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const prevCitationRef = useRef(citation);
    useEffect(() => {
        if (citation !== prevCitationRef.current) {
            setContext(null);
            setError(null);
            prevCitationRef.current = citation;
        }
    }, [citation]);

    if (!citation) return null;

    const isText = citation.type === "text";

    async function handleViewContext(chunkId: string) {
        setLoading(true);
        setError(null);
        try {
            const data = await fetchChunkContext(chunkId);
            setContext(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to load context");
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="fixed inset-y-0 right-0 w-full sm:w-[400px] lg:w-[450px] bg-parchment shadow-2xl transform transition-transform duration-300 ease-in-out z-50 flex flex-col border-l border-[#dcd3b8]">
            {/* Header */}
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

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-6">
                {isText ? (
                    <TextCitationContent citation={citation as TextCitation} />
                ) : (
                    <GraphCitationContent citation={citation as GraphCitation} />
                )}
            </div>

            {/* Full context viewer (conditionally shown) */}
            {context && <FullContextViewer context={context} />}

            {/* Error message */}
            {error && (
                <div className="px-6 py-2 bg-red-50 border-t border-red-200">
                    <p className="text-sm text-red-600">{error}</p>
                </div>
            )}

            {/* Footer */}
            <div className="p-6 border-t border-[#dcd3b8] bg-[#ebe5d5]">
                {isText && (citation as TextCitation).chunk_id ? (
                    <button
                        onClick={() =>
                            context
                                ? setContext(null)
                                : handleViewContext((citation as TextCitation).chunk_id!)
                        }
                        disabled={loading}
                        className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-primary text-white font-sans font-bold rounded-lg hover:bg-primary-dark transition-colors shadow-sm disabled:opacity-50"
                    >
                        {loading ? (
                            <Loader2 size={18} className="animate-spin" />
                        ) : (
                            <ExternalLink size={18} />
                        )}
                        {context ? "Close Full Context" : "View Full Context"}
                    </button>
                ) : (
                    <button
                        disabled
                        title={isText ? "No chunk ID available" : "Graph visualization coming soon"}
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

function FullContextViewer({ context }: { context: ChunkContext }) {
    const markRef = useRef<HTMLElement>(null);

    useEffect(() => {
        if (markRef.current) {
            markRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
        }
    }, [context]);

    const before = context.full_chapter_text.slice(0, context.highlight_start);
    const highlighted = context.full_chapter_text.slice(
        context.highlight_start,
        context.highlight_end
    );
    const after = context.full_chapter_text.slice(context.highlight_end);

    return (
        <div className="border-t border-[#dcd3b8] bg-white/30">
            {/* Chapter header */}
            <div className="px-6 py-3 bg-[#ebe5d5]/50 border-b border-[#dcd3b8]">
                <p className="font-sans text-xs font-semibold text-[#8c8578] uppercase">
                    Full Chapter Context
                </p>
                <p className="font-serif text-sm text-parchment-text">
                    {context.book} — {context.chapter}
                </p>
                <p className="font-sans text-xs text-[#8c8578] mt-1">
                    Chunk {context.chunk_index} of {context.total_chunks_in_chapter}
                </p>
            </div>

            {/* Scrollable full text with highlight */}
            <div className="px-6 py-4 max-h-[50vh] overflow-y-auto">
                <p className="font-serif text-base text-parchment-text leading-relaxed whitespace-pre-wrap">
                    {before}
                    <mark ref={markRef} className="bg-primary/20 text-parchment-text px-0.5 rounded">
                        {highlighted}
                    </mark>
                    {after}
                </p>
            </div>
        </div>
    );
}

function TextCitationContent({ citation }: { citation: TextCitation }) {
    return (
        <div className="space-y-6">
            <div className="space-y-2">
                <h3 className="font-sans text-sm font-semibold text-[#8c8578] uppercase">Source Chapter</h3>
                <p className="font-serif text-2xl text-parchment-text border-b-2 border-primary/20 pb-2 inline-block">
                    {cleanSourceLabel(citation.source)}
                </p>
            </div>

            <div className="space-y-3">
                <h3 className="font-sans text-sm font-semibold text-[#8c8578] uppercase">Passage Content</h3>
                <div className="relative pl-6">
                    <div className="absolute left-0 top-0 bottom-0 w-1 bg-primary/50 rounded-full" />
                    <p className="font-serif text-lg text-parchment-text leading-loose whitespace-pre-wrap">
                        {citation.content}
                    </p>
                </div>
            </div>

            <div className="flex gap-2">
                <span className="inline-flex items-center px-2 py-1 rounded bg-[#dcd3b8]/50 text-[#5c5548] text-xs font-medium">
                    Rel: {(Math.exp(-citation.score / 1000) * 100).toFixed(1)}%
                </span>
            </div>
        </div>
    );
}

function GraphCitationContent({ citation }: { citation: GraphCitation }) {
    return (
        <div className="space-y-6">
            <div className="space-y-2">
                <h3 className="font-sans text-sm font-semibold text-[#8c8578] uppercase">Fact Relationship</h3>
                <p className="font-serif text-xl text-parchment-text p-4 bg-white/50 rounded-lg border border-[#dcd3b8] text-center">
                    {citation.fact}
                </p>
            </div>

            <div className="space-y-3">
                <h3 className="font-sans text-sm font-semibold text-[#8c8578] uppercase">Traversal Metadata</h3>
                <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 bg-[#f5f5f5] rounded-lg border border-gray-200">
                        <p className="text-xs text-gray-500 uppercase">Depth</p>
                        <p className="text-lg font-bold text-gray-800">{citation.depth}-hop</p>
                    </div>
                </div>
            </div>

            {citation.source_ref && (
                <div className="space-y-3">
                    <h3 className="font-sans text-sm font-semibold text-[#8c8578] uppercase">Provenance</h3>
                    <pre className="p-3 bg-gray-100 rounded text-xs overflow-x-auto text-gray-700">
                        {JSON.stringify(citation.source_ref, null, 2)}
                    </pre>
                </div>
            )}
        </div>
    );
}
