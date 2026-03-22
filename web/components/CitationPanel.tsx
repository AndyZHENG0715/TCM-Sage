"use client";

import { useState, useRef, useEffect } from "react";
import { Citation, TextCitation, GraphCitation } from "@/lib/types";
import { cleanSourceLabel } from "@/lib/citations";
import { X, BookOpen, ExternalLink, Loader2 } from "lucide-react";
import { fetchChunkContext, ChunkContext } from "@/lib/api";
import Link from "next/link";
import { KGViewer } from "./KGViewer";

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
            
            if (citation && citation.type === "text" && (citation as TextCitation).chunk_id) {
                const chunkId = (citation as TextCitation).chunk_id!;
                setLoading(true);
                fetchChunkContext(chunkId)
                    .then(data => setContext(data))
                    .catch(err => setError(err instanceof Error ? err.message : "Failed to load context"))
                    .finally(() => setLoading(false));
            }
        }
    }, [citation]);

    if (!citation) return null;

    const isText = citation.type === "text";

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
                    <TextCitationContent citation={citation as TextCitation} context={context} loading={loading} />
                ) : (
                    <GraphCitationContent citation={citation as GraphCitation} />
                )}
            </div>

            {/* Error message */}
            {error && (
                <div className="px-6 py-2 bg-red-50 border-t border-red-200">
                    <p className="text-sm text-red-600">{error}</p>
                </div>
            )}

            {/* Footer */}
            <div className="p-6 border-t border-[#dcd3b8] bg-[#ebe5d5]">
                {isText && (citation as TextCitation).chunk_id ? (
                    <Link
                        href={`/source/${(citation as TextCitation).chunk_id}`}
                        target="_blank"
                        className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-primary text-white font-sans font-bold rounded-lg hover:bg-primary-dark transition-colors shadow-sm disabled:opacity-50"
                    >
                        {loading ? (
                            <Loader2 size={18} className="animate-spin" />
                        ) : (
                            <ExternalLink size={18} />
                        )}
                        View Full Context
                    </Link>
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

function TextCitationContent({ citation, context, loading }: { citation: TextCitation, context: ChunkContext | null, loading: boolean }) {
    let passageContent = <>{citation.content}</>;
    
    if (loading) {
        passageContent = (
            <div className="flex items-center gap-2 text-[#8c8578]">
                <Loader2 size={16} className="animate-spin" />
                <span className="text-sm italic">Loading full paragraph context...</span>
            </div>
        );
    } else if (context) {
        const { full_chapter_text, highlight_start, highlight_end } = context;
        const before = full_chapter_text.slice(0, highlight_start);
        const highlighted = full_chapter_text.slice(highlight_start, highlight_end);
        const after = full_chapter_text.slice(highlight_end);
        
        const paraStart = before.lastIndexOf("\n") + 1;
        let paraEnd = after.indexOf("\n");
        if (paraEnd === -1) paraEnd = after.length;
        
        const paragraphBefore = before.slice(paraStart);
        const paragraphAfter = after.slice(0, paraEnd);
        
        passageContent = (
            <>
                {paragraphBefore}
                <mark className="bg-primary/20 text-parchment-text px-0.5 rounded">
                    {highlighted}
                </mark>
                {paragraphAfter}
            </>
        );
    }

    const sourceDisplay = context ? `${context.book} — ${context.chapter}` : cleanSourceLabel(citation.source);

    return (
        <div className="space-y-6">
            <div className="space-y-2">
                <h3 className="font-sans text-sm font-semibold text-[#8c8578] uppercase">Source Chapter</h3>
                <p className="font-serif text-2xl text-parchment-text border-b-2 border-primary/20 pb-2 inline-block">
                    {sourceDisplay}
                </p>
            </div>

            <div className="space-y-3">
                <h3 className="font-sans text-sm font-semibold text-[#8c8578] uppercase">Passage Content</h3>
                <div className="relative pl-6">
                    <div className="absolute left-0 top-0 bottom-0 w-1 bg-primary/50 rounded-full" />
                    <p className="font-serif text-lg text-parchment-text leading-loose whitespace-pre-wrap">
                        {passageContent}
                    </p>
                </div>
            </div>

            <div className="flex gap-2">
                <span className="inline-flex items-center px-2 py-1 rounded bg-[#dcd3b8]/50 text-[#5c5548] text-xs font-medium">
                    Rel: {Math.max(0, (1 - citation.score / 4) * 100).toFixed(1)}%
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
                <KGViewer citation={citation} />
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
