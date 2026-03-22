"use client";

import { useEffect, useState, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import { fetchChunkContext, ChunkContext } from "@/lib/api";
import { ArrowLeft, Loader2, BookOpen } from "lucide-react";

export default function SourcePage() {
    const params = useParams();
    const router = useRouter();
    const chunkId = params.chunkId as string;

    const [context, setContext] = useState<ChunkContext | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const markRef = useRef<HTMLElement>(null);

    useEffect(() => {
        if (!chunkId) return;

        setLoading(true);
        fetchChunkContext(chunkId)
            .then((data) => setContext(data))
            .catch((err) =>
                setError(err instanceof Error ? err.message : "Failed to load chapter context")
            )
            .finally(() => setLoading(false));
    }, [chunkId]);

    useEffect(() => {
        if (context && markRef.current) {
            // Scroll the highlighted section into view smoothly after a short delay
            // to ensure rendering is complete
            setTimeout(() => {
                markRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
            }, 100);
        }
    }, [context]);

    if (loading) {
        return (
            <div className="min-h-screen bg-parchment flex items-center justify-center p-8">
                <div className="flex flex-col items-center gap-4 text-[#8c8578]">
                    <Loader2 size={40} className="animate-spin text-primary" />
                    <p className="font-sans text-lg animate-pulse">Unrolling scrolls...</p>
                </div>
            </div>
        );
    }

    if (error || !context) {
        return (
            <div className="min-h-screen bg-parchment flex flex-col items-center justify-center p-8">
                <div className="max-w-md text-center space-y-6">
                    <div className="inline-flex p-4 rounded-full bg-red-100 text-red-600">
                        <BookOpen size={48} />
                    </div>
                    <div className="space-y-2">
                        <h1 className="text-2xl font-serif font-bold text-red-800">Cannot Read Scroll</h1>
                        <p className="text-red-600/80 font-sans">{error || "Scroll not found."}</p>
                    </div>
                    <button
                        onClick={() => router.back()}
                        className="inline-flex items-center gap-2 px-6 py-3 bg-[#dcd3b8] hover:bg-[#cabe9e] text-[#5c5548] font-bold rounded-lg transition-colors shadow-sm"
                    >
                        <ArrowLeft size={20} />
                        Return to Study
                    </button>
                </div>
            </div>
        );
    }

    const { full_chapter_text, highlight_start, highlight_end, book, chapter } = context;
    const before = full_chapter_text.slice(0, highlight_start);
    const highlighted = full_chapter_text.slice(highlight_start, highlight_end);
    const after = full_chapter_text.slice(highlight_end);

    return (
        <div className="min-h-screen bg-parchment selection:bg-primary/20 selection:text-primary-dark">
            {/* Header Navbar */}
            <div className="sticky top-0 z-10 w-full bg-[#ebe5d5]/90 backdrop-blur-sm border-b border-[#dcd3b8] shadow-sm">
                <div className="max-w-4xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
                    <button
                        onClick={() => router.back()}
                        className="flex items-center gap-2 text-[#8c8578] hover:text-primary-dark transition-colors group"
                    >
                        <ArrowLeft size={20} className="transform group-hover:-translate-x-1 transition-transform" />
                        <span className="font-sans font-semibold">Back to Session</span>
                    </button>
                    
                    <div className="flex flex-col items-end">
                        <span className="font-sans text-xs font-bold text-[#8c8578] uppercase tracking-wider">
                            Source Document
                        </span>
                        <span className="font-serif text-lg font-bold text-parchment-text leading-none">
                            {book}
                        </span>
                    </div>
                </div>
            </div>

            {/* Main Content Area */}
            <main className="max-w-4xl mx-auto px-4 sm:px-6 py-12 pb-32">
                <div className="space-y-8">
                    {/* Chapter Title */}
                    <header className="text-center space-y-4 mb-16 pb-12 border-b border-[#dcd3b8]/50 inline-block w-full">
                        <h1 className="font-serif text-4xl sm:text-5xl font-bold text-primary-dark tracking-wide">
                            {chapter}
                        </h1>
                        <div className="flex items-center justify-center gap-3 text-sm font-sans font-medium text-[#c0b59a]">
                            <span>§</span>
                            <span className="uppercase tracking-widest">{book}</span>
                            <span>§</span>
                        </div>
                    </header>

                    {/* Scroll Content */}
                    <article className="prose prose-lg max-w-none text-parchment-text font-serif leading-[2.2] md:leading-[2.5] text-lg sm:text-xl">
                        <p className="whitespace-pre-wrap text-justify antialiased">
                            {before}
                            <mark
                                ref={markRef}
                                className="bg-primary/30 text-primary-dark px-1 rounded-sm shadow-[0_0_8px_rgba(139,90,43,0.3)] transition-all duration-500 ease-in-out"
                            >
                                {highlighted}
                            </mark>
                            {after}
                        </p>
                    </article>
                </div>
            </main>
        </div>
    );
}
