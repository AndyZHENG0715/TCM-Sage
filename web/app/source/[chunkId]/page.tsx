"use client";

import { ChunkContext, fetchChunkContext } from "@/lib/api";
import { getDisplaySourceLabel, getOcrArtifacts } from "@/lib/citations";
import { useI18n } from "@/i18n/context";
import { ArrowLeft, BookOpen, FileText, LayoutList, Loader2 } from "lucide-react";
import Link from "next/link";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";

function getBackendUrl(): string {
    return "/api/backend";
}

function HighlightedText({
    text,
    highlightStart,
    highlightEnd,
    markRef,
}: {
    text: string;
    highlightStart: number;
    highlightEnd: number;
    markRef: React.RefObject<HTMLElement | null>;
}) {
    const before = text.slice(0, highlightStart);
    const highlighted = text.slice(highlightStart, highlightEnd);
    const after = text.slice(highlightEnd);

    return (
        <p className="whitespace-pre-wrap text-justify antialiased">
            {before}
            {highlighted && (
                <mark
                    ref={markRef}
                    className="bg-primary/30 text-primary-dark px-1 rounded-sm shadow-[0_0_8px_rgba(139,90,43,0.3)] transition-all duration-500 ease-in-out"
                >
                    {highlighted}
                </mark>
            )}
            {after}
        </p>
    );
}

function SourceDocument({ chunkId }: { chunkId: string }) {
    const router = useRouter();
    const searchParams = useSearchParams();
    const { t } = useI18n();
    const fromChat = searchParams.get("from") === "chat";
    const [context, setContext] = useState<ChunkContext | null>(null);
    const [fullText, setFullText] = useState<string | null>(null);
    const [viewMode, setViewMode] = useState<"chapter" | "full">("chapter");
    const [error, setError] = useState<string | null>(null);
    const markRef = useRef<HTMLElement>(null);

    const handleBack = () => {
        const hasOpener = window.opener !== null;
        const historyLength = window.history.length;

        if (fromChat) {
            window.close();
            return;
        }

        // New tab flow: prefer closing the child tab.
        if (hasOpener) {
            window.close();
            return;
        }

        if (historyLength > 1) {
            router.back();
            return;
        }

        // Last fallback for direct URL entry/no history.
        router.push("/");
    };

    useEffect(() => {
        let isCancelled = false;

        const loadSourceData = async () => {
            try {
                const data = await fetchChunkContext(chunkId);
                if (isCancelled) {
                    return;
                }

                setContext(data);

                const res = await fetch(
                    `${getBackendUrl()}/books/${encodeURIComponent(data.book)}`
                );
                if (!res.ok) {
                    throw new Error(`Failed to fetch book: ${res.status}`);
                }

                const payload = (await res.json()) as { content?: string };
                if (!isCancelled) {
                    setFullText(typeof payload.content === "string" ? payload.content : null);
                }
            } catch (fetchError) {
                if (!isCancelled) {
                    setError(
                        fetchError instanceof Error
                            ? fetchError.message
                            : t.common.loading
                    );
                }
            }
        };

        void loadSourceData();

        return () => {
            isCancelled = true;
        };
    }, [chunkId, t.common.loading]);

    useEffect(() => {
        if (!context || !markRef.current) {
            return;
        }

        const frameId = window.requestAnimationFrame(() => {
            markRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
        });

        return () => window.cancelAnimationFrame(frameId);
    }, [context, viewMode]);

    const loading = !context && !error;

    if (loading) {
        return (
            <div className="min-h-screen bg-parchment flex items-center justify-center p-8" suppressHydrationWarning>
                <div className="flex flex-col items-center gap-4 text-[#8c8578]">
                    <Loader2 size={40} className="animate-spin text-primary" suppressHydrationWarning />
                    <p className="font-sans text-lg animate-pulse">{t.source.loading}</p>
                </div>
            </div>
        );
    }

    if (error || !context) {
        return (
            <div className="min-h-screen bg-parchment flex flex-col items-center justify-center p-8" suppressHydrationWarning>
                <div className="max-w-md text-center space-y-6">
                    <div className="inline-flex p-4 rounded-full bg-red-100 text-red-600">
                        <BookOpen size={48} />
                    </div>
                    <div className="space-y-2">
                        <h1 className="text-2xl font-serif font-bold text-red-800">{t.source.errorTitle}</h1>
                        <p className="text-red-600/80 font-sans">{error || t.source.errorMessage}</p>
                    </div>
                    <Link
                        href="/"
                        className="inline-flex items-center gap-2 px-6 py-3 bg-[#dcd3b8] hover:bg-[#cabe9e] text-[#5c5548] font-bold rounded-lg transition-colors shadow-sm"
                    >
                        <ArrowLeft size={20} />
                        {t.source.returnToStudy}
                    </Link>
                </div>
            </div>
        );
    }

    const chapterTitle = getDisplaySourceLabel(
        context.chapter,
        context.chapter_display
    );
    const ocrArtifacts = getOcrArtifacts(context.full_chapter_text);

    // Try to find the highlight in the full book text if available
    let displayText = context.full_chapter_text;
    let hStart = context.highlight_start;
    let hEnd = context.highlight_end;

    if (viewMode === "full" && fullText) {
        displayText = fullText;
        // Search for the paragraph in the full book to find the offset
        const paragraphIndex = fullText.indexOf(context.paragraph_text);
        if (paragraphIndex !== -1) {
            hStart = paragraphIndex + context.paragraph_highlight_start;
            hEnd = paragraphIndex + context.paragraph_highlight_end;
        } else {
            // Fallback: try to find the full chapter text
            const chapterIndex = fullText.indexOf(context.full_chapter_text);
            if (chapterIndex !== -1) {
                hStart = chapterIndex + context.highlight_start;
                hEnd = chapterIndex + context.highlight_end;
            } else {
                // If not found, we can't highlight correctly in the full book view
                hStart = 0;
                hEnd = 0;
            }
        }
    }

    return (
        <div className="min-h-screen bg-parchment selection:bg-primary/20 selection:text-primary-dark" suppressHydrationWarning>
            <div className="sticky top-0 z-10 w-full bg-[#ebe5d5]/90 backdrop-blur-sm border-b border-[#dcd3b8] shadow-sm">
                <div className="max-w-4xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
                    <button
                        onClick={handleBack}
                        className="flex items-center gap-2 text-[#8c8578] hover:text-primary-dark transition-colors group"
                    >
                        <ArrowLeft size={20} className="transform group-hover:-translate-x-1 transition-transform" />
                        <span className="font-sans font-semibold">{t.source.backToSession}</span>
                    </button>

                    <div className="flex flex-col items-end">
                        <span className="font-sans text-xs font-bold text-[#8c8578] uppercase tracking-wider">
                            {t.source.sourceDocument}
                        </span>
                        <span className="font-serif text-lg font-bold text-parchment-text leading-none">
                            {context.book}
                        </span>
                    </div>
                </div>
            </div>

            <main className="max-w-4xl mx-auto px-4 sm:px-6 py-12 pb-32">
                <div className="space-y-8">
                    <header className="text-center space-y-4 mb-8 pb-8 border-b border-[#dcd3b8]/50 inline-block w-full">
                        <h1 className="font-serif text-4xl sm:text-5xl font-bold text-primary-dark tracking-wide">
                            {viewMode === "chapter" ? (chapterTitle || t.citation.textSource) : context.book}
                        </h1>
                        <div className="flex items-center justify-center gap-3 text-sm font-sans font-medium text-[#c0b59a]">
                            <span>§</span>
                            <span className="uppercase tracking-widest">{context.book}</span>
                            <span>§</span>
                        </div>
                    </header>

                    {fullText && (
                        <div className="flex justify-center mb-8">
                            <div className="inline-flex rounded-lg border border-[#dcd3b8] bg-[#ebe5d5] p-1">
                                <button
                                    onClick={() => setViewMode("chapter")}
                                    className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all ${
                                        viewMode === "chapter"
                                            ? "bg-white text-primary-dark shadow-sm font-semibold"
                                            : "text-[#8c8578] hover:text-primary-dark"
                                    }`}
                                >
                                    <LayoutList size={18} />
                                    <span>{t.source.chapterView}</span>
                                </button>
                                <button
                                    onClick={() => setViewMode("full")}
                                    className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all ${
                                        viewMode === "full"
                                            ? "bg-white text-primary-dark shadow-sm font-semibold"
                                            : "text-[#8c8578] hover:text-primary-dark"
                                    }`}
                                >
                                    <FileText size={18} />
                                    <span>{t.source.fullBook}</span>
                                </button>
                            </div>
                        </div>
                    )}

                    {ocrArtifacts.length > 0 && viewMode === "chapter" && (
                        <div className="rounded-lg border border-amber-300/60 bg-amber-50/70 px-4 py-3 text-sm text-amber-900">
                            {t.source.ocrWarning} {ocrArtifacts.join(", ")}
                        </div>
                    )}

                    <article className="prose prose-lg max-w-none text-parchment-text font-serif leading-[2.2] md:leading-[2.5] text-lg sm:text-xl">
                        <HighlightedText
                            text={displayText}
                            highlightStart={hStart}
                            highlightEnd={hEnd}
                            markRef={markRef}
                        />
                    </article>
                </div>
            </main>
        </div>
    );
}

export default function SourcePage() {
    const params = useParams();
    const chunkId = params.chunkId as string | undefined;

    if (!chunkId) {
        return null;
    }

    return <SourceDocument key={chunkId} chunkId={chunkId} />;
}
