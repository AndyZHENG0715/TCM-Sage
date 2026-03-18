"use client";

import { cleanSourceLabel } from "@/lib/citations";
import { Message, Citation } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Copy, ThumbsUp, Info, AlertTriangle, Check } from "lucide-react";
import { useState, useMemo, type ComponentPropsWithoutRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MessageBubbleProps {
    message: Message;
    onCitationClick?: (citation: Citation) => void;
}

/** Sentinel prefix used to mark citation placeholders inside markdown. */
const CITE_PREFIX = "%%CITE_";
const CITE_SUFFIX = "%%";

export function MessageBubble({
    message,
    onCitationClick,
}: MessageBubbleProps) {
    const isUser = message.role === "user";
    const [copied, setCopied] = useState(false);

    const handleCopy = () => {
        navigator.clipboard.writeText(message.content);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    /**
     * Pre-process assistant content: replace `[n]` citation refs with
     * inline code placeholders that ReactMarkdown won't mangle.
     * e.g. `[1]` -> `` `%%CITE_1%%` ``
     */
    const processedContent = useMemo(() => {
        if (isUser) return message.content;
        return message.content.replace(
            /\[(\d+)\]/g,
            (_match, num) => `\`${CITE_PREFIX}${num}${CITE_SUFFIX}\``
        );
    }, [message.content, isUser]);

    /** Custom renderers for ReactMarkdown. */
    const markdownComponents = useMemo(() => ({
        /* Intercept inline <code> to render citation badges */
        code({ children, ...props }: ComponentPropsWithoutRef<"code">) {
            const text = String(children).trim();
            if (text.startsWith(CITE_PREFIX) && text.endsWith(CITE_SUFFIX)) {
                const num = parseInt(text.slice(CITE_PREFIX.length, -CITE_SUFFIX.length));
                const citation = message.citations?.find((c) => c.number === num);
                if (citation) {
                    return (
                        <button
                            onClick={() => onCitationClick?.(citation)}
                            className="inline-flex items-center justify-center mx-1 px-1.5 py-0.5 text-xs font-sans font-bold text-primary bg-primary/10 rounded-full hover:bg-primary/20 transition-colors cursor-pointer align-super"
                        >
                            {num}
                        </button>
                    );
                }
                return <span>[{num}]</span>;
            }
            return <code className="bg-black/5 px-1.5 py-0.5 rounded text-sm font-mono" {...props}>{children}</code>;
        },
        /* Style block-level elements for the parchment card */
        p({ children }: ComponentPropsWithoutRef<"p">) {
            return <p className="mb-3 last:mb-0 leading-relaxed">{children}</p>;
        },
        h1({ children }: ComponentPropsWithoutRef<"h1">) {
            return <h1 className="text-2xl font-bold mb-3 mt-4 first:mt-0">{children}</h1>;
        },
        h2({ children }: ComponentPropsWithoutRef<"h2">) {
            return <h2 className="text-xl font-bold mb-2 mt-3 first:mt-0">{children}</h2>;
        },
        h3({ children }: ComponentPropsWithoutRef<"h3">) {
            return <h3 className="text-lg font-semibold mb-2 mt-3 first:mt-0">{children}</h3>;
        },
        ul({ children }: ComponentPropsWithoutRef<"ul">) {
            return <ul className="list-disc pl-6 mb-3 space-y-1">{children}</ul>;
        },
        ol({ children }: ComponentPropsWithoutRef<"ol">) {
            return <ol className="list-decimal pl-6 mb-3 space-y-1">{children}</ol>;
        },
        li({ children }: ComponentPropsWithoutRef<"li">) {
            return <li className="leading-relaxed">{children}</li>;
        },
        blockquote({ children }: ComponentPropsWithoutRef<"blockquote">) {
            return <blockquote className="border-l-4 border-primary/40 pl-4 italic my-3 text-parchment-text/80">{children}</blockquote>;
        },
        strong({ children }: ComponentPropsWithoutRef<"strong">) {
            return <strong className="font-bold">{children}</strong>;
        },
        hr() {
            return <hr className="my-4 border-[#dcd3b8]" />;
        },
    }), [message.citations, onCitationClick]);

    // Render content
    const renderContent = () => {
        if (isUser) return <p className="whitespace-pre-wrap font-sans text-white">{message.content}</p>;

        return (
            <div className="font-serif text-parchment-text text-lg prose-parchment">
                <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                    {processedContent}
                </ReactMarkdown>
            </div>
        );
    };

    return (
        <div
            className={cn(
                "flex w-full mb-6",
                isUser ? "justify-end" : "justify-start"
            )}
        >
            <div
                className={cn(
                    "max-w-[85%] sm:max-w-[75%]",
                    isUser
                        ? "bg-primary/20 backdrop-blur-sm border border-primary/20 rounded-2xl rounded-tr-sm p-4 text-parchment"
                        : "flex flex-col gap-3"
                )}
            >
                {!isUser && (
                    <div className="bg-parchment rounded-xl shadow-lg border border-[#e3dac3] p-6 relative overflow-hidden">
                        {/* Decorative texture or noise could go here */}

                        {/* Header with Severity Badge */}
                        <div className="flex justify-between items-start mb-3">
                            <div className="flex gap-2">
                                {message.severity === "informational" && (
                                    <span className="inline-flex items-center gap-1 px-2 py-1 rounded bg-blue-100/50 text-blue-800 text-xs font-bold border border-blue-200">
                                        <Info size={12} /> INFORMATIONAL
                                    </span>
                                )}
                                {message.severity === "prescriptive" && (
                                    <span className="inline-flex items-center gap-1 px-2 py-1 rounded bg-amber-100/50 text-amber-800 text-xs font-bold border border-amber-200">
                                        <AlertTriangle size={12} /> CLINICAL CONTEXT
                                    </span>
                                )}
                            </div>
                        </div>

                        {/* Content */}
                        <div className="text-parchment-text">
                            {renderContent()}
                        </div>

                        {/* Sources Footer */}
                        {message.citations && message.citations.length > 0 && (
                            <div className="mt-6 pt-4 border-t border-[#dcd3b8]">
                                <p className="text-xs font-sans font-semibold text-[#8c8578] mb-2 uppercase tracking-wider">Sources:</p>
                                <div className="flex flex-wrap gap-2">
                                    {message.citations.map((citation) => (
                                        <button
                                            key={citation.number}
                                            onClick={() => onCitationClick?.(citation)}
                                            className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white/50 border border-[#dcd3b8] rounded-md hover:bg-white hover:border-primary/50 transition-all group max-w-full"
                                        >
                                            <span className="flex items-center justify-center w-4 h-4 text-[10px] font-bold text-white bg-primary rounded-full group-hover:bg-primary-dark">
                                                {citation.number}
                                            </span>
                                            <span className="text-xs text-[#5c5548] truncate max-w-[150px]">
                                                {citation.type === "text" ? cleanSourceLabel(citation.source) : citation.fact}
                                            </span>
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {isUser && renderContent()}

                {/* Action Buttons for Assistant */}
                {!isUser && message.content && (
                    <div className="flex items-center gap-2 mt-1 ml-2">
                        <button
                            onClick={handleCopy}
                            className="p-1.5 text-gray-500 hover:text-primary transition-colors rounded-full hover:bg-primary/10"
                            title="Copy response"
                        >
                            {copied ? <Check size={16} /> : <Copy size={16} />}
                        </button>
                        <button
                            className="p-1.5 text-gray-500 hover:text-primary transition-colors rounded-full hover:bg-primary/10"
                            title="Helpful"
                        >
                            <ThumbsUp size={16} />
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}
