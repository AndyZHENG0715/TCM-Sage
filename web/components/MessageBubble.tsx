"use client";

import { useMemo, useState, type ComponentPropsWithoutRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Copy, ThumbsUp, Info, AlertTriangle, Check } from "lucide-react";
import { Citation, Message } from "@/lib/types";
import { getDisplaySourceLabel } from "@/lib/citations";
import { cn } from "@/lib/utils";

interface MessageBubbleProps {
    message: Message;
    onCitationClick?: (citation: Citation) => void;
}

const CITE_PREFIX = "%%CITE_";
const CITE_SUFFIX = "%%";
const SOURCE_HEADER_RE = /^\s*(?:\*\*)?(?:Sources?|References?)(?:\*\*)?\s*:?\s*$/i;
const SOURCE_LIST_RE = /^\s*(?:[-*•]|\d+[.)])\s+/;
const SOURCE_ENTRY_RE = /^\s*(?:\[\d+\]|Source\s*:|KG Fact\s*:)/i;

function stripTrailingReferenceSection(content: string): string {
    const trimmedContent = content.trimEnd();
    const lines = trimmedContent.split(/\r?\n/);

    // Search from the end for a line that looks like a "Sources" header
    for (let index = lines.length - 1; index >= 0; index -= 1) {
        const line = lines[index].trim();
        if (!SOURCE_HEADER_RE.test(line)) {
            continue;
        }

        // Check if all subsequent non-empty lines are source entries
        const tailLines = lines.slice(index + 1).filter((l) => l.trim().length > 0);
        const isReferenceTail =
            tailLines.length === 0 ||
            tailLines.every((l) => {
                const trimmedLine = l.trim();
                return (
                    SOURCE_LIST_RE.test(trimmedLine) ||
                    SOURCE_ENTRY_RE.test(trimmedLine) ||
                    // Also handle [1], [2] etc without the "Source:" prefix
                    /^\[\d+\]/.test(trimmedLine)
                );
            });

        if (isReferenceTail) {
            return lines.slice(0, index).join("\n").trimEnd();
        }
    }

    // Fallback: Check for a very common pattern where the header and list are together at the end
    // but might have been missed by the line-by-line check
    const multiLineMatch = trimmedContent.match(
        /(?:\n\n|\r\n\r\n)(?:\*\*)?(?:Sources?|References?)(?:\*\*)?\s*:?\s*\n(?:\s*[-*•\d\[].*[\n\r]*)*$/i
    );
    if (multiLineMatch) {
        return trimmedContent.slice(0, multiLineMatch.index).trimEnd();
    }

    return trimmedContent;
}

function normalizeQuotedBoldMarkdown(content: string): string {
    return content
        // Remove spaces inside bold markers: ** text ** -> **text**
        .replace(/\*\*\s+/g, "**")
        .replace(/\s+\*\*/g, "**")
        // Ensure no spaces between bold markers and common Chinese/English quotes
        .replace(/\*\*(?=["“「『])/g, "**")
        .replace(/(?<=[”」』"])\*\*/g, "**")
        // Re-insert single space OUTSIDE bold markers if they are adjacent to other text 
        // (but only for English text to avoid breaking Chinese rendering which doesn't need spaces)
        // Actually, let's keep it simple as per the plan's focus on quotes.
        .replace(/([“「『])\*\*/g, "$1**")
        .replace(/\*\*([”」』])/g, "**$1");
}

function postProcessAssistantContent(content: string): string {
    return normalizeQuotedBoldMarkdown(stripTrailingReferenceSection(content)).replace(
        /\[(\d+)\]/g,
        (_match, number) => `\`${CITE_PREFIX}${number}${CITE_SUFFIX}\``
    );
}

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

    const processedContent = useMemo(() => {
        if (isUser) {
            return message.content;
        }

        return postProcessAssistantContent(message.content);
    }, [isUser, message.content]);

    const markdownComponents = useMemo(() => ({
        code({ children, ...props }: ComponentPropsWithoutRef<"code">) {
            const text = String(children).trim();
            if (text.startsWith(CITE_PREFIX) && text.endsWith(CITE_SUFFIX)) {
                const citationNumber = Number.parseInt(
                    text.slice(CITE_PREFIX.length, -CITE_SUFFIX.length),
                    10
                );
                const citation = message.citations?.find(
                    (item) => item.number === citationNumber
                );

                if (citation) {
                    return (
                        <button
                            onClick={() => onCitationClick?.(citation)}
                            className="inline-flex items-center justify-center mx-1 px-1.5 py-0.5 text-xs font-sans font-bold text-primary bg-primary/10 rounded-full hover:bg-primary/20 transition-colors cursor-pointer align-super"
                        >
                            {citationNumber}
                        </button>
                    );
                }

                return <span>[{citationNumber}]</span>;
            }

            return (
                <code
                    className="bg-black/5 px-1.5 py-0.5 rounded text-sm font-mono"
                    {...props}
                >
                    {children}
                </code>
            );
        },
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
            return (
                <blockquote className="border-l-4 border-primary/40 pl-4 italic my-3 text-parchment-text/80">
                    {children}
                </blockquote>
            );
        },
        strong({ children }: ComponentPropsWithoutRef<"strong">) {
            return <strong className="font-bold">{children}</strong>;
        },
        hr() {
            return <hr className="my-4 border-[#dcd3b8]" />;
        },
    }), [message.citations, onCitationClick]);

    const renderContent = () => {
        if (isUser) {
            return (
                <p className="whitespace-pre-wrap font-sans text-white">
                    {message.content}
                </p>
            );
        }

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

                        <div className="text-parchment-text">{renderContent()}</div>

                        {message.citations && message.citations.length > 0 && (
                            <div className="mt-6 pt-4 border-t border-[#dcd3b8]">
                                <p className="text-xs font-sans font-semibold text-[#8c8578] mb-2 uppercase tracking-wider">
                                    Sources:
                                </p>
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
                                                {citation.type === "text"
                                                    ? getDisplaySourceLabel(citation.source) || "Source excerpt"
                                                    : citation.fact}
                                            </span>
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {isUser && renderContent()}

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
