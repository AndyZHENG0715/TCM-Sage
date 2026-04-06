"use client";

import { getDisplaySourceLabel } from "@/lib/citations";
import { useI18n } from "@/i18n/context";
import {
    createMarkdownComponents,
    postProcessAssistantContent,
} from "@/lib/markdown";
import { Citation, Message } from "@/lib/types";
import { cn } from "@/lib/utils";
import { AlertTriangle, Check, CheckCircle2, Copy, Info, ThumbsUp } from "lucide-react";
import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MessageBubbleProps {
    message: Message;
    onCitationClick?: (citation: Citation) => void;
}

export function MessageBubble({
    message,
    onCitationClick,
}: MessageBubbleProps) {
    const isUser = message.role === "user";
    const [copied, setCopied] = useState(false);
    const { t } = useI18n();

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

    const markdownComponents = useMemo(
        () => createMarkdownComponents(message.citations, onCitationClick),
        [message.citations, onCitationClick]
    );

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
                                        <Info size={12} /> {t.verification.informational}
                                    </span>
                                )}
                                {message.severity === "prescriptive" && (
                                    <span className="inline-flex items-center gap-1 px-2 py-1 rounded bg-amber-100/50 text-amber-800 text-xs font-bold border border-amber-200">
                                        <AlertTriangle size={12} /> {t.verification.clinicalContext}
                                    </span>
                                )}
                            </div>
                        </div>

                        <div className="text-parchment-text">{renderContent()}</div>

                        {message.verification && (
                            <div
                                className={cn(
                                    "mt-2 inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium",
                                    message.verification.status === "SUPPORTED"
                                        ? "bg-emerald-500/10 text-emerald-400"
                                        : "bg-amber-500/10 text-amber-400"
                                )}
                            >
                                {message.verification.status === "SUPPORTED" ? (
                                    <CheckCircle2 size={12} />
                                ) : (
                                    <AlertTriangle size={12} />
                                )}
                                {message.verification.status === "SUPPORTED"
                                    ? t.verification.supported
                                    : t.verification.unsupported}
                            </div>
                        )}

                        {message.citations && message.citations.length > 0 && (
                            <div className="mt-6 pt-4 border-t border-[#dcd3b8]">
                                <p className="text-xs font-sans font-semibold text-[#8c8578] mb-2 uppercase tracking-wider">
                                    {t.citation.sources}
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
                                                        ? getDisplaySourceLabel(citation.source) || t.citation.textSource
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
                            title={copied ? t.common.copied : t.chat.copyResponse}
                            aria-label={copied ? t.common.copied : t.chat.copyResponse}
                        >
                            {copied ? <Check size={16} /> : <Copy size={16} />}
                        </button>
                        <button
                            className="p-1.5 text-gray-500 hover:text-primary transition-colors rounded-full hover:bg-primary/10"
                            title={t.chat.helpful}
                            aria-label={t.chat.helpful}
                        >
                            <ThumbsUp size={16} />
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}
