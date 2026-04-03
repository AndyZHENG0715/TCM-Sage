"use client";

import { Message, Citation } from "@/lib/types";
import { MessageBubble } from "./MessageBubble";
import { ChatInput } from "./ChatInput";
import { useEffect, useRef } from "react";

interface ChatAreaProps {
    messages: Message[];
    isStreaming: boolean;
    title: string | null;
    onSend: (message: string) => void;
    onCitationClick: (citation: Citation) => void;
}

export function ChatArea({
    messages,
    isStreaming,
    title,
    onSend,
    onCitationClick,
}: ChatAreaProps) {
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) {
            // Auto scroll to bottom
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages, isStreaming]);

    return (
        <div className="flex flex-col h-full bg-background-dark relative">
            {/* Header */}
            <div className="px-6 py-4 border-b border-white/5 bg-background-dark/80 backdrop-blur-md sticky top-0 z-10">
                <h2 className="text-lg font-serif font-medium text-parchment truncate text-center">
                    {title || "New Investigation"}
                </h2>
            </div>

            {/* Messages */}
            <div
                ref={scrollRef}
                className="flex-1 overflow-y-auto w-full px-4 sm:px-6 py-6 scroll-smooth"
            >
                <div className="w-full max-w-4xl mx-auto min-h-full">
                    {messages.filter((m) => m.role === "user" || m.content.length > 0).length === 0 ? (
                        <div className="flex flex-col items-center justify-center h-full text-center select-none pb-20">
                            <div className="w-24 h-24 rounded-full bg-gradient-to-br from-primary/20 to-transparent flex items-center justify-center mb-6">
                                <span className="text-4xl">☯</span>
                            </div>
                            <h3 className="text-xl font-serif font-bold text-parchment mb-2">
                                TCM-Sage Research Assistant
                            </h3>
                            <p className="text-sm text-gray-400 max-w-md mb-6">
                                Ask clinical questions, explore classic texts, and synthesize TCM knowledge with verifiable citations.
                            </p>
                            <div className="flex flex-wrap justify-center gap-2 max-w-lg">
                                {[
                                    { label: "水蛭性味功效", query: "水蛭的性味是什么？出自哪本经典？主治什么？" },
                                    { label: "蛇床子配伍禁忌", query: "蛇床子有哪些配伍禁忌？恶什么药？" },
                                    { label: "麻黄汤组成剂量", query: "麻黄汤的完整药物组成和剂量是什么？请引用原文。" },
                                    { label: "独活寄生汤全方", query: "独活寄生汤有多少味药？请全部列出并注明剂量。" },
                                    { label: "丹溪痰证治法", query: "《丹溪心法》中关于痰证的治法有哪些？分别用什么药？" },
                                ].map((item) => (
                                    <button
                                        key={item.label}
                                        type="button"
                                        onClick={() => onSend(item.query)}
                                        className="text-xs px-3 py-1.5 rounded-full border border-primary/20 text-primary/70 hover:text-primary hover:bg-primary/10 transition-colors"
                                    >
                                        {item.label}
                                    </button>
                                ))}
                            </div>
                        </div>
                    ) : (
                        messages
                            .filter((msg) => msg.role === "user" || msg.content.length > 0)
                            .map((msg, idx) => (
                                <MessageBubble
                                    key={idx}
                                    message={msg}
                                    onCitationClick={onCitationClick}
                                />
                            ))
                    )}

                    {/* Streaming Indicator */}
                    {isStreaming && (
                        <div className="flex justify-start w-full mb-6">
                            <div className="flex items-center gap-1.5 p-4 bg-parchment/10 rounded-2xl rounded-tl-sm">
                                <div className="w-2 h-2 bg-primary rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                                <div className="w-2 h-2 bg-primary rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                                <div className="w-2 h-2 bg-primary rounded-full animate-bounce"></div>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Input */}
            <div className="w-full bg-gradient-to-t from-background-dark to-transparent pt-4">
                <ChatInput onSend={onSend} isLoading={isStreaming} />
            </div>
        </div>
    );
}
