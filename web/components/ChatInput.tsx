"use client";

import { useState, useRef, useEffect } from "react";
import { Paperclip, ArrowUp } from "lucide-react";
import { cn } from "@/lib/utils";
import { useI18n } from "@/i18n/context";
import { ARENA_SAMPLE_PROMPTS } from "@/lib/arenaPrompts";

interface ChatInputProps {
    onSend: (message: string) => void;
    isLoading: boolean;
}


export function ChatInput({ onSend, isLoading }: ChatInputProps) {
    const [input, setInput] = useState("");
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const { t } = useI18n();

    const handleSubmit = () => {
        if (!input.trim() || isLoading) return;
        onSend(input);
        setInput("");
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    // Auto-resize textarea
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = "auto";
            textareaRef.current.style.height = `${Math.min(
                textareaRef.current.scrollHeight,
                200
            )}px`;
        }
    }, [input]);

    return (
        <div className="w-full max-w-4xl mx-auto px-4 pb-[calc(env(safe-area-inset-bottom)+1rem)]">
            {/* Sample Chips */}
            <div className="flex flex-wrap gap-2 mb-4 justify-center">
                {ARENA_SAMPLE_PROMPTS.slice(0, 5).map((item) => (
                    <button
                        key={item.label}
                        onClick={() => onSend(item.query)}
                        disabled={isLoading}
                        className="min-h-11 px-4 py-2 bg-background-dark/50 border border-primary/20 rounded-full text-sm text-parchment hover:bg-primary/10 hover:border-primary/50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {item.label}
                    </button>
                ))}
            </div>

            {/* Input Area */}
            <div className="relative bg-[#1a2c2a] rounded-xl border border-primary/20 shadow-lg focus-within:border-primary/50 transition-colors">
                <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={t.chat.inputPlaceholder}
                    rows={1}
                    disabled={isLoading}
                    className="w-full bg-transparent text-parchment placeholder-gray-500 p-4 pr-24 resize-none outline-none max-h-[200px] overflow-y-auto font-sans"
                />

                <div className="absolute right-2 bottom-2 flex items-center gap-1">
                    <button
                        disabled
                        className="min-h-11 min-w-11 p-2 text-gray-500 hover:text-parchment transition-colors rounded-lg cursor-not-allowed group relative"
                        title={t.chat.attachmentComingSoon}
                        aria-label={t.chat.attachmentComingSoon}
                    >
                        <Paperclip size={20} />
                        <span className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 px-2 py-1 bg-black text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">{t.chat.attachmentComingSoon}</span>
                    </button>

                    <button
                        onClick={handleSubmit}
                        disabled={!input.trim() || isLoading}
                        className={cn(
                            "min-h-11 min-w-11 p-2 rounded-lg transition-all duration-200",
                            input.trim() && !isLoading
                                ? "bg-primary text-background-dark hover:bg-primary-dark shadow-[0_0_10px_rgba(25,230,212,0.3)]"
                                : "bg-gray-700 text-gray-400 cursor-not-allowed"
                        )}
                        aria-label={t.arena.submit}
                    >
                        <ArrowUp size={20} />
                    </button>
                </div>
            </div>

            <p className="text-center text-xs text-gray-500 mt-2">
                {t.chat.disclaimer}
            </p>
        </div>
    );
}
