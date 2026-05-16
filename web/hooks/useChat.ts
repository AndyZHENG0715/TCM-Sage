"use client";

import { useCallback, useRef, useState } from "react";
import { Citation, CitationBounds, Message, Settings, Verification } from "@/lib/types";
import { streamQuery } from "@/lib/api";

function updateLastAssistantMessage(
    messages: Message[],
    updater: (message: Message) => Message
) {
    const next = [...messages];
    const lastMessage = next[next.length - 1];
    if (lastMessage?.role !== "assistant") {
        return messages;
    }

    next[next.length - 1] = updater(lastMessage);
    return next;
}

export function useChat(settings: Settings) {
    const [messages, setMessages] = useState<Message[]>([]);
    const [isStreaming, setIsStreaming] = useState(false);
    const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
    const [error, setError] = useState<string | null>(null);

    const abortControllerRef = useRef<AbortController | null>(null);

    const sendMessage = useCallback(
        async (content: string) => {
            if (!content.trim() || isStreaming) {
                return;
            }

            const chatHistory = messages.map((message) => ({
                role: message.role,
                content: message.content,
            }));

            const userMessage: Message = {
                role: "user",
                content: content.trim(),
                timestamp: Date.now(),
            };

            const assistantPlaceholder: Message = {
                role: "assistant",
                content: "",
                timestamp: Date.now(),
            };

            setMessages((prev) => [...prev, userMessage, assistantPlaceholder]);
            setIsStreaming(true);
            setError(null);

            try {
                abortControllerRef.current = new AbortController();

                const stream = streamQuery(content, chatHistory, settings);
                let fullContent = "";
                let citations: Citation[] = [];
                let citationBounds: CitationBounds | undefined;
                let severity: "informational" | "prescriptive" | undefined;
                let verification: Verification | undefined;

                for await (const event of stream) {
                    if (event.type === "text") {
                        fullContent += event.content;
                        setMessages((prev) =>
                            updateLastAssistantMessage(prev, (message) => ({
                                ...message,
                                content: fullContent,
                            }))
                        );
                        continue;
                    }

                    if (event.type === "metadata") {
                        citations = event.citations;
                        citationBounds = event.citationBounds;
                        severity = event.severity;
                        verification = event.verification;

                        setMessages((prev) =>
                            updateLastAssistantMessage(prev, (message) => ({
                                ...message,
                                citations,
                                citationBounds,
                                severity,
                                verification,
                            }))
                        );
                        continue;
                    }

                    setError(event.message || "The response stream reported an error.");
                    fullContent += `\n\n[Error: ${event.message}]`;
                    setMessages((prev) =>
                        updateLastAssistantMessage(prev, (message) => ({
                            ...message,
                            content: fullContent,
                        }))
                    );
                }
            } catch (error) {
                if (error instanceof Error && error.name === "AbortError") {
                    return;
                }

                setError("Failed to send message. Please try again.");
                setMessages((prev) =>
                    updateLastAssistantMessage(prev, (message) => ({
                        ...message,
                        content: `${message.content}\n\n[System Error: Failed to get response]`,
                    }))
                );
            } finally {
                setIsStreaming(false);
                abortControllerRef.current = null;
            }
        },
        [isStreaming, messages, settings]
    );

    const setMessagesList = useCallback((nextMessages: Message[]) => {
        setMessages(nextMessages);
    }, []);

    return {
        messages,
        isStreaming,
        error,
        sendMessage,
        setMessages: setMessagesList,
        activeCitation,
        setActiveCitation,
    };
}
