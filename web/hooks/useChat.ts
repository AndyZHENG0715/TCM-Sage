"use client";

import { useState, useCallback, useRef } from "react";
import { Message, Citation } from "@/lib/types";
import { streamQuery } from "@/lib/api";

export function useChat() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [isStreaming, setIsStreaming] = useState(false);
    const [activeCitation, setActiveCitation] = useState<Citation | null>(null);

    // AbortController for cancelling streams
    const abortControllerRef = useRef<AbortController | null>(null);

    const sendMessage = useCallback(
        async (content: string) => {
            if (!content.trim() || isStreaming) return;

            const userMessage: Message = {
                role: "user",
                content: content.trim(),
                timestamp: Date.now(),
            };

            setMessages((prev) => [...prev, userMessage]);
            setIsStreaming(true);

            const assistantMessagePlaceholder: Message = {
                role: "assistant",
                content: "",
                timestamp: Date.now(),
            };

            setMessages((prev) => [...prev, assistantMessagePlaceholder]);

            try {
                abortControllerRef.current = new AbortController();
                // Note: fetch in streamQueryRobust doesn't use signal yet, but we can ignore result

                const stream = streamQuery(content);

                let fullContent = "";
                let citations: Citation[] = [];
                let severity: "informational" | "prescriptive" | undefined;
                let verification: { status: string; explanation: string } | undefined;

                for await (const event of stream) {
                    if (event.type === "text") {
                        fullContent += event.content;
                        setMessages((prev) => {
                            const newMessages = [...prev];
                            const lastMsg = newMessages[newMessages.length - 1];
                            if (lastMsg.role === "assistant") {
                                lastMsg.content = fullContent;
                            }
                            return newMessages;
                        });
                    } else if (event.type === "metadata") {
                        citations = event.citations;
                        severity = event.severity;
                        verification = event.verification as { status: string; explanation: string };

                        setMessages((prev) => {
                            const newMessages = [...prev];
                            const lastMsg = newMessages[newMessages.length - 1];
                            if (lastMsg.role === "assistant") {
                                lastMsg.citations = citations;
                                lastMsg.severity = severity;
                                lastMsg.verification = verification;
                            }
                            return newMessages;
                        });
                    } else if (event.type === "error") {
                        console.error("Stream error:", event.message);
                        // Append error to content or handle appropriately
                        fullContent += `\n\n[Error: ${event.message}]`;
                        setMessages((prev) => {
                            const newMessages = [...prev];
                            const lastMsg = newMessages[newMessages.length - 1];
                            if (lastMsg.role === "assistant") {
                                lastMsg.content = fullContent;
                            }
                            return newMessages;
                        });
                    }
                }
            } catch (error) {
                console.error("Chat error:", error);
                setMessages((prev) => {
                    const newMessages = [...prev];
                    const lastMsg = newMessages[newMessages.length - 1];
                    if (lastMsg.role === "assistant") {
                        lastMsg.content += "\n\n[System Error: Failed to get response]";
                    }
                    return newMessages;
                });
            } finally {
                setIsStreaming(false);
                abortControllerRef.current = null;
            }
        },
        [isStreaming]
    );

    const setMessagesList = useCallback((msgs: Message[]) => {
        setMessages(msgs);
    }, []);

    return {
        messages,
        isStreaming,
        sendMessage,
        setMessages: setMessagesList,
        activeCitation,
        setActiveCitation,
    };
}
