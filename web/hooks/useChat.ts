"use client";

import { useCallback, useRef, useState } from "react";
import { Citation, CitationBounds, Message, Settings, Verification } from "@/lib/types";
import { streamQuery } from "@/lib/api";
import { getLiveAccessCode, PRESENTATION_MODE } from "@/lib/showcase";

// The public release backend fixes its settings, accepts at most six history
// turns and needs the project access code.
const PUBLIC_HISTORY_LIMIT = 6;

function friendlyError(error: unknown, chinese: boolean) {
    const status = error instanceof Error ? /API Error: (\d+)/.exec(error.message)?.[1] : undefined;
    if (status === "401" || status === "403") {
        return chinese ? "访问码无效。请检查后重试。" : "That access code was not accepted. Check it and try again.";
    }
    if (status === "429") {
        return chinese ? "今天的免费额度已用完或有人正在提问，请稍后再试。" : "The free quota is used up or another question is running. Please try again later.";
    }
    return chinese ? "实时服务暂不可用，请稍后再试。" : "The live service is unavailable. Please try again later.";
}

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

export function useChat(settings: Settings, chinese = false) {
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

            if (!PRESENTATION_MODE && !getLiveAccessCode()) {
                setError(chinese ? "请先在顶部输入项目访问码，再提问。" : "Enter the project access code at the top before asking.");
                return;
            }

            const fullHistory = messages.map((message) => ({
                role: message.role,
                content: message.content,
            }));
            const chatHistory = PRESENTATION_MODE ? fullHistory : fullHistory.slice(-PUBLIC_HISTORY_LIMIT);

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

                const stream = streamQuery(content, chatHistory, PRESENTATION_MODE ? settings : undefined);
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

                const failure = PRESENTATION_MODE ? "Failed to send message. Please try again." : friendlyError(error, chinese);
                setError(failure);
                setMessages((prev) =>
                    updateLastAssistantMessage(prev, (message) => ({
                        ...message,
                        content: PRESENTATION_MODE
                            ? `${message.content}\n\n[System Error: Failed to get response]`
                            : `${message.content}\n\n[${failure}]`.trim(),
                    }))
                );
            } finally {
                setIsStreaming(false);
                abortControllerRef.current = null;
            }
        },
        [chinese, isStreaming, messages, settings]
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
