"use client";

import { useCallback, useRef, useState } from "react";
import { submitArenaVote, streamArenaQuery } from "@/lib/api";
import type { Citation } from "@/lib/types";

export type VoteOption = "a" | "b" | "tie";

export interface ArenaRoundVote {
    roundNumber: number;
    query: string;
    responseA: string;
    responseB: string;
    positionMapping: Record<string, string>; // {"a": "rag"} or {"a": "plain"}
    vote: VoteOption;
    comment?: string | null;
    citationsA?: Citation[];
    citationsB?: Citation[];
}

export interface ArenaState {
    // Per-round streaming state
    responseA: string;
    responseB: string;
    isStreamingA: boolean;
    isStreamingB: boolean;
    errorA: string | null;
    errorB: string | null;
    metadataA: Record<string, unknown> | null;
    metadataB: Record<string, unknown> | null;
    arenaConfig: { position_mapping: Record<string, string> } | null;

    // Session state
    sessionId: string;
    roundNumber: number;
    votes: ArenaRoundVote[];
    selectedModel: string;

    // UI state
    canVote: boolean; // true when both streams done
    hasVotedThisRound: boolean;
    showReveal: boolean;
}

export function useArena(initialSessionId: string, initialModel = "qwen-plus") {
    const [state, setState] = useState<ArenaState>({
        responseA: "",
        responseB: "",
        isStreamingA: false,
        isStreamingB: false,
        errorA: null,
        errorB: null,
        metadataA: null,
        metadataB: null,
        arenaConfig: null,
        sessionId: initialSessionId,
        roundNumber: 1,
        votes: [],
        selectedModel: initialModel,
        canVote: false,
        hasVotedThisRound: false,
        showReveal: false,
    });

    // Independent chat histories for A and B
    const chatHistoryARef = useRef<{ role: string; content: string }[]>([]);
    const chatHistoryBRef = useRef<{ role: string; content: string }[]>([]);
    const abortControllerRef = useRef<AbortController | null>(null);
    const currentQueryRef = useRef<string>("");

    const sendArenaQuery = useCallback(
        async (question: string) => {
            if (!question.trim()) return;

            // Abort any in-flight streams
            abortControllerRef.current?.abort();
            abortControllerRef.current = new AbortController();
            currentQueryRef.current = question;

            setState((prev) => ({
                ...prev,
                responseA: "",
                responseB: "",
                isStreamingA: true,
                isStreamingB: true,
                errorA: null,
                errorB: null,
                metadataA: null,
                metadataB: null,
                arenaConfig: null,
                canVote: false,
                hasVotedThisRound: false,
            }));

            try {
                const stream = streamArenaQuery(
                    question,
                    chatHistoryARef.current,
                    chatHistoryBRef.current,
                    state.selectedModel,
                    state.sessionId,
                    state.roundNumber,
                    abortControllerRef.current.signal
                );

                let collectedA = "";
                let collectedB = "";
                let doneA = false;
                let doneB = false;

                for await (const event of stream) {
                    if (event.type === "text_a") {
                        collectedA += event.content;
                        setState((prev) => ({ ...prev, responseA: collectedA }));
                    } else if (event.type === "text_b") {
                        collectedB += event.content;
                        setState((prev) => ({ ...prev, responseB: collectedB }));
                    } else if (event.type === "metadata_a") {
                        doneA = true;
                        setState((prev) => ({
                            ...prev,
                            metadataA: event.data,
                            isStreamingA: false,
                            canVote: doneA && doneB,
                        }));
                    } else if (event.type === "metadata_b") {
                        doneB = true;
                        setState((prev) => ({
                            ...prev,
                            metadataB: event.data,
                            isStreamingB: false,
                            canVote: doneA && doneB,
                        }));
                    } else if (event.type === "arena_config") {
                        setState((prev) => ({ ...prev, arenaConfig: event.data }));
                    } else if (event.type === "error") {
                        const panel = (event.data as { panel?: string }).panel;
                        if (panel === "a") {
                            doneA = true;
                            setState((prev) => ({
                                ...prev,
                                errorA: String((event.data as { message?: string }).message ?? "Error"),
                                isStreamingA: false,
                                canVote: doneA && doneB,
                            }));
                        } else {
                            doneB = true;
                            setState((prev) => ({
                                ...prev,
                                errorB: String((event.data as { message?: string }).message ?? "Error"),
                                isStreamingB: false,
                                canVote: doneA && doneB,
                            }));
                        }
                    }
                }

                // Ensure streaming flags cleared at end
                setState((prev) => ({
                    ...prev,
                    isStreamingA: false,
                    isStreamingB: false,
                    canVote: true,
                }));
            } catch (err) {
                if ((err as Error).name !== "AbortError") {
                    setState((prev) => ({
                        ...prev,
                        errorA: "Stream failed",
                        errorB: "Stream failed",
                        isStreamingA: false,
                        isStreamingB: false,
                    }));
                }
            }
        },
        [state.selectedModel, state.sessionId, state.roundNumber]
    );

    const submitVote = useCallback(
        async (vote: VoteOption, comment?: string) => {
            const currentState = state; // capture snapshot

            // Build vote record
            const roundVote: ArenaRoundVote = {
                roundNumber: currentState.roundNumber,
                query: currentQueryRef.current,
                responseA: currentState.responseA,
                responseB: currentState.responseB,
                positionMapping: currentState.arenaConfig?.position_mapping ?? {},
                vote,
                comment,
                citationsA: (currentState.metadataA?.citations as Citation[]) ?? [],
                citationsB: (currentState.metadataB?.citations as Citation[]) ?? [],
            };

            setState((prev) => ({
                ...prev,
                votes: [...prev.votes, roundVote],
                roundNumber: prev.roundNumber + 1,
                hasVotedThisRound: true,
            }));

            // Fire-and-forget to backend
            await submitArenaVote({
                session_id: currentState.sessionId,
                round_number: roundVote.roundNumber,
                query: roundVote.query,
                response_a: roundVote.responseA,
                response_b: roundVote.responseB,
                model_name: currentState.selectedModel,
                position_mapping: roundVote.positionMapping,
                vote,
                comment: comment ?? null,
            });

            // Append to independent histories
            chatHistoryARef.current = [
                ...chatHistoryARef.current,
                { role: "user", content: currentQueryRef.current },
                { role: "assistant", content: currentState.responseA },
            ];
            chatHistoryBRef.current = [
                ...chatHistoryBRef.current,
                { role: "user", content: currentQueryRef.current },
                { role: "assistant", content: currentState.responseB },
            ];
        },
        [state]
    );

    const setSelectedModel = useCallback((model: string) => {
        setState((prev) => ({ ...prev, selectedModel: model }));
    }, []);

    const revealAll = useCallback(() => {
        setState((prev) => ({ ...prev, showReveal: true }));
    }, []);

    const resetSession = useCallback(() => {
        abortControllerRef.current?.abort();
        chatHistoryARef.current = [];
        chatHistoryBRef.current = [];
        currentQueryRef.current = "";
        setState({
            responseA: "",
            responseB: "",
            isStreamingA: false,
            isStreamingB: false,
            errorA: null,
            errorB: null,
            metadataA: null,
            metadataB: null,
            arenaConfig: null,
            sessionId: crypto.randomUUID(),
            roundNumber: 1,
            votes: [],
            selectedModel: state.selectedModel,
            canVote: false,
            hasVotedThisRound: false,
            showReveal: false,
        });
    }, [state.selectedModel]);

    return {
        state,
        sendArenaQuery,
        submitVote,
        setSelectedModel,
        revealAll,
        resetSession,
    };
}
