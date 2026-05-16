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
    positionMapping: Record<string, string>;
    vote: VoteOption;
    comment?: string | null;
    citationsA?: Citation[];
    citationsB?: Citation[];
}

export interface ArenaState {
    responseA: string;
    responseB: string;
    isStreamingA: boolean;
    isStreamingB: boolean;
    errorA: string | null;
    errorB: string | null;
    voteError: string | null;
    metadataA: Record<string, unknown> | null;
    metadataB: Record<string, unknown> | null;
    arenaConfig: { position_mapping: Record<string, string> } | null;
    sessionId: string;
    roundNumber: number;
    votes: ArenaRoundVote[];
    selectedModel: string;
    canVote: boolean;
    hasVotedThisRound: boolean;
    showReveal: boolean;
}

export function useArena(initialSessionId: string, initialModel = "qwen-flash") {
    const [state, setState] = useState<ArenaState>({
        responseA: "",
        responseB: "",
        isStreamingA: false,
        isStreamingB: false,
        errorA: null,
        errorB: null,
        voteError: null,
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

    const chatHistoryARef = useRef<{ role: string; content: string }[]>([]);
    const chatHistoryBRef = useRef<{ role: string; content: string }[]>([]);
    const abortControllerRef = useRef<AbortController | null>(null);
    const currentQueryRef = useRef<string>("");

    const sendArenaQuery = useCallback(
        async (question: string) => {
            if (!question.trim()) return;

            abortControllerRef.current?.abort();
            abortControllerRef.current = new AbortController();
            currentQueryRef.current = question;

            const selectedModel = state.selectedModel;
            const sessionId = state.sessionId;
            const roundNumber = state.roundNumber;

            setState((prev) => ({
                ...prev,
                responseA: "",
                responseB: "",
                isStreamingA: true,
                isStreamingB: true,
                errorA: null,
                errorB: null,
                voteError: null,
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
                    selectedModel,
                    sessionId,
                    roundNumber,
                    abortControllerRef.current.signal
                );

                let collectedA = "";
                let collectedB = "";
                let doneA = false;
                let doneB = false;
                let bothStarted = false;

                for await (const event of stream) {
                    if (event.type === "text_a") {
                        collectedA += event.content;
                        if (!bothStarted && collectedB) bothStarted = true;
                        if (bothStarted) {
                            setState((prev) => ({ ...prev, responseA: collectedA, responseB: collectedB }));
                        }
                    } else if (event.type === "text_b") {
                        collectedB += event.content;
                        if (!bothStarted && collectedA) bothStarted = true;
                        if (bothStarted) {
                            setState((prev) => ({ ...prev, responseA: collectedA, responseB: collectedB }));
                        }
                    } else if (event.type === "metadata_a") {
                        doneA = true;
                        setState((prev) => ({
                            ...prev,
                            metadataA: event.data,
                            isStreamingA: false,
                            responseA: collectedA,
                            canVote: doneA && doneB,
                        }));
                    } else if (event.type === "metadata_b") {
                        doneB = true;
                        setState((prev) => ({
                            ...prev,
                            metadataB: event.data,
                            isStreamingB: false,
                            responseB: collectedB,
                            canVote: doneA && doneB,
                        }));
                    } else if (event.type === "arena_config") {
                        setState((prev) => ({ ...prev, arenaConfig: event.data }));
                    } else if (event.type === "error") {
                        const panel = (event.data as { panel?: string }).panel;
                        const message = String((event.data as { message?: string }).message ?? "Error");
                        if (panel === "a") {
                            doneA = true;
                            setState((prev) => ({ ...prev, errorA: message, isStreamingA: false, canVote: doneB }));
                        } else {
                            doneB = true;
                            setState((prev) => ({ ...prev, errorB: message, isStreamingB: false, canVote: doneA }));
                        }
                    }
                }

                setState((prev) => ({
                    ...prev,
                    responseA: collectedA || prev.responseA,
                    responseB: collectedB || prev.responseB,
                    isStreamingA: false,
                    isStreamingB: false,
                    canVote: (collectedA.length > 0 || prev.responseA.length > 0) &&
                        (collectedB.length > 0 || prev.responseB.length > 0),
                }));
            } catch (err) {
                if ((err as Error).name !== "AbortError") {
                    setState((prev) => ({
                        ...prev,
                        errorA: "Stream failed",
                        errorB: "Stream failed",
                        isStreamingA: false,
                        isStreamingB: false,
                        canVote: false,
                    }));
                }
            }
        },
        [state.roundNumber, state.selectedModel, state.sessionId]
    );

    const submitVote = useCallback(
        async (vote: VoteOption, comment?: string) => {
            const currentState = state;

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
                voteError: null,
            }));

            try {
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
            } catch {
                setState((prev) => ({
                    ...prev,
                    votes: prev.votes.slice(0, -1),
                    roundNumber: currentState.roundNumber,
                    canVote: currentState.canVote,
                    hasVotedThisRound: false,
                    voteError: "Unable to save your vote. Please try again.",
                }));
            }
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
            voteError: null,
            metadataA: null,
            metadataB: null,
            arenaConfig: null,
            sessionId:
                typeof crypto !== "undefined" && crypto.randomUUID
                    ? crypto.randomUUID()
                    : Math.random().toString(36).slice(2) + Date.now().toString(36),
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
