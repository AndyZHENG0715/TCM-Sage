"use client";

import { useCallback, useState } from "react";
import { ChatSession } from "@/lib/types";

const HISTORY_KEY = "tcm-sage-history";

function areSessionsEquivalent(a: ChatSession, b: ChatSession) {
    return (
        a.id === b.id &&
        a.title === b.title &&
        a.createdAt === b.createdAt &&
        JSON.stringify(a.messages) === JSON.stringify(b.messages)
    );
}

function generateUUID() {
    if (typeof crypto !== "undefined" && crypto.randomUUID) {
        return crypto.randomUUID();
    }
    // Fallback for non-secure contexts (HTTP)
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
        const r = (Math.random() * 16) | 0;
        const v = c === "x" ? r : (r & 0x3) | 0x8;
        return v.toString(16);
    });
}

export function useHistory() {
    const [sessions, setSessions] = useState<ChatSession[]>(() => {
        if (typeof window === "undefined") {
            return [];
        }

        const stored = localStorage.getItem(HISTORY_KEY);
        if (!stored) {
            return [];
        }

        try {
            const parsed = JSON.parse(stored) as ChatSession[];
            return parsed.sort((a, b) => b.updatedAt - a.updatedAt);
        } catch (error) {
            console.error("Failed to parse history", error);
            return [];
        }
    });
    const isLoaded = true;

    const saveSession = useCallback((session: ChatSession) => {
        setSessions((prev) => {
            const existingIndex = prev.findIndex((item) => item.id === session.id);
            const existingSession = existingIndex >= 0 ? prev[existingIndex] : null;

            if (existingSession && areSessionsEquivalent(existingSession, session)) {
                return prev;
            }

            const next = existingIndex >= 0 ? [...prev] : [session, ...prev];
            if (existingIndex >= 0) {
                next[existingIndex] = session;
            }

            next.sort((a, b) => b.updatedAt - a.updatedAt);
            localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
            return next;
        });
    }, []);

    const deleteSession = useCallback((id: string) => {
        setSessions((prev) => {
            const next = prev.filter((session) => session.id !== id);
            if (next.length === prev.length) {
                return prev;
            }

            localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
            return next;
        });
    }, []);

    const createSession = useCallback(
        (): ChatSession => ({
            id: generateUUID(),
            title: "New Research Chat",
            messages: [],
            createdAt: Date.now(),
            updatedAt: Date.now(),
        }),
        []
    );

    return {
        sessions,
        saveSession,
        deleteSession,
        createSession,
        isLoaded,
    };
}
