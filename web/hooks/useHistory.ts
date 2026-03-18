"use client";

import { useState, useEffect } from "react";
import { ChatSession } from "@/lib/types";

const HISTORY_KEY = "tcm-sage-history";

export function useHistory() {
    const [sessions, setSessions] = useState<ChatSession[]>([]);
    const [isLoaded, setIsLoaded] = useState(false);

    useEffect(() => {
        const stored = localStorage.getItem(HISTORY_KEY);
        if (stored) {
            try {
                const parsed = JSON.parse(stored);
                // Sort by updatedAt desc
                // eslint-disable-next-line
                setSessions(
                    parsed.sort(
                        (a: ChatSession, b: ChatSession) => b.updatedAt - a.updatedAt
                    )
                );
            } catch (e) {
                console.error("Failed to parse history", e);
            }
        }
        setIsLoaded(true);
    }, []);

    const saveSession = (session: ChatSession) => {
        setSessions((prev) => {
            const existingIndex = prev.findIndex((s) => s.id === session.id);
            let newSessions;
            if (existingIndex >= 0) {
                newSessions = [...prev];
                newSessions[existingIndex] = session;
            } else {
                newSessions = [session, ...prev];
            }
            // Sort again
            newSessions.sort((a, b) => b.updatedAt - a.updatedAt);
            localStorage.setItem(HISTORY_KEY, JSON.stringify(newSessions));
            return newSessions;
        });
    };

    const deleteSession = (id: string) => {
        setSessions((prev) => {
            const newSessions = prev.filter((s) => s.id !== id);
            localStorage.setItem(HISTORY_KEY, JSON.stringify(newSessions));
            return newSessions;
        });
    };

    const getSession = (id: string) => {
        return sessions.find((s) => s.id === id);
    };

    const createSession = (): ChatSession => {
        const newSession: ChatSession = {
            id: crypto.randomUUID(),
            title: "New Research Chat",
            messages: [],
            createdAt: Date.now(),
            updatedAt: Date.now(),
        };
        // We don't save it immediately to avoid empty chats clunking up history
        // It will be saved when first message is sent
        return newSession;
    };

    return {
        sessions,
        saveSession,
        deleteSession,
        getSession,
        createSession,
        isLoaded,
    };
}
