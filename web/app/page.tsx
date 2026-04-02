"use client";

import { useEffect, useMemo, useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { ChatArea } from "@/components/ChatArea";
import { CitationPanel } from "@/components/CitationPanel";
import { SettingsModal } from "@/components/SettingsModal";
import { useChat } from "@/hooks/useChat";
import { useHistory } from "@/hooks/useHistory";
import { useKeepAlive } from "@/hooks/useKeepAlive";
import { useSettings } from "@/hooks/useSettings";
import { ChatSession, Citation, Message } from "@/lib/types";

function getCitationPanelKey(citation: Citation | null) {
    if (!citation) {
        return "empty";
    }

    if (citation.type === "text") {
        return citation.chunk_id || `text-${citation.number}`;
    }

    return `graph-${citation.number}-${citation.fact}`;
}

function buildSessionTitle(messages: Message[], existingTitle?: string) {
    if (existingTitle && existingTitle !== "New Research Chat") {
        return existingTitle;
    }

    const firstUserMessage = messages.find((message) => message.role === "user");
    if (!firstUserMessage) {
        return existingTitle ?? "New Research Chat";
    }

    const trimmed = firstUserMessage.content.trim();
    if (!trimmed) {
        return existingTitle ?? "New Research Chat";
    }

    return `${trimmed.slice(0, 30)}${trimmed.length > 30 ? "..." : ""}`;
}

function areSessionsEquivalent(current: ChatSession | null, draft: ChatSession) {
    if (!current) {
        return false;
    }

    return (
        current.id === draft.id &&
        current.title === draft.title &&
        current.createdAt === draft.createdAt &&
        JSON.stringify(current.messages) === JSON.stringify(draft.messages)
    );
}

export default function Home() {
    const {
        defaultSettings,
        settings,
        capabilities,
        updateSettings,
        resetDefaults,
        isLoaded: isSettingsLoaded,
    } = useSettings();
    const {
        sessions,
        saveSession,
        deleteSession,
        createSession,
        isLoaded: isHistoryLoaded,
    } = useHistory();
    const {
        messages,
        isStreaming,
        sendMessage,
        setMessages,
        activeCitation,
        setActiveCitation,
    } = useChat(settings);

    useKeepAlive();

    const [currentSessionId, setCurrentSessionId] = useState<string | null>(() => createSession().id);
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);
    const currentSession = useMemo(
        () => sessions.find((session) => session.id === currentSessionId) ?? null,
        [currentSessionId, sessions]
    );

    useEffect(() => {
        if (!currentSessionId || messages.length === 0) {
            return;
        }

        const draftSession: ChatSession = {
            id: currentSessionId,
            title: buildSessionTitle(messages, currentSession?.title),
            messages,
            createdAt: currentSession?.createdAt ?? Date.now(),
            updatedAt: Date.now(),
        };

        if (areSessionsEquivalent(currentSession, draftSession)) {
            return;
        }

        saveSession(draftSession);
    }, [currentSession, currentSessionId, messages, saveSession]);

    const handleNewChat = () => {
        const newSession = createSession();
        setCurrentSessionId(newSession.id);
        setMessages([]);
        setActiveCitation(null);
    };

    const handleDeleteSession = (sessionId: string) => {
        if (currentSessionId === sessionId) {
            const replacementSession = createSession();
            setCurrentSessionId(replacementSession.id);
            setMessages([]);
            setActiveCitation(null);
        }

        deleteSession(sessionId);
    };

    const handleSelectSession = (session: ChatSession) => {
        setCurrentSessionId(session.id);
        setMessages(session.messages);
        setActiveCitation(null);
    };

    const currentTitle = useMemo(
        () => currentSession?.title ?? buildSessionTitle(messages),
        [currentSession, messages]
    );

    if (!isSettingsLoaded || !isHistoryLoaded) {
        return (
            <div className="flex items-center justify-center h-screen bg-background-dark text-parchment">
                <div className="animate-pulse flex flex-col items-center">
                    <div className="w-12 h-12 bg-primary/20 rounded-full mb-4" />
                    Loading TCM-Sage...
                </div>
            </div>
        );
    }

    return (
        <div className="flex h-screen overflow-hidden bg-background-dark text-parchment font-sans selection:bg-primary/30">
            <Sidebar
                sessions={sessions}
                currentSessionId={currentSessionId}
                onSelectSession={handleSelectSession}
                onNewChat={handleNewChat}
                onDeleteSession={handleDeleteSession}
                onOpenSettings={() => setIsSettingsOpen(true)}
                className="shrink-0 z-20"
            />

            <main className="flex-1 relative flex flex-col min-w-0 transition-all duration-300">
                <ChatArea
                    messages={messages}
                    isStreaming={isStreaming}
                    title={currentTitle}
                    onSend={sendMessage}
                    onCitationClick={setActiveCitation}
                />
            </main>

            <CitationPanel
                key={getCitationPanelKey(activeCitation)}
                citation={activeCitation}
                onClose={() => setActiveCitation(null)}
            />

            {isSettingsOpen && (
                <SettingsModal
                    isOpen={isSettingsOpen}
                    onClose={() => setIsSettingsOpen(false)}
                    settings={settings}
                    defaultSettings={defaultSettings}
                    capabilities={capabilities}
                    onSave={updateSettings}
                    onReset={resetDefaults}
                />
            )}
        </div>
    );
}
