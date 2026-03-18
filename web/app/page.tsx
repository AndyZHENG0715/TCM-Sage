"use client";

import { useState, useEffect } from "react";
import { Sidebar } from "@/components/Sidebar";
import { ChatArea } from "@/components/ChatArea";
import { CitationPanel } from "@/components/CitationPanel";
import { SettingsModal } from "@/components/SettingsModal";
import { useChat } from "@/hooks/useChat";
import { useHistory } from "@/hooks/useHistory";
import { useSettings } from "@/hooks/useSettings";
import { useKeepAlive } from "@/hooks/useKeepAlive";
import { ChatSession } from "@/lib/types";

export default function Home() {
  const { settings, updateSettings, resetDefaults, isLoaded: isSettingsLoaded } = useSettings();
  const {
    sessions,
    saveSession,
    deleteSession,
    createSession,
    getSession,
    isLoaded: isHistoryLoaded
  } = useHistory();

  const {
    messages,
    isStreaming,
    sendMessage,
    setMessages,
    activeCitation,
    setActiveCitation
  } = useChat();

  useKeepAlive();

  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  // Initialize with a new chat if no history or on first load
  useEffect(() => {
    if (isHistoryLoaded && !currentSessionId) {
      // Optionally load the most recent session or start new
      // For now, let's start fresh or just leave it empty until user selects/creates
      // Actually, better UX is to start a new chat immediately if nothing selected
      if (sessions.length > 0) {
        // Maybe select most recent?
        // setCurrentSessionId(sessions[0].id);
        // setMessages(sessions[0].messages);

        // Or just start new:
        const newSession = createSession();
        // eslint-disable-next-line
        setCurrentSessionId(newSession.id);
        setMessages([]);
      } else {
        const newSession = createSession();
        // eslint-disable-next-line
        setCurrentSessionId(newSession.id);
        setMessages([]);
      }
    }
  }, [isHistoryLoaded]); // Run once when history loads

  // Auto-save session when messages change
  useEffect(() => {
    if (currentSessionId && messages.length > 0) {
      const session = getSession(currentSessionId);
      if (session) {
        saveSession({
          ...session,
          messages,
          updatedAt: Date.now(),
          // Generate title from first user message if it's "New Research Chat"
          title: session.title === "New Research Chat" && messages[0].role === "user"
            ? messages[0].content.slice(0, 30) + (messages[0].content.length > 30 ? "..." : "")
            : session.title
        });
      } else {
        // Session might not exist in history array yet if it was just created in memory
        // createSession returns a session object but doesn't add to state until saved?
        // No, createSession in useHistory just returns object.
        // So we need to reconstruct it if we don't find it?
        // Actually, we should use the current session ID to create a new entry if needed.
        saveSession({
          id: currentSessionId,
          title: messages[0].role === "user"
            ? messages[0].content.slice(0, 30)
            : "New Research Chat",
          messages,
          createdAt: Date.now(),
          updatedAt: Date.now()
        });
      }
    }
  }, [messages, currentSessionId]);

  const handleNewChat = () => {
    const newSession = createSession();
    setCurrentSessionId(newSession.id);
    setMessages([]);
  };

  const handleDeleteSession = (sessionId: string) => {
    deleteSession(sessionId);
    if (currentSessionId === sessionId) {
      handleNewChat();
    }
  };

  const handleSelectSession = (session: ChatSession) => {
    setCurrentSessionId(session.id);
    setMessages(session.messages);
  };

  if (!isSettingsLoaded || !isHistoryLoaded) {
    return (
      <div className="flex items-center justify-center h-screen bg-background-dark text-parchment">
        <div className="animate-pulse flex flex-col items-center">
          <div className="w-12 h-12 bg-primary/20 rounded-full mb-4"></div>
          Loading TCM-Sage...
        </div>
      </div>
    );
  }

  const currentTitle = currentSessionId ? (getSession(currentSessionId)?.title ?? null) : null;

  return (
    <div className="flex h-screen overflow-hidden bg-background-dark text-parchment font-sans selection:bg-primary/30">
      {/* Sidebar */}
      <Sidebar
        currentSessionId={currentSessionId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
        onDeleteSession={handleDeleteSession}
        onOpenSettings={() => setIsSettingsOpen(true)}
        className="shrink-0 z-20"
      />

      {/* Main Chat Area */}
      <main className="flex-1 relative flex flex-col min-w-0 transition-all duration-300">
        <ChatArea
          messages={messages}
          isStreaming={isStreaming}
          title={currentTitle}
          onSend={sendMessage}
          onCitationClick={setActiveCitation}
        />
      </main>

      {/* Citation Panel (Overlay on Desktop too for now to match design) */}
      {/* Or separate column if space permits? Design seems to be overlay/slide-over */}
      {activeCitation && (
        <CitationPanel
          citation={activeCitation}
          onClose={() => setActiveCitation(null)}
        />
      )}

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onSave={updateSettings}
        onReset={resetDefaults}
      />
    </div>
  );
}
