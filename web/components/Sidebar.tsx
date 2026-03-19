"use client";

import { ChatSession } from "@/lib/types";
import { cn } from "@/lib/utils";
import {
    MessageSquarePlus,
    Settings,
    User,
    PanelLeftClose,
    PanelLeftOpen,
    MessageSquare,
    Trash2,
} from "lucide-react";
import { useState } from "react";

interface SidebarProps {
    sessions: ChatSession[];
    currentSessionId: string | null;
    onSelectSession: (session: ChatSession) => void;
    onNewChat: () => void;
    onDeleteSession: (sessionId: string) => void;
    onOpenSettings: () => void;
    className?: string;
}

export function Sidebar({
    sessions,
    currentSessionId,
    onSelectSession,
    onNewChat,
    onDeleteSession,
    onOpenSettings,
    className,
}: SidebarProps) {
    const [collapsed, setCollapsed] = useState(false);

    const handleDelete = (e: React.MouseEvent, sessionId: string) => {
        e.stopPropagation();
        if (window.confirm("Delete this conversation? This cannot be undone.")) {
            onDeleteSession(sessionId);
        }
    };

    // Group sessions
    const groupedSessions = sessions.reduce((acc, session) => {
        const date = new Date(session.updatedAt);
        const now = new Date();
        const isToday = date.toDateString() === now.toDateString();
        const isYesterday =
            new Date(now.setDate(now.getDate() - 1)).toDateString() ===
            date.toDateString();

        let group = "Older";
        if (isToday) group = "Today";
        else if (isYesterday) group = "Yesterday";
        else if (now.getTime() - date.getTime() < 7 * 24 * 60 * 60 * 1000)
            group = "Last Week";

        if (!acc[group]) acc[group] = [];
        acc[group].push(session);
        return acc;
    }, {} as Record<string, ChatSession[]>);

    const groupOrder = ["Today", "Yesterday", "Last Week", "Older"];

    return (
        <div
            className={cn(
                "flex flex-col h-full bg-sidebar-dark border-r border-white/5 transition-all duration-300 relative",
                collapsed ? "w-16" : "w-64 md:w-72",
                className
            )}
        >
            {/* Header */}
            <div className="p-4 flex items-center justify-between">
                {!collapsed && (
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded bg-gradient-to-br from-primary to-primary-dark/50 flex items-center justify-center font-bold text-background-dark">
                            S
                        </div>
                        <h1 className="font-serif font-bold text-lg text-parchment tracking-wide">
                            TCM-Sage
                        </h1>
                    </div>
                )}
                <button
                    onClick={() => setCollapsed(!collapsed)}
                    className="p-2 text-gray-400 hover:text-parchment transition-colors rounded-lg hover:bg-white/5"
                >
                    {collapsed ? <PanelLeftOpen size={20} /> : <PanelLeftClose size={20} />}
                </button>
            </div>

            {/* New Chat Button */}
            <div className="px-3 mb-4">
                <button
                    onClick={onNewChat}
                    className={cn(
                        "flex items-center gap-3 w-full p-3 rounded-lg border border-primary/20 hover:bg-white/5 transition-all group",
                        collapsed ? "justify-center" : ""
                    )}
                >
                    <MessageSquarePlus
                        size={20}
                        className="text-primary group-hover:drop-shadow-[0_0_8px_rgba(25,230,212,0.5)] transition-all"
                    />
                    {!collapsed && (
                        <span className="font-sans font-medium text-parchment text-sm">
                            New Research Chat
                        </span>
                    )}
                </button>
            </div>

            {/* Chat History List */}
            <div className="flex-1 overflow-y-auto px-3 py-2 space-y-6 scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent">
                {collapsed ? (
                    // Collapsed view: just icons or simple list?
                    // Just show recent icons
                    sessions.slice(0, 5).map(session => (
                        <button
                            key={session.id}
                            onClick={() => onSelectSession(session)}
                            className={cn(
                                "w-full p-2 flex justify-center rounded-lg hover:bg-white/5 transition-colors relative group",
                                currentSessionId === session.id && "bg-white/10 text-primary"
                            )}
                            title={session.title}
                        >
                            <MessageSquare size={18} />
                        </button>
                    ))
                ) : (
                    groupOrder.map((group) => {
                        const groupSessions = groupedSessions[group];
                        if (!groupSessions || groupSessions.length === 0) return null;

                        return (
                            <div key={group}>
                                <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2 px-2">
                                    {group}
                                </h3>
                                <div className="space-y-1">
                                    {groupSessions.map((session) => (
                                        <div
                                            key={session.id}
                                            className={cn(
                                                "group flex items-center justify-between p-2 rounded-lg cursor-pointer transition-colors",
                                                currentSessionId === session.id
                                                    ? "bg-white/10 text-parchment"
                                                    : "text-gray-400 hover:bg-white/5 hover:text-parchment"
                                            )}
                                            onClick={() => onSelectSession(session)}
                                        >
                                            <div className="flex items-center gap-3 overflow-hidden">
                                                {/* <MessageSquare size={16} className="shrink-0" /> */}
                                                <span className="text-sm truncate font-medium">
                                                    {session.title || "New Chat"}
                                                </span>
                                            </div>

                                            {/* Delete button (visible on hover) */}
                                            <button
                                                onClick={(e) => handleDelete(e, session.id)}
                                                className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-400 transition-opacity"
                                                title="Delete chat"
                                            >
                                                <Trash2 size={14} />
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        );
                    })
                )}
            </div>

            {/* Footer / User Profile */}
            <div className="p-4 border-t border-white/5">
                <button className={cn(
                    "flex items-center gap-3 w-full p-2 rounded-lg hover:bg-white/5 transition-colors text-left",
                    collapsed ? "justify-center" : ""
                )}>
                    <div className="w-8 h-8 rounded-full bg-parchment text-background-dark flex items-center justify-center font-bold shrink-0">
                        <User size={16} />
                    </div>
                    {!collapsed && (
                        <div className="flex-1 overflow-hidden">
                            <p className="text-sm font-medium text-parchment truncate">Dr. Zhang</p>
                            <p className="text-xs text-gray-500 truncate">Pro Plan</p>
                        </div>
                    )}
                    {!collapsed && (
                        <div
                            onClick={(e) => {
                                e.stopPropagation(); // prevent profile click
                                onOpenSettings();
                            }}
                            className="p-2 text-gray-400 hover:text-parchment hover:rotate-90 transition-all cursor-pointer"
                        >
                            <Settings size={18} />
                        </div>
                    )}
                </button>
            </div>
        </div>
    );
}
