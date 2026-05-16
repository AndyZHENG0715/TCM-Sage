"use client";

import Link from "next/link";
import { useState } from "react";
import {
    BarChart3,
    ExternalLink,
    MessageSquare,
    MessageSquarePlus,
    PanelLeftClose,
    PanelLeftOpen,
    Settings,
    Swords,
    Trash2,
    User,
} from "lucide-react";
import { useI18n } from "@/i18n/context";
import { ChatSession } from "@/lib/types";
import { cn } from "@/lib/utils";

interface SidebarProps {
    sessions: ChatSession[];
    currentSessionId: string | null;
    onSelectSession: (session: ChatSession) => void;
    onNewChat: () => void;
    onDeleteSession: (sessionId: string) => void;
    onOpenSettings: () => void;
    className?: string;
}

type SessionGroupKey = "today" | "yesterday" | "lastWeek" | "older";

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
    const { locale, t, toggleLocale } = useI18n();
    const localeToggle =
        locale === "zh"
            ? { label: "繁", ariaLabel: "切换到繁體中文" }
            : locale === "zh-Hant"
              ? { label: "EN", ariaLabel: "切換到英文" }
              : { label: "简", ariaLabel: "Switch to Simplified Chinese" };

    const handleDelete = (e: React.MouseEvent<HTMLButtonElement>, sessionId: string) => {
        e.preventDefault();
        e.stopPropagation();
        if (window.confirm(`${t.sidebar.deleteChat}?`)) {
            onDeleteSession(sessionId);
        }
    };

    const groupedSessions = sessions.reduce((acc, session) => {
        const date = new Date(session.updatedAt);
        const today = new Date();
        const yesterday = new Date(today);
        yesterday.setDate(today.getDate() - 1);

        const isToday = date.toDateString() === today.toDateString();
        const isYesterday = yesterday.toDateString() === date.toDateString();

        let group: SessionGroupKey = "older";
        if (isToday) group = "today";
        else if (isYesterday) group = "yesterday";
        else if (today.getTime() - date.getTime() < 7 * 24 * 60 * 60 * 1000) group = "lastWeek";

        if (!acc[group]) acc[group] = [];
        acc[group].push(session);
        return acc;
    }, {} as Record<SessionGroupKey, ChatSession[]>);

    const groupOrder: SessionGroupKey[] = ["today", "yesterday", "lastWeek", "older"];
    const groupLabels: Record<SessionGroupKey, string> = {
        today: t.sidebar.today,
        yesterday: t.sidebar.yesterday,
        lastWeek: t.sidebar.lastWeek,
        older: t.sidebar.older,
    };

    return (
        <div
            className={cn(
                "relative flex h-full flex-col border-r border-white/5 bg-sidebar-dark transition-all duration-300",
                collapsed ? "w-16" : "w-64 md:w-72",
                className
            )}
        >
            <div className="flex items-center justify-between p-4">
                {!collapsed && (
                    <div className="flex items-center gap-2">
                        <div className="flex h-8 w-8 items-center justify-center rounded bg-gradient-to-br from-primary to-primary-dark/50 font-bold text-background-dark">
                            S
                        </div>
                        <h1 className="font-serif text-lg font-bold tracking-wide text-parchment">{t.common.appName}</h1>
                    </div>
                )}
                <button
                    type="button"
                    onClick={() => setCollapsed(!collapsed)}
                    className="min-h-11 min-w-11 rounded-lg p-2.5 text-gray-400 transition-colors hover:bg-white/5 hover:text-parchment"
                    aria-label={collapsed ? t.common.confirm : t.common.close}
                >
                    {collapsed ? <PanelLeftOpen size={20} /> : <PanelLeftClose size={20} />}
                </button>
            </div>

            <div className="mb-1 px-3">
                <button
                    type="button"
                    onClick={onNewChat}
                    className={cn(
                        "flex min-h-11 w-full items-center gap-3 rounded-lg border border-primary/20 p-3 transition-all hover:bg-white/5 group",
                        collapsed && "justify-center"
                    )}
                >
                    <MessageSquarePlus
                        size={20}
                        className="text-primary transition-all group-hover:drop-shadow-[0_0_8px_rgba(25,230,212,0.5)]"
                    />
                    {!collapsed && <span className="font-sans text-sm font-medium text-parchment">{t.sidebar.newChat}</span>}
                </button>
            </div>

            <div className="mb-2 space-y-1 px-3">
                <Link
                    href="/arena"
                    className={cn(
                        "flex min-h-11 w-full items-center gap-3 rounded-lg border border-primary/20 p-3 transition-all hover:bg-white/5 group",
                        collapsed && "justify-center"
                    )}
                >
                    <Swords size={20} className="text-primary transition-all group-hover:drop-shadow-[0_0_8px_rgba(25,230,212,0.5)]" />
                    {!collapsed && <span className="font-sans text-sm font-medium text-parchment">{t.sidebar.arena}</span>}
                </Link>
                <Link
                    href="/arena/stats"
                    className={cn(
                        "flex min-h-11 w-full items-center gap-3 rounded-lg border border-primary/20 p-3 transition-all hover:bg-white/5 group",
                        collapsed && "justify-center"
                    )}
                >
                    <BarChart3 size={20} className="text-primary transition-all group-hover:drop-shadow-[0_0_8px_rgba(25,230,212,0.5)]" />
                    {!collapsed && <span className="font-sans text-sm font-medium text-parchment">{t.sidebar.arenaStats}</span>}
                </Link>
                <a
                    href="https://forms.gle/Sm62ucNSKQzGGPJ76"
                    target="_blank"
                    rel="noopener noreferrer"
                    className={cn(
                        "flex min-h-11 w-full items-center gap-3 rounded-lg border border-primary/20 p-3 transition-all hover:bg-white/5 group",
                        collapsed && "justify-center"
                    )}
                >
                    <ExternalLink size={20} className="text-primary transition-all group-hover:drop-shadow-[0_0_8px_rgba(25,230,212,0.5)]" />
                    {!collapsed && <span className="font-sans text-sm font-medium text-parchment">{t.sidebar.feedback}</span>}
                </a>
            </div>

            <div className="flex-1 space-y-6 overflow-y-auto px-3 py-2 scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent">
                {collapsed ? (
                    sessions.slice(0, 5).map((session) => (
                        <button
                            key={session.id}
                            type="button"
                            onClick={() => onSelectSession(session)}
                            className={cn(
                                "relative flex min-h-11 w-full justify-center rounded-lg p-2 transition-colors hover:bg-white/5 group",
                                currentSessionId === session.id && "bg-white/10 text-primary"
                            )}
                            title={session.title || t.sidebar.newChat}
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
                                <h3 className="mb-2 px-2 text-xs font-bold uppercase tracking-wider text-gray-500">
                                    {groupLabels[group]}
                                </h3>
                                <div className="space-y-1">
                                    {groupSessions.map((session) => (
                                        <div
                                            key={session.id}
                                            className={cn(
                                                "group flex items-center gap-2 rounded-lg transition-colors",
                                                currentSessionId === session.id
                                                    ? "bg-white/10 text-parchment"
                                                    : "text-gray-400 hover:bg-white/5 hover:text-parchment"
                                            )}
                                        >
                                            <button
                                                type="button"
                                                onClick={() => onSelectSession(session)}
                                                className="flex min-h-11 min-w-0 flex-1 items-center gap-3 overflow-hidden p-2 text-left"
                                            >
                                                <span className="truncate text-sm font-medium">
                                                    {session.title || t.sidebar.newChat}
                                                </span>
                                            </button>

                                            <button
                                                type="button"
                                                onPointerDown={(e) => e.stopPropagation()}
                                                onClick={(e) => handleDelete(e, session.id)}
                                                className="min-h-11 min-w-11 shrink-0 rounded p-2 opacity-0 transition group-hover:opacity-100 hover:text-red-400"
                                                title={t.sidebar.deleteChat}
                                                aria-label={`${t.sidebar.deleteChat} ${session.title || t.sidebar.newChat}`}
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

            <div className="flex flex-col gap-2 border-t border-white/5 p-4">
                <button
                    type="button"
                    onClick={toggleLocale}
                    className={cn(
                        "flex min-h-11 items-center rounded-full border border-primary/20 bg-primary/10 text-primary transition-colors hover:bg-primary/15",
                        collapsed ? "justify-center px-0" : "justify-between px-3 py-2"
                    )}
                    aria-label={localeToggle.ariaLabel}
                >
                    {!collapsed && <span className="text-xs uppercase tracking-[0.2em] text-parchment/70">简→繁→EN→简</span>}
                    <span className="rounded-full bg-primary/10 px-3 py-1 text-xs font-semibold text-primary">
                        {localeToggle.label}
                    </span>
                </button>

                <div
                    className={cn(
                        "flex min-h-11 w-full items-center gap-3 rounded-lg p-2 text-left transition-colors hover:bg-white/5",
                        collapsed && "justify-center"
                    )}
                >
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-parchment text-background-dark font-bold">
                        <User size={16} />
                    </div>
                    {!collapsed && (
                        <div className="flex-1 overflow-hidden">
                            <p className="truncate text-sm font-medium text-parchment">{t.sidebar.userName}</p>
                            <p className="truncate text-xs text-gray-500">{t.sidebar.userPlan}</p>
                        </div>
                    )}
                    {!collapsed && (
                        <button
                            type="button"
                            onClick={(e) => {
                                e.stopPropagation();
                                onOpenSettings();
                            }}
                            className="group flex min-h-11 min-w-11 items-center justify-center p-2 text-gray-400 transition-colors hover:text-parchment"
                            aria-label={t.settings.title}
                        >
                            <span className="inline-flex origin-center transition-transform duration-200 ease-out group-hover:rotate-90">
                                <Settings size={18} />
                            </span>
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
}
