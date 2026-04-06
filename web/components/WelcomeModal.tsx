"use client";

import { useState, useSyncExternalStore, useCallback } from "react";
import Link from "next/link";
import { Swords, MessageSquareText, X } from "lucide-react";
import { useI18n } from "@/i18n/context";

function useHasSeenWelcome() {
    return useSyncExternalStore(
        () => () => {},
        () => localStorage.getItem("tcm-sage-welcome-seen") === "1",
        () => true, // SSR: assume seen to avoid flash
    );
}

export function WelcomeModal() {
    const hasSeen = useHasSeenWelcome();
    const [dismissed, setDismissed] = useState(false);
    const { t } = useI18n();

    const dismiss = useCallback(() => {
        setDismissed(true);
        localStorage.setItem("tcm-sage-welcome-seen", "1");
    }, []);

    if (hasSeen || dismissed) return null;

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 backdrop-blur-sm">
            <div className="bg-sidebar-dark border border-gray-700 rounded-2xl p-8 max-w-lg mx-4 shadow-2xl relative">
                <button
                    onClick={dismiss}
                    className="absolute top-4 right-4 p-1 text-gray-500 hover:text-parchment transition-colors"
                    aria-label={t.common.close}
                >
                    <X size={18} />
                </button>

                <h1 className="text-2xl font-serif font-bold text-parchment mb-2">
                    {t.welcome.title}
                </h1>
                <p className="text-sm text-gray-400 mb-6">
                    {t.welcome.description}
                </p>

                <div className="space-y-3">
                    <Link
                        href="/arena"
                        onClick={dismiss}
                        className="flex items-start gap-4 p-4 rounded-xl border border-primary/30 bg-primary/5 hover:bg-primary/10 transition-colors group"
                    >
                        <div className="p-2 rounded-lg bg-primary/10 text-primary group-hover:bg-primary/20 transition-colors mt-0.5">
                            <Swords size={22} />
                        </div>
                        <div>
                            <h3 className="font-semibold text-parchment text-base">
                                {t.welcome.arenaTitle}
                                <span className="ml-2 text-xs font-normal px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300">
                                    {t.welcome.arenaHelp}
                                </span>
                            </h3>
                            <p className="text-sm text-gray-400 mt-1 leading-relaxed">
                                {t.welcome.arenaDescription}
                            </p>
                        </div>
                    </Link>

                    <Link
                        href="/"
                        onClick={dismiss}
                        className="flex items-start gap-4 p-4 rounded-xl border border-gray-700 hover:border-gray-600 hover:bg-white/5 transition-colors group"
                    >
                        <div className="p-2 rounded-lg bg-white/5 text-gray-400 group-hover:text-parchment transition-colors mt-0.5">
                            <MessageSquareText size={22} />
                        </div>
                        <div>
                            <h3 className="font-semibold text-parchment text-base">
                                {t.welcome.mainChatTitle}
                            </h3>
                            <p className="text-sm text-gray-400 mt-1 leading-relaxed">
                                {t.welcome.mainChatDescription}
                            </p>
                        </div>
                    </Link>
                </div>

                <p className="text-xs text-gray-500 mt-5 text-center">
                    {t.welcome.footer}
                </p>
                <p className="text-[10px] text-gray-600 mt-2 text-center leading-relaxed">
                    {t.welcome.corpusNote}
                </p>
            </div>
        </div>
    );
}
