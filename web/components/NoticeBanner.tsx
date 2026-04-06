"use client";

import { useState } from "react";
import { X } from "lucide-react";
import { useI18n } from "@/i18n/context";

export function NoticeBanner() {
    const [dismissed, setDismissed] = useState(false);
    const { t } = useI18n();

    if (dismissed) return null;

    return (
        <div className="relative bg-amber-500/10 border-b border-amber-500/20 px-4 py-2 text-center text-sm text-amber-200 shrink-0">
            <span>{t.notice.banner}</span>
            <button
                onClick={() => setDismissed(true)}
                className="absolute right-3 top-1/2 -translate-y-1/2 p-1 text-amber-400 hover:text-amber-200 transition-colors"
                aria-label={t.common.close}
            >
                <X size={14} />
            </button>
        </div>
    );
}
