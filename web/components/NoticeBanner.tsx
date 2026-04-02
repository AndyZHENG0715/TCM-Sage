"use client";

import { useState } from "react";
import { X } from "lucide-react";

export function NoticeBanner() {
    const [dismissed, setDismissed] = useState(false);

    if (dismissed) return null;

    return (
        <div className="relative bg-amber-500/10 border-b border-amber-500/20 px-4 py-2 text-center text-sm text-amber-200 shrink-0">
            <span>⚠️ This system is under active development. Some features may be unstable. Thank you for your patience!</span>
            <button
                onClick={() => setDismissed(true)}
                className="absolute right-3 top-1/2 -translate-y-1/2 p-1 text-amber-400 hover:text-amber-200 transition-colors"
                aria-label="Dismiss notice"
            >
                <X size={14} />
            </button>
        </div>
    );
}