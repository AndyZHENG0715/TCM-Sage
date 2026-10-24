"use client";

import { useI18n } from "@/i18n/context";

export function NoticeBanner() {
    const { locale } = useI18n();

    return (
        <div className="relative bg-amber-500/10 border-b border-amber-500/20 px-4 py-2 text-center text-sm text-amber-200 shrink-0">
            <span>{locale === "en" ? "Research prototype — not medical advice. Answers and citations can be wrong; verify original sources." : "研究原型，非医疗建议。回答与引用可能出错，请核对原始来源。"}</span>
        </div>
    );
}
