"use client";

import Link from "next/link";
import { useI18n } from "@/i18n/context";

export function PublicEvaluation() {
    const { locale } = useI18n();
    const chinese = locale !== "en";
    return <main className="max-w-3xl mx-auto px-5 py-10 text-parchment">
        <Link href="/" className="text-primary text-sm">← {chinese ? "回到示例" : "Back to the showcase"}</Link>
        <p className="uppercase text-xs tracking-widest text-parchment/50 mt-10 mb-3">{chinese ? "历史研究记录" : "Historical research record"}</p>
        <h1 className="font-serif text-3xl mb-5">{chinese ? "Arena：偏好评价" : "Arena: a preference evaluation"}</h1>
        <p className="text-parchment/70 leading-relaxed mb-6">{chinese ? "最终报告记录了59次盲测偏好投票：38次偏好TCM-Sage，15次偏好普通模型，6次平局。这是报告中的固定历史样本，不是实时统计。" : "The final report records 59 blind preference votes: 38 preferred TCM-Sage, 15 preferred the plain-model baseline, and 6 were ties. This is a frozen historical sample from the report, not a live counter."}</p>
        <div className="grid grid-cols-3 gap-3 mb-6">{[{ label: "TCM-Sage", value: 38 }, { label: chinese ? "普通模型" : "Plain model", value: 15 }, { label: chinese ? "平局" : "Tie", value: 6 }].map(item => <div key={item.label} className="rounded-xl border border-white/10 bg-sidebar-dark p-4"><div className="text-3xl font-serif mb-2">{item.value}</div><div className="text-xs text-parchment/60">{item.label}</div></div>)}</div>
        <p className="text-sm text-parchment/65 leading-relaxed mb-5">{chinese ? "投票衡量整体偏好，不能证明临床有效性、引用准确率或不会产生幻觉。报告指出评价者以中医学生为主，样本有限，系统在评价期间也有调整。59票不代表59位独立参与者。" : "Votes measure overall preference. They do not establish clinical effectiveness, citation accuracy or absence of hallucinations. The report notes a limited evaluator pool, mostly TCM students, and iterative system changes during evaluation. Fifty-nine votes does not imply 59 independent participants."}</p>
        <p className="rounded-xl border border-primary/25 p-4 text-sm mb-6">{chinese ? "公开展示未开放实时Arena生成或投票，也不会将新访客数据混入该研究样本。" : "Public live Arena generation and voting are disabled. Visitor activity is not mixed into this historical research sample."}</p>
        <a href="https://github.com/AndyZHENG0715/TCM-Sage/blob/main/docs/report/chapters/chapter5.tex" target="_blank" rel="noreferrer" className="text-primary underline text-sm">{chinese ? "来源：最终报告，第5章" : "Source: final report, chapter 5"}</a>
    </main>;
}
