"use client";

import { useRef, useState } from "react";
import Link from "next/link";
import { ArrowRight, BookOpen, Globe2, LockKeyhole, Network, Play, Square } from "lucide-react";
import { useI18n } from "@/i18n/context";
import { CitationPanel } from "./CitationPanel";
import { MessageBubble } from "./MessageBubble";
import { KGViewer } from "./KGViewer";
import { streamQuery } from "@/lib/api";
import { RECORDED_EXAMPLES, recordedMessage, replayExample, setLiveAccessCode } from "@/lib/showcase";
import type { Citation, GraphCitation, Message } from "@/lib/types";

export function PublicShowcase() {
    const { locale, setLocale } = useI18n();
    const chinese = locale !== "en";
    const [selected, setSelected] = useState(RECORDED_EXAMPLES[0]);
    const [message, setMessage] = useState<Message | null>(null);
    const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
    const [busy, setBusy] = useState(false);
    const [live, setLive] = useState(false);
    const [code, setCode] = useState("");
    const [question, setQuestion] = useState("");
    const [error, setError] = useState("");
    const request = useRef<AbortController | null>(null);
    const output = message ?? recordedMessage(selected, chinese);
    const graphCitation = selected.citations.find(c => c.type === "graph") as GraphCitation | undefined;

    const stop = () => {
        request.current?.abort();
        request.current = null;
        setBusy(false);
    };

    const replay = async (example = selected) => {
        stop();
        const controller = new AbortController();
        request.current = controller;
        setSelected(example);
        setActiveCitation(null);
        setError("");
        setLiveAccessCode("");
        setLive(false);
        setBusy(true);
        const prepared = recordedMessage(example, chinese);
        setMessage({ ...prepared, content: "" });
        let content = "";
        for await (const part of replayExample(example, chinese, controller.signal)) {
            content += part;
            setMessage({ ...prepared, content });
        }
        if (request.current === controller) {
            setBusy(false);
            request.current = null;
        }
    };

    const askLive = async () => {
        if (!question.trim() || !code.trim() || busy) return;
        setLiveAccessCode(code);
        setError("");
        setActiveCitation(null);
        setBusy(true);
        const controller = new AbortController();
        request.current = controller;
        let next: Message = { role: "assistant", content: "", timestamp: Date.now() };
        setMessage(next);
        try {
            for await (const event of streamQuery(question.trim(), [], undefined, controller.signal)) {
                if (controller.signal.aborted) break;
                if (event.type === "text") next = { ...next, content: next.content + event.content };
                else if (event.type === "metadata") next = { ...next, citations: event.citations, verification: event.verification, citationBounds: event.citationBounds };
                else throw new Error("Live request unavailable");
                setMessage(next);
            }
        } catch {
            if (!controller.signal.aborted) setError(chinese ? "实时服务暂不可用或访问被限制。可以继续使用下方预录示例；没有自动切换到付费服务。" : "Live access is unavailable or limited. Recorded examples remain available below; no paid fallback was selected.");
        } finally {
            if (request.current === controller) {
                setBusy(false);
                request.current = null;
            }
        }
    };

    return (
        <main className="min-h-screen bg-background-dark text-parchment">
            <header className="border-b border-white/10">
                <div className="max-w-6xl mx-auto flex min-h-16 items-center justify-between gap-3 px-4 sm:px-6">
                    <Link href="/" className="font-serif text-xl tracking-tight">TCM-Sage<span className="text-primary">.</span></Link>
                    <nav className="flex items-center gap-3 sm:gap-6 text-xs sm:text-sm text-parchment/70">
                        <Link href="/arena" className="hover:text-parchment">{chinese ? "研究记录" : "Evaluation"}</Link>
                        <button onClick={() => { stop(); setMessage(null); setLocale(chinese ? "en" : "zh"); }} className="flex items-center gap-1 hover:text-parchment" aria-label="Change language"><Globe2 size={15} />{locale === "en" ? "中文" : "English"}</button>
                        <button onClick={() => { stop(); setLive(!live); setMessage(null); setError(""); setLiveAccessCode(""); }} className="flex items-center gap-1 hover:text-parchment"><LockKeyhole size={14} />{live ? (chinese ? "回到演示" : "Showcase") : (chinese ? "实时预览" : "Live preview")}</button>
                    </nav>
                </div>
            </header>

            <div className="max-w-6xl mx-auto px-4 sm:px-6 py-7 sm:py-10">
                <section className="max-w-3xl mb-5">
                    <p className="text-xs uppercase tracking-[.18em] text-primary mb-2">{chinese ? "古籍 · 引用 · 可追溯" : "Classical texts · citations · context"}</p>
                    <h1 className="font-serif text-3xl sm:text-4xl leading-tight mb-3">{chinese ? "从答案，回到原文。" : "An answer is a starting point. Read the source."}</h1>
                    <p className="text-sm text-parchment/65 leading-relaxed">{chinese ? "探索中医古籍中的文字、上下文和知识图谱关系。下方为人工整理的预录示例，不是实时生成。" : "Explore passages, context and recorded graph relationships from classical Chinese medicine texts. Start with a curated example, prepared in advance."}</p>
                </section>

                {live && <section className="max-w-3xl mb-6 rounded-xl border border-primary/30 bg-sidebar-dark p-4 space-y-3" aria-label="Live preview access">
                    <h2 className="font-semibold text-sm">{chinese ? "受控实时预览" : "Controlled live preview"}</h2>
                    <p className="text-xs text-parchment/65">{chinese ? "需要项目访问码；发送问题时由服务器验证。请勿输入姓名、联系方式或患者身份资料。免费服务可能休眠、限流或不可用。" : "A project access code is checked by the server when you send. Do not enter names, contact details or patient identifiers. Free services may sleep, be limited or unavailable."}</p>
                    <input value={code} onChange={e => setCode(e.target.value)} type="password" autoComplete="off" maxLength={256} placeholder={chinese ? "项目访问码（不是模型 API Key）" : "Project access code (not a model API key)"} aria-label="Project access code" className="w-full px-3 py-2 text-sm rounded-lg border border-white/15 bg-background-dark" />
                    <textarea value={question} onChange={e => setQuestion(e.target.value)} maxLength={2000} rows={3} placeholder={chinese ? "输入古籍研究问题…" : "Ask a question about the classical texts…"} aria-label="Research question" className="w-full px-3 py-2 text-sm rounded-lg border border-white/15 bg-background-dark resize-y" />
                    <button onClick={() => void askLive()} disabled={busy || !code.trim() || !question.trim()} className="inline-flex items-center gap-2 rounded-lg bg-primary-dark px-4 py-2 text-sm text-white disabled:opacity-40">{chinese ? "发送实时问题" : "Send live question"}<ArrowRight size={14} /></button>
                </section>}

                <section aria-label="Recorded examples" className="mb-5">
                    <div className="flex gap-2 overflow-x-auto pb-2">
                        {RECORDED_EXAMPLES.map(example => <button key={example.id} onClick={() => void replay(example)} className={`shrink-0 rounded-lg border px-3 py-2 text-xs transition-colors ${selected.id === example.id && !live ? "border-primary/50 bg-primary/10 text-parchment" : "border-white/10 text-parchment/60 hover:border-white/25"}`}>{chinese ? example.titleZh : example.title}</button>)}
                    </div>
                </section>

                <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_280px]">
                    <section className="min-w-0" aria-live="polite">
                        <div className="flex flex-wrap items-center justify-between gap-2 mb-3 text-xs text-parchment/60">
                            <span>{output.recorded ? (chinese ? "预录示例 · 人工整理 · 原文可核对" : "Recorded example · manually curated · inspect the source") : (chinese ? "实时输出 · 可能出错" : "Live output · fallible")}</span>
                            {busy ? <button onClick={stop} className="flex items-center gap-1"><Square size={12} />{chinese ? "停止" : "Stop"}</button> : <button onClick={() => void replay()} className="flex items-center gap-1 text-primary"><Play size={13} />{chinese ? "回放示例" : "Replay example"}</button>}
                        </div>
                        <h2 className="font-serif text-lg mb-4">{output.recorded ? (chinese ? selected.questionZh : selected.question) : question}</h2>
                        {output.content ? <MessageBubble message={output} onCitationClick={setActiveCitation} /> : <div className="rounded-xl bg-parchment p-6 text-parchment-text text-sm">{chinese ? "准备输出…" : "Preparing response…"}</div>}
                        {error && <p role="alert" className="rounded-lg border border-amber-600/30 bg-amber-900/15 p-3 text-sm text-amber-100 mb-4">{error}</p>}
                        <p className="text-xs text-parchment/50 leading-relaxed">{chinese ? "研究原型，非医疗建议。回答、引用和图谱都可能出错，请核对原始来源。" : "Research prototype, not medical advice. Answers, citations and graph relationships can be wrong. Verify the original source."}</p>
                    </section>

                    {output.recorded && <aside className="rounded-xl border border-white/10 bg-sidebar-dark p-4 self-start">
                        <h2 className="text-sm font-semibold flex gap-2 items-center mb-2"><Network size={16} />{chinese ? "预录图谱片段" : "Recorded graph snapshot"}</h2>
                        <p className="text-xs text-parchment/55 leading-relaxed mb-3">{chinese ? "SymMap 2.0 中记录的关系；不是临床有效性证明，也不等同于古籍引文。" : "Relationships recorded in SymMap 2.0; separate from the classical citation and not evidence of clinical benefit."}</p>
                        {graphCitation && <div className="h-52 rounded-lg overflow-hidden bg-parchment"><KGViewer citation={graphCitation} subgraph={selected.graph} /></div>}
                        <Link href={`/kg/${encodeURIComponent(selected.graphEntity)}`} className="mt-3 inline-flex text-xs items-center gap-1 text-primary">{chinese ? "查看已记录关系" : "Explore recorded relationships"}<ArrowRight size={12} /></Link>
                    </aside>}
                </div>

                <section className="mt-12 pt-6 border-t border-white/10 grid sm:grid-cols-3 gap-5 text-sm">
                    <div><h2 className="flex gap-2 items-center font-semibold mb-2"><BookOpen size={15} />{chinese ? "检索" : "Retrieve"}</h2><p className="text-parchment/55 text-xs leading-relaxed">{chinese ? "实时路径检索古籍段落，并可结合图谱；示例路径只读取本地记录。" : "The live path retrieves classical passages and may add graph context. Showcase examples read local records only."}</p></div>
                    <div><h2 className="font-semibold mb-2">{chinese ? "生成与检查" : "Generate and check"}</h2><p className="text-parchment/55 text-xs leading-relaxed">{chinese ? "研究系统包含问题分类、生成和验证步骤。验证也会失败，不能保证正确。" : "The research pipeline classifies, generates and checks responses. Verification is fallible and cannot guarantee correctness."}</p></div>
                    <div><h2 className="font-semibold mb-2">{chinese ? "回到证据" : "Return to evidence"}</h2><p className="text-parchment/55 text-xs leading-relaxed">{chinese ? "打开引用查看原文。古籍中的历史主张不等于现代科学证据。" : "Open citations to inspect the text. Historical claims in a classical source are not modern scientific evidence."}</p></div>
                </section>
            </div>
            {activeCitation && <div className="fixed inset-y-0 right-0 z-50 w-full sm:w-[400px] lg:w-[450px] shadow-2xl"><CitationPanel key={`${activeCitation.type}-${activeCitation.number}`} citation={activeCitation} onClose={() => setActiveCitation(null)} /></div>}
        </main>
    );
}
