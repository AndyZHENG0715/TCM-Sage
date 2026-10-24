import data from "./showcase-data.json";
import type { Citation, Message } from "./types";
import type { BookContent, ChunkContext, SubgraphResponse } from "./api";

export type RecordedExample = {
    id: string;
    title: string;
    titleZh: string;
    question: string;
    questionZh: string;
    answer: string;
    answerZh: string;
    citations: Citation[];
    graphEntity: string;
    graph: SubgraphResponse;
};

const showcase = data as unknown as {
    examples: RecordedExample[];
    contexts: Record<string, ChunkContext>;
    books: Record<string, string>;
    graphs: Record<string, SubgraphResponse>;
};

export const RECORDED_EXAMPLES = showcase.examples;
export const PRESENTATION_MODE = process.env.NEXT_PUBLIC_PRESENTATION_MODE === "1";
let accessCode = "";

// A session-memory credential: never bundled, logged, put in a URL or persisted.
export function setLiveAccessCode(value: string) { accessCode = value.trim(); }
export function getLiveAccessCode() { return accessCode; }

export function recordedContext(id: string): ChunkContext | undefined {
    return showcase.contexts[id];
}
export function recordedBook(name: string): BookContent | undefined {
    return Object.hasOwn(showcase.books, name) ? { content: showcase.books[name] } : undefined;
}
export function recordedGraph(entity: string): SubgraphResponse | undefined {
    let name = entity;
    try { name = decodeURIComponent(entity); } catch { /* Keep the original lookup key. */ }
    return showcase.graphs[name];
}
export function recordedGraphSearch(query: string, limit: number) {
    const nodes = new Map<string, SubgraphResponse["nodes"][number]>();
    for (const graph of Object.values(showcase.graphs)) {
        for (const node of graph.nodes) nodes.set(node.id, node);
    }
    return [...nodes.values()].filter(node => node.label.toLowerCase().includes(query.toLowerCase())).slice(0, Math.min(limit, 20));
}

export function recordedMessage(example: RecordedExample, chinese: boolean): Message {
    return {
        role: "assistant",
        content: chinese ? example.answerZh : example.answer,
        citations: example.citations,
        timestamp: 0,
        recorded: true,
    };
}

export async function* replayExample(example: RecordedExample, chinese: boolean, signal?: AbortSignal) {
    const text = chinese ? example.answerZh : example.answer;
    for (let start = 0; start < text.length; start += 24) {
        if (signal?.aborted) return;
        yield text.slice(start, start + 24);
        await new Promise(resolve => setTimeout(resolve, 18));
    }
}
