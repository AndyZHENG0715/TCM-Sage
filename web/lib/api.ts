import { Citation } from "./types";

const BACKEND_URL =
    process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export type StreamEvent =
    | { type: "text"; content: string }
    | {
        type: "metadata";
        citations: Citation[];
        severity: "informational" | "prescriptive";
        verification: unknown;
    }
    | { type: "error"; message: string };

// Improved SSE Parser
export async function* streamQuery(
    question: string
): AsyncGenerator<StreamEvent, void, unknown> {
    const response = await fetch(`${BACKEND_URL}/query`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API Error: ${response.status} ${errorText}`);
    }

    if (!response.body) throw new Error("No response body");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Split by double newline to separate events
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || ""; // Keep incomplete part

        for (const part of parts) {
            const lines = part.split("\n");
            let eventType = "message";
            let data = "";

            for (const line of lines) {
                if (line.startsWith("event:")) {
                    eventType = line.slice(6).trim();
                } else if (line.startsWith("data:")) {
                    data = line.slice(5).trim();
                }
            }

            if (eventType === "metadata") {
                try {
                    const parsed = JSON.parse(data);
                    yield {
                        type: "metadata",
                        citations: parsed.citations,
                        severity: parsed.severity,
                        verification: parsed.verification
                    };
                } catch (e) {
                    console.error("Failed to parse metadata", e);
                }
            } else if (eventType === "error") {
                try {
                    const parsed = JSON.parse(data);
                    yield { type: "error", message: parsed.message };
                } catch {
                    yield { type: "error", message: data };
                }
            } else if (data) {
                // Text chunk
                // Unescape newlines
                const content = data.replace(/\\n/g, "\n");
                yield { type: "text", content };
            }
        }
    }
}

export async function fetchConfig() {
    try {
        const res = await fetch(`${BACKEND_URL}/config`);
        if (!res.ok) throw new Error("Failed to fetch config");
        return await res.json();
    } catch (error) {
        console.error("Error fetching config:", error);
        return null; // Return null to fallback to defaults
    }
}

export async function healthCheck() {
    try {
        const res = await fetch(`${BACKEND_URL}/health`);
        return res.ok;
    } catch {
        return false;
    }
}

export type ChunkContext = {
    chunk_id: string;
    book: string;
    chapter: string;
    chunk_index: number;
    full_chapter_text: string;
    highlight_start: number;
    highlight_end: number;
    total_chunks_in_chapter: number;
};

export async function fetchChunkContext(chunkId: string): Promise<ChunkContext> {
    const res = await fetch(
        `${BACKEND_URL}/source/${encodeURIComponent(chunkId)}/context`
    );
    if (!res.ok) {
        throw new Error(`Failed to fetch context: ${res.status}`);
    }
    return res.json();
}
