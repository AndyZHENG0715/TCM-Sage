import {
    Citation,
    CitationBounds,
    DEFAULT_SETTINGS,
    DEFAULT_SETTINGS_CAPABILITIES,
    Settings,
    SettingsCapabilities,
    Verification,
} from "./types";

const BACKEND_URL = "/api/backend";

type QuerySettingsPayload = {
    provider: string;
    model: string | null;
    informational_temperature: number;
    prescriptive_temperature: number;
    classifier_follow_main: boolean;
    classifier_provider: string | null;
    classifier_model: string | null;
    verifier_follow_main: boolean;
    verifier_provider: string | null;
    verifier_model: string | null;
    retrieval_k: number;
    hybrid_retrieval_enabled: boolean;
    graph_depth: number;
    graph_max_results: number;
};

type RuntimeConfigResponse = {
    provider: string;
    model: string | null;
    informational_temperature: number;
    prescriptive_temperature: number;
    classifier_follow_main: boolean;
    classifier_provider: string;
    classifier_model: string | null;
    verifier_follow_main: boolean;
    verifier_provider: string;
    verifier_model: string | null;
    retrieval_k: number;
    hybrid_enabled: boolean;
    hybrid_available: boolean;
    graph_depth: number;
    graph_max_results: number;
};

export type SettingsBootstrap = {
    defaultSettings: Settings;
    capabilities: SettingsCapabilities;
};

export type ChunkContext = {
    chunk_id: string;
    book: string;
    chapter: string;
    chapter_display: string;
    chunk_index: number;
    full_chapter_text: string;
    highlight_start: number;
    highlight_end: number;
    paragraph_text: string;
    paragraph_highlight_start: number;
    paragraph_highlight_end: number;
    total_chunks_in_chapter: number;
};

export type BookContent = {
    content: string;
};

export interface SubgraphResponse {
    nodes: Array<{ id: string; label: string; type: string }>;
    edges: Array<{ source: string; target: string; label: string }>;
    cited_ids: string[];
}

export type StreamEvent =
    | { type: "text"; content: string }
    | {
        type: "metadata";
        citations: Citation[];
        severity: "informational" | "prescriptive";
        verification?: Verification;
        citationBounds?: CitationBounds;
    }
    | { type: "error"; message: string };

function normalizeVerification(parsed: Record<string, unknown>): Verification | undefined {
    const verification = parsed.verification;
    if (
        verification &&
        typeof verification === "object" &&
        "status" in verification &&
        "explanation" in verification
    ) {
        return verification as Verification;
    }

    const legacyStatus = parsed.verification_result;
    if (typeof legacyStatus === "string") {
        return {
            status: legacyStatus,
            explanation:
                legacyStatus === "SUPPORTED"
                    ? "The answer appears supported by the retrieved citations."
                    : "The answer may include claims not directly supported by the retrieved citations.",
        };
    }

    return undefined;
}

function toOptionalString(value: string): string | null {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
}

function serializeRuntimeSettings(settings: Settings): QuerySettingsPayload {
    return {
        provider: settings.llmProvider,
        model: toOptionalString(settings.llmModel),
        informational_temperature: settings.informationalTemperature,
        prescriptive_temperature: settings.prescriptiveTemperature,
        classifier_follow_main: settings.classifierFollowMain,
        classifier_provider: toOptionalString(settings.classifierProvider),
        classifier_model: toOptionalString(settings.classifierModel),
        verifier_follow_main: settings.verifierFollowMain,
        verifier_provider: toOptionalString(settings.verifierProvider),
        verifier_model: toOptionalString(settings.verifierModel),
        retrieval_k: settings.retrievalK,
        hybrid_retrieval_enabled: settings.hybridRetrieval,
        graph_depth: settings.graphDepth,
        graph_max_results: settings.graphMaxResults,
    };
}

function mapConfigToSettings(config: RuntimeConfigResponse): SettingsBootstrap {
    const capabilities: SettingsCapabilities = {
        hybridAvailable:
            typeof config.hybrid_available === "boolean"
                ? config.hybrid_available
                : DEFAULT_SETTINGS_CAPABILITIES.hybridAvailable,
    };

    const defaultSettings: Settings = {
        ...DEFAULT_SETTINGS,
        llmProvider: config.provider,
        llmModel: config.model ?? "",
        informationalTemperature: config.informational_temperature,
        prescriptiveTemperature: config.prescriptive_temperature,
        classifierFollowMain: config.classifier_follow_main,
        classifierProvider: config.classifier_provider ?? "",
        classifierModel: config.classifier_model ?? "",
        verifierFollowMain: config.verifier_follow_main,
        verifierProvider: config.verifier_provider ?? "",
        verifierModel: config.verifier_model ?? "",
        retrievalK: config.retrieval_k,
        hybridRetrieval: capabilities.hybridAvailable ? config.hybrid_enabled : false,
        graphDepth: config.graph_depth,
        graphMaxResults: config.graph_max_results ?? 20,
    };

    return {
        defaultSettings,
        capabilities,
    };
}

export async function* streamQuery(
    question: string,
    chatHistory: { role: string; content: string }[] = [],
    settings?: Settings
): AsyncGenerator<StreamEvent, void, unknown> {
    const requestBody: {
        question: string;
        chat_history: { role: string; content: string }[];
        settings?: QuerySettingsPayload;
    } = {
        question,
        chat_history: chatHistory,
    };

    if (settings) {
        requestBody.settings = serializeRuntimeSettings(settings);
    }

    const queryUrl = `${BACKEND_URL}/query`;

    let response: Response;
    try {
        response = await fetch(queryUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(requestBody),
        });
    } catch (error) {
        throw error;
    }

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API Error: ${response.status} ${errorText}`);
    }

    if (!response.body) {
        throw new Error("No response body");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) {
            break;
        }

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

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
                    const parsed = JSON.parse(data) as Record<string, unknown>;
                    yield {
                        type: "metadata",
                        citations: (parsed.citations as Citation[]) ?? [],
                        severity: parsed.severity as "informational" | "prescriptive",
                        verification: normalizeVerification(parsed),
                        citationBounds: parsed.citation_bounds as CitationBounds | undefined,
                    };
                } catch (error) {
                    console.error("Failed to parse metadata", error);
                }
                continue;
            }

            if (eventType === "error") {
                try {
                    const parsed = JSON.parse(data);
                    yield { type: "error", message: parsed.message };
                } catch {
                    yield { type: "error", message: data };
                }
                continue;
            }

            if (data) {
                yield { type: "text", content: data.replace(/\\n/g, "\n") };
            }
        }
    }
}

export async function fetchConfig(): Promise<SettingsBootstrap | null> {
    try {
        const [res, arenaModels] = await Promise.all([
            fetch(`${BACKEND_URL}/config`),
            fetchArenaModels(),
        ]);
        if (!res.ok) {
            throw new Error("Failed to fetch config");
        }

        const parsed = (await res.json()) as RuntimeConfigResponse;
        const bootstrap = mapConfigToSettings(parsed);
        return {
            ...bootstrap,
            defaultSettings: {
                ...bootstrap.defaultSettings,
                arenaModels: {
                    flash: arenaModels.flash?.trim() || bootstrap.defaultSettings.arenaModels.flash,
                    plus: arenaModels.plus?.trim() || bootstrap.defaultSettings.arenaModels.plus,
                    max: arenaModels.max?.trim() || bootstrap.defaultSettings.arenaModels.max,
                },
            },
        };
    } catch (error) {
        console.error("Error fetching config:", error);
        return null;
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

export async function fetchChunkContext(chunkId: string): Promise<ChunkContext> {
    let normalizedChunkId = chunkId;
    try {
        const decoded = decodeURIComponent(chunkId);
        if (decoded) {
            normalizedChunkId = decoded;
        }
    } catch {
        normalizedChunkId = chunkId;
    }

    const requestUrl = `${BACKEND_URL}/source/${encodeURIComponent(normalizedChunkId)}/context`;

    const res = await fetch(requestUrl);
    if (!res.ok) {
        throw new Error(`Failed to fetch context: ${res.status}`);
    }
    return res.json();
}

export async function fetchSubgraph(entity: string, hops: number = 2): Promise<SubgraphResponse> {
    const params = new URLSearchParams({ entity, hops: String(hops) });
    const res = await fetch(`${BACKEND_URL}/graph/subgraph?${params.toString()}`);
    if (!res.ok) {
        return { nodes: [], edges: [], cited_ids: [] };
    }
    return res.json() as Promise<SubgraphResponse>;
}

export interface GraphSearchResult {
    id: string;
    label: string;
    type: string;
}

export async function fetchGraphSearch(query: string, limit: number = 20): Promise<GraphSearchResult[]> {
    const params = new URLSearchParams({ q: query, limit: String(limit) });
    const res = await fetch(`${BACKEND_URL}/graph/search?${params.toString()}`);
    if (!res.ok) {
        return [];
    }
    const data = (await res.json()) as { results: GraphSearchResult[] };
    return data.results;
}

export async function fetchBookContent(bookName: string): Promise<BookContent> {
    const res = await fetch(`${BACKEND_URL}/books/${encodeURIComponent(bookName)}`);
    if (!res.ok) {
        throw new Error(`Failed to fetch book: ${res.status}`);
    }
    return res.json();
}

// ─────────────────────────────────────────────────────
// Arena API functions
// ─────────────────────────────────────────────────────

export type ArenaSSEEvent =
    | { type: "text_a"; content: string }
    | { type: "text_b"; content: string }
    | { type: "metadata_a"; data: Record<string, unknown> }
    | { type: "metadata_b"; data: Record<string, unknown> }
    | {
          type: "arena_config";
          data: {
              position_mapping: Record<string, string>;
              session_id: string;
              round_number: number;
          };
      }
    | { type: "error"; data: Record<string, unknown> };

export async function* streamArenaQuery(
    question: string,
    chatHistoryA: { role: string; content: string }[],
    chatHistoryB: { role: string; content: string }[],
    modelName: string,
    sessionId: string,
    roundNumber: number,
    signal?: AbortSignal
): AsyncGenerator<ArenaSSEEvent> {
    const res = await fetch(`${BACKEND_URL}/arena/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            question,
            chat_history_a: chatHistoryA,
            chat_history_b: chatHistoryB,
            model_name: modelName,
            session_id: sessionId,
            round_number: roundNumber,
        }),
        signal,
    });

    if (!res.ok || !res.body) {
        throw new Error(`Arena query failed: ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let currentEvent = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
            if (line.startsWith("event: ")) {
                currentEvent = line.slice(7).trim();
            } else if (line.startsWith("data: ")) {
                const raw = line.slice(6);
                if (currentEvent === "text_a") {
                    yield { type: "text_a", content: raw.replace(/\\n/g, "\n") };
                } else if (currentEvent === "text_b") {
                    yield { type: "text_b", content: raw.replace(/\\n/g, "\n") };
                } else if (currentEvent === "metadata_a") {
                    try {
                        yield { type: "metadata_a", data: JSON.parse(raw) };
                    } catch {
                        // skip
                    }
                } else if (currentEvent === "metadata_b") {
                    try {
                        yield { type: "metadata_b", data: JSON.parse(raw) };
                    } catch {
                        // skip
                    }
                } else if (currentEvent === "arena_config") {
                    try {
                        yield { type: "arena_config", data: JSON.parse(raw) };
                    } catch {
                        // skip
                    }
                } else if (currentEvent === "error") {
                    try {
                        yield { type: "error", data: JSON.parse(raw) };
                    } catch {
                        // skip
                    }
                }
                currentEvent = "";
            } else if (line === "") {
                currentEvent = "";
            }
        }
    }
}

export async function submitArenaVote(vote: {
    session_id: string;
    round_number: number;
    query: string;
    response_a: string;
    response_b: string;
    model_name: string;
    position_mapping: Record<string, string>;
    vote: "a" | "b" | "tie";
    comment?: string | null;
    timestamp?: string;
}): Promise<void> {
    await fetch(`${BACKEND_URL}/arena/vote`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...vote, timestamp: vote.timestamp ?? new Date().toISOString() }),
    });
}

export async function fetchArenaModels(): Promise<Partial<Settings["arenaModels"]>> {
    try {
        const res = await fetch(`${BACKEND_URL}/arena/models`);
        if (!res.ok) return {};
        return (await res.json()) as Partial<Settings["arenaModels"]>;
    } catch {
        return {};
    }
}

export async function fetchArenaStats() {
    const res = await fetch(`${BACKEND_URL}/arena/stats`);
    if (!res.ok) throw new Error("Failed to fetch arena stats");
    return res.json();
}
