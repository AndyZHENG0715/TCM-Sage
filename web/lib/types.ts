export type TextCitation = {
    number: number;
    type: "text";
    source: string;
    content: string;
    chunk_id?: string;
    score: number;
};

export type GraphCitation = {
    number: number;
    type: "graph";
    fact: string;
    depth: number;
    source_ref?: Record<string, unknown>;
};

export type Citation = TextCitation | GraphCitation;

export type Message = {
    role: "user" | "assistant";
    content: string;
    citations?: Citation[];
    severity?: "informational" | "prescriptive"; // Based on backend
    verification?: {
        status: string;
        explanation: string;
    };
    timestamp: number;
};

export type Settings = {
    llmProvider: string;
    llmModel: string;
    informationalTemperature: number;
    prescriptiveTemperature: number;
    retrievalK: number;
    hybridRetrieval: boolean;
    graphDepth: number;
    responseStyle: "concise" | "detailed" | "academic";
    citationFormat: "chapter" | "section";
    themeMode: "dark";
};

export type ChatSession = {
    id: string;
    title: string;
    messages: Message[];
    createdAt: number;
    updatedAt: number;
};

export const DEFAULT_SETTINGS: Settings = {
    llmProvider: "alibaba",
    llmModel: "",
    informationalTemperature: 0.1,
    prescriptiveTemperature: 0.0,
    retrievalK: 5,
    hybridRetrieval: true,
    graphDepth: 1,
    responseStyle: "detailed",
    citationFormat: "chapter",
    themeMode: "dark",
};
