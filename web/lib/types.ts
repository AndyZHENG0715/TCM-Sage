export type Verification = {
    status: string;
    explanation: string;
};

export type CitationBounds = {
    is_valid: boolean;
    out_of_range: number[];
    found_citations: number[];
};

export type TextCitation = {
    number: number;
    type: "text";
    source: string;
    content: string;
    chunk_id?: string;
    score: number;
    relevance_percent: number;
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
    severity?: "informational" | "prescriptive";
    verification?: Verification;
    citationBounds?: CitationBounds;
    timestamp: number;
};

export type SettingsCapabilities = {
    hybridAvailable: boolean;
};

export type Settings = {
    llmProvider: string;
    llmModel: string;
    informationalTemperature: number;
    prescriptiveTemperature: number;
    classifierFollowMain: boolean;
    classifierProvider: string;
    classifierModel: string;
    verifierFollowMain: boolean;
    verifierProvider: string;
    verifierModel: string;
    arenaModels: {
        flash: string;
        plus: string;
        max: string;
    };
    retrievalK: number;
    hybridRetrieval: boolean;
    graphDepth: number;
    graphMaxResults: number;
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
    classifierFollowMain: true,
    classifierProvider: "",
    classifierModel: "",
    verifierFollowMain: true,
    verifierProvider: "",
    verifierModel: "",
    arenaModels: {
        flash: "qwen-flash",
        plus: "qwen-plus",
        max: "qwen-max",
    },
    retrievalK: 5,
    hybridRetrieval: true,
    graphDepth: 1,
    graphMaxResults: 20,
    responseStyle: "detailed",
    citationFormat: "chapter",
    themeMode: "dark",
};

export const DEFAULT_SETTINGS_CAPABILITIES: SettingsCapabilities = {
    hybridAvailable: true,
};
