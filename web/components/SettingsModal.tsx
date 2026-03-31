"use client";

import { Settings, SettingsCapabilities } from "@/lib/types";
import { cn } from "@/lib/utils";
import { RotateCcw, Save, X } from "lucide-react";
import { ReactNode, useState } from "react";

interface SettingsModalProps {
    isOpen: boolean;
    onClose: () => void;
    settings: Settings;
    defaultSettings: Settings;
    capabilities: SettingsCapabilities;
    onSave: (settings: Settings) => void;
    onReset: () => void;
}

const PROVIDER_OPTIONS = [
    { label: "Alibaba Cloud (Qwen)", value: "alibaba" },
    { label: "OpenAI", value: "openai" },
    { label: "Anthropic", value: "anthropic" },
    { label: "Google Gemini", value: "google" },
    { label: "OpenRouter", value: "openrouter" },
    { label: "Together AI", value: "together" },
    { label: "Ollama (Local)", value: "ollama" },
    { label: "LM Studio (Local)", value: "lmstudio" },
] as const;

type TabId = "model" | "retrieval" | "output";

function ProviderSelect({
    label,
    value,
    onChange,
    disabled = false,
}: {
    label: string;
    value: string;
    onChange: (value: string) => void;
    disabled?: boolean;
}) {
    return (
        <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">{label}</label>
            <select
                value={value}
                onChange={(event) => onChange(event.target.value)}
                disabled={disabled}
                className="w-full rounded-lg border border-white/10 bg-background-dark px-3 py-2 text-parchment outline-none transition-colors focus:border-primary/50 disabled:cursor-not-allowed disabled:opacity-50"
            >
                {PROVIDER_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                        {option.label}
                    </option>
                ))}
            </select>
        </div>
    );
}

function ModelInput({
    label,
    value,
    placeholder,
    onChange,
    disabled = false,
}: {
    label: string;
    value: string;
    placeholder: string;
    onChange: (value: string) => void;
    disabled?: boolean;
}) {
    return (
        <div className="space-y-2">
            <label className="text-sm font-medium text-gray-300">{label}</label>
            <input
                type="text"
                value={value}
                onChange={(event) => onChange(event.target.value)}
                placeholder={placeholder}
                disabled={disabled}
                className="w-full rounded-lg border border-white/10 bg-background-dark px-3 py-2 text-parchment outline-none transition-colors placeholder:text-gray-600 focus:border-primary/50 disabled:cursor-not-allowed disabled:opacity-50"
            />
        </div>
    );
}

function TemperatureControl({
    label,
    value,
    minLabel,
    maxLabel,
    onChange,
}: {
    label: string;
    value: number;
    minLabel: string;
    maxLabel: string;
    onChange: (value: number) => void;
}) {
    return (
        <div className="space-y-4">
            <div className="flex justify-between">
                <label className="text-sm font-medium text-gray-300">{label}</label>
                <span className="text-xs font-mono text-primary">{value.toFixed(1)}</span>
            </div>
            <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={value}
                onChange={(event) => onChange(parseFloat(event.target.value))}
                className="h-1 w-full cursor-pointer appearance-none rounded-lg bg-white/10 accent-primary"
            />
            <div className="flex justify-between text-xs text-gray-500">
                <span>{minLabel}</span>
                <span>{maxLabel}</span>
            </div>
        </div>
    );
}

function FollowMainToggle({
    title,
    description,
    checked,
    onToggle,
}: {
    title: string;
    description: string;
    checked: boolean;
    onToggle: () => void;
}) {
    return (
        <div className="flex items-center justify-between rounded-lg border border-white/5 bg-background-dark/60 p-4">
            <div>
                <span className="block text-sm font-medium text-parchment">{title}</span>
                <span className="text-xs text-gray-400">{description}</span>
            </div>
            <button
                type="button"
                role="switch"
                aria-checked={checked}
                onClick={onToggle}
                className={cn(
                    "relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-primary/50",
                    checked ? "bg-primary" : "bg-gray-600"
                )}
            >
                <span
                    className={cn(
                        "inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-200",
                        checked ? "translate-x-6" : "translate-x-1"
                    )}
                />
            </button>
        </div>
    );
}

function Section({
    title,
    description,
    children,
}: {
    title: string;
    description: string;
    children: ReactNode;
}) {
    return (
        <section className="space-y-4 rounded-xl border border-white/5 bg-white/5 p-4">
            <div>
                <h3 className="text-base font-serif font-semibold text-parchment">{title}</h3>
                <p className="mt-1 text-xs text-gray-400">{description}</p>
            </div>
            {children}
        </section>
    );
}

export function SettingsModal({
    isOpen,
    onClose,
    settings: initialSettings,
    defaultSettings,
    capabilities,
    onSave,
    onReset,
}: SettingsModalProps) {
    const [localSettings, setLocalSettings] = useState<Settings>(initialSettings);
    const [activeTab, setActiveTab] = useState<TabId>("model");

    if (!isOpen) {
        return null;
    }

    const handleChange = <K extends keyof Settings>(key: K, value: Settings[K]) => {
        setLocalSettings((prev) => ({ ...prev, [key]: value } as Settings));
    };

    const handleSave = () => {
        onSave(localSettings);
        onClose();
    };

    const handleReset = () => {
        onReset();
        setLocalSettings(defaultSettings);
    };

    const tabs: { id: TabId; label: string }[] = [
        { id: "model", label: "Model Parameters" },
        { id: "retrieval", label: "Retrieval & Knowledge" },
        { id: "output", label: "Output Preferences" },
    ];

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm">
            <div className="flex max-h-[90vh] w-full max-w-2xl flex-col overflow-hidden rounded-xl border border-primary/20 bg-[#1a2c2a] shadow-2xl">
                <div className="flex items-center justify-between border-b border-white/5 bg-sidebar-dark p-6">
                    <h2 className="text-xl font-serif font-bold text-parchment">Configuration</h2>
                    <button
                        type="button"
                        onClick={onClose}
                        className="rounded-full p-2 text-gray-400 transition-colors hover:bg-white/5 hover:text-parchment"
                    >
                        <X size={20} />
                    </button>
                </div>

                <div className="flex border-b border-white/5 bg-sidebar-dark/50 px-6">
                    {tabs.map((tab) => (
                        <button
                            key={tab.id}
                            type="button"
                            onClick={() => setActiveTab(tab.id)}
                            className={cn(
                                "border-b-2 px-4 py-3 text-sm font-medium transition-colors",
                                activeTab === tab.id
                                    ? "border-primary text-primary"
                                    : "border-transparent text-gray-400 hover:text-parchment"
                            )}
                        >
                            {tab.label}
                        </button>
                    ))}
                </div>

                <div className="flex-1 space-y-6 overflow-y-auto p-6 text-parchment">
                    {activeTab === "model" && (
                        <div className="space-y-6">
                            <Section
                                title="Main LLM"
                                description="These settings are sent with each query and now control live generation."
                            >
                                <ProviderSelect
                                    label="LLM Provider"
                                    value={localSettings.llmProvider}
                                    onChange={(value) => handleChange("llmProvider", value)}
                                />
                                <ModelInput
                                    label="Model ID"
                                    value={localSettings.llmModel}
                                    placeholder="e.g. qwen/qwen3.5-9b, gemini-2.5-flash"
                                    onChange={(value) => handleChange("llmModel", value)}
                                />
                                <TemperatureControl
                                    label="Temperature (Informational)"
                                    value={localSettings.informationalTemperature}
                                    minLabel="Precise"
                                    maxLabel="Creative"
                                    onChange={(value) => handleChange("informationalTemperature", value)}
                                />
                                <TemperatureControl
                                    label="Temperature (Prescriptive)"
                                    value={localSettings.prescriptiveTemperature}
                                    minLabel="Conservative"
                                    maxLabel="Flexible"
                                    onChange={(value) => handleChange("prescriptiveTemperature", value)}
                                />
                            </Section>

                            <Section
                                title="Classifier"
                                description="Used to route between informational and prescriptive handling. Temperature stays fixed at 0.0."
                            >
                                <FollowMainToggle
                                    title="Follow main LLM"
                                    description="Reuse the main provider and model for classification."
                                    checked={localSettings.classifierFollowMain}
                                    onToggle={() =>
                                        handleChange("classifierFollowMain", !localSettings.classifierFollowMain)
                                    }
                                />
                                {!localSettings.classifierFollowMain && (
                                    <div className="space-y-4 border-l-2 border-primary/20 pl-4">
                                        <ProviderSelect
                                            label="Classifier Provider"
                                            value={localSettings.classifierProvider}
                                            onChange={(value) => handleChange("classifierProvider", value)}
                                        />
                                        <ModelInput
                                            label="Classifier Model ID"
                                            value={localSettings.classifierModel}
                                            placeholder="Optional, leave blank for provider default"
                                            onChange={(value) => handleChange("classifierModel", value)}
                                        />
                                    </div>
                                )}
                            </Section>

                            <Section
                                title="Verifier"
                                description="Used for support checks after generation. Temperature stays fixed at 0.0."
                            >
                                <FollowMainToggle
                                    title="Follow main LLM"
                                    description="Reuse the main provider and model for verification."
                                    checked={localSettings.verifierFollowMain}
                                    onToggle={() =>
                                        handleChange("verifierFollowMain", !localSettings.verifierFollowMain)
                                    }
                                />
                                {!localSettings.verifierFollowMain && (
                                    <div className="space-y-4 border-l-2 border-primary/20 pl-4">
                                        <ProviderSelect
                                            label="Verifier Provider"
                                            value={localSettings.verifierProvider}
                                            onChange={(value) => handleChange("verifierProvider", value)}
                                        />
                                        <ModelInput
                                            label="Verifier Model ID"
                                            value={localSettings.verifierModel}
                                            placeholder="Optional, leave blank for provider default"
                                            onChange={(value) => handleChange("verifierModel", value)}
                                        />
                                    </div>
                                )}
                            </Section>

                            <Section
                                title="Arena Models"
                                description="Model IDs used for Arena blind evaluation. These map to the flash/plus/max tiers."
                            >
                                <ModelInput
                                    label="Flash (Fast)"
                                    value={localSettings.arenaModels.flash}
                                    placeholder="e.g. qwen-turbo"
                                    onChange={(value) =>
                                        handleChange("arenaModels", {
                                            ...localSettings.arenaModels,
                                            flash: value,
                                        })
                                    }
                                />
                                <ModelInput
                                    label="Plus (Balanced)"
                                    value={localSettings.arenaModels.plus}
                                    placeholder="e.g. qwen-plus"
                                    onChange={(value) =>
                                        handleChange("arenaModels", {
                                            ...localSettings.arenaModels,
                                            plus: value,
                                        })
                                    }
                                />
                                <ModelInput
                                    label="Max (Quality)"
                                    value={localSettings.arenaModels.max}
                                    placeholder="e.g. qwen-max"
                                    onChange={(value) =>
                                        handleChange("arenaModels", {
                                            ...localSettings.arenaModels,
                                            max: value,
                                        })
                                    }
                                />
                            </Section>
                        </div>
                    )}

                    {activeTab === "retrieval" && (
                        <div className="space-y-6">
                            <div className="space-y-4">
                                <div className="flex justify-between">
                                    <label className="text-sm font-medium text-gray-300">Retrieval Depth (K)</label>
                                    <span className="text-xs font-mono text-primary">
                                        {localSettings.retrievalK} chunks
                                    </span>
                                </div>
                                <input
                                    type="range"
                                    min="1"
                                    max="20"
                                    step="1"
                                    value={localSettings.retrievalK}
                                    onChange={(event) =>
                                        handleChange("retrievalK", parseInt(event.target.value, 10))
                                    }
                                    className="h-1 w-full cursor-pointer appearance-none rounded-lg bg-white/10 accent-primary"
                                />
                                <div className="flex justify-between text-xs text-gray-500">
                                    <span>Faster</span>
                                    <span>More Context</span>
                                </div>
                            </div>

                            <div className="flex items-center justify-between rounded-lg border border-white/5 bg-white/5 p-4">
                                <div>
                                    <span className="block text-sm font-medium">Knowledge Graph</span>
                                    <span className="text-xs text-gray-400">
                                        {capabilities.hybridAvailable
                                            ? "Enable hybrid retrieval with KG facts"
                                            : "Not available on this backend because graph data is missing"}
                                    </span>
                                </div>
                                <button
                                    type="button"
                                    role="switch"
                                    aria-checked={localSettings.hybridRetrieval}
                                    onClick={() => handleChange("hybridRetrieval", !localSettings.hybridRetrieval)}
                                    disabled={!capabilities.hybridAvailable}
                                    className={cn(
                                        "relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:cursor-not-allowed disabled:opacity-50",
                                        localSettings.hybridRetrieval ? "bg-primary" : "bg-gray-600"
                                    )}
                                >
                                    <span
                                        className={cn(
                                            "inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-200",
                                            localSettings.hybridRetrieval ? "translate-x-6" : "translate-x-1"
                                        )}
                                    />
                                </button>
                            </div>

                            {capabilities.hybridAvailable && localSettings.hybridRetrieval && (
                                <div className="space-y-4 border-l-2 border-primary/20 pl-4">
                                    <div className="flex justify-between">
                                        <label className="text-sm font-medium text-gray-300">
                                            Graph Traversal Depth
                                        </label>
                                        <span className="text-xs font-mono text-primary">
                                            {localSettings.graphDepth}-hop
                                        </span>
                                    </div>
                                    <input
                                        type="range"
                                        min="1"
                                        max="3"
                                        step="1"
                                        value={localSettings.graphDepth}
                                        onChange={(event) =>
                                            handleChange("graphDepth", parseInt(event.target.value, 10))
                                        }
                                        className="h-1 w-full cursor-pointer appearance-none rounded-lg bg-white/10 accent-primary"
                                    />

                                    <div className="space-y-4">
                                        <div className="flex justify-between">
                                            <label className="text-sm font-medium text-gray-300">
                                                KG Max Results
                                            </label>
                                            <span className="text-xs font-mono text-primary">
                                                {localSettings.graphMaxResults} results
                                            </span>
                                        </div>
                                        <input
                                            type="range"
                                            min="1"
                                            max="50"
                                            step="1"
                                            value={localSettings.graphMaxResults}
                                            onChange={(event) =>
                                                handleChange("graphMaxResults", parseInt(event.target.value, 10))
                                            }
                                            className="h-1 w-full cursor-pointer appearance-none rounded-lg bg-white/10 accent-primary"
                                        />
                                        <div className="flex justify-between text-xs text-gray-500">
                                            <span>Fewer</span>
                                            <span>More</span>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {activeTab === "output" && (
                        <div className="space-y-6">
                            <div className="rounded-xl border border-amber-400/20 bg-amber-500/10 p-4 text-sm text-amber-100">
                                `Response Style`, `Citation Format`, and `Theme Mode` are not yet applied at runtime in this build.
                            </div>

                            <fieldset disabled className="space-y-6 opacity-50">
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-gray-300">Response Style</label>
                                    <div className="grid grid-cols-3 gap-2">
                                        {(["concise", "detailed", "academic"] as const).map((style) => (
                                            <button
                                                key={style}
                                                type="button"
                                                onClick={() => handleChange("responseStyle", style)}
                                                className={cn(
                                                    "rounded-lg border px-3 py-2 text-sm capitalize transition-colors",
                                                    localSettings.responseStyle === style
                                                        ? "border-primary bg-primary/20 text-primary"
                                                        : "border-white/10 bg-background-dark text-gray-400"
                                                )}
                                            >
                                                {style}
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-gray-300">Citation Format</label>
                                    <div className="flex gap-4">
                                        <label className="group flex items-center gap-2">
                                            <input
                                                type="radio"
                                                name="citationFormat"
                                                value="chapter"
                                                checked={localSettings.citationFormat === "chapter"}
                                                onChange={() => handleChange("citationFormat", "chapter")}
                                                className="accent-primary"
                                            />
                                            <span className="text-sm text-gray-400">Chapter/Verse</span>
                                        </label>
                                        <label className="group flex items-center gap-2">
                                            <input
                                                type="radio"
                                                name="citationFormat"
                                                value="section"
                                                checked={localSettings.citationFormat === "section"}
                                                onChange={() => handleChange("citationFormat", "section")}
                                                className="accent-primary"
                                            />
                                            <span className="text-sm text-gray-400">Modern Section</span>
                                        </label>
                                    </div>
                                </div>

                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-gray-300">Theme Mode</label>
                                    <input
                                        type="text"
                                        value="Dark mode is currently fixed"
                                        disabled
                                        className="w-full rounded-lg border border-white/10 bg-background-dark px-3 py-2 text-parchment"
                                    />
                                </div>
                            </fieldset>
                        </div>
                    )}
                </div>

                <div className="flex items-center justify-between border-t border-white/5 bg-sidebar-dark/50 p-6">
                    <button
                        type="button"
                        onClick={handleReset}
                        className="flex items-center gap-2 px-4 py-2 text-sm text-gray-400 transition-colors hover:text-white"
                    >
                        <RotateCcw size={16} />
                        Reset Default
                    </button>
                    <div className="flex gap-3">
                        <button
                            type="button"
                            onClick={onClose}
                            className="rounded-lg px-4 py-2 text-sm text-parchment transition-colors hover:bg-white/5"
                        >
                            Cancel
                        </button>
                        <button
                            type="button"
                            onClick={handleSave}
                            className="flex items-center gap-2 rounded-lg bg-primary px-6 py-2 text-sm font-bold text-background-dark shadow-[0_0_15px_rgba(25,230,212,0.2)] transition-colors hover:bg-primary-dark"
                        >
                            <Save size={16} />
                            Save Configuration
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
