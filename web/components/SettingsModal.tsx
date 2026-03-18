"use client";

import { Settings, DEFAULT_SETTINGS } from "@/lib/types";
import { cn } from "@/lib/utils";
import { X, Save, RotateCcw } from "lucide-react";
import { useState, useEffect } from "react";

interface SettingsModalProps {
    isOpen: boolean;
    onClose: () => void;
    settings: Settings;
    onSave: (settings: Settings) => void;
    onReset: () => void;
}

export function SettingsModal({
    isOpen,
    onClose,
    settings: initialSettings,
    onSave,
    onReset,
}: SettingsModalProps) {
    const [localSettings, setLocalSettings] = useState<Settings>(initialSettings);
    const [activeTab, setActiveTab] = useState<"model" | "retrieval" | "output">("model");

    useEffect(() => {
        // eslint-disable-next-line
        setLocalSettings(initialSettings);
    }, [initialSettings, isOpen]);

    if (!isOpen) return null;

    const handleChange = (key: keyof Settings, value: string | number | boolean) => {
        setLocalSettings((prev) => ({ ...prev, [key]: value }));
    };

    const handleSave = () => {
        onSave(localSettings);
        onClose();
    };

    const handleReset = () => {
        setLocalSettings(DEFAULT_SETTINGS);
        onReset();
        setLocalSettings(DEFAULT_SETTINGS);
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
            <div className="w-full max-w-2xl bg-[#1a2c2a] border border-primary/20 rounded-xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">

                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-white/5 bg-sidebar-dark">
                    <h2 className="text-xl font-serif font-bold text-parchment">Configuration</h2>
                    <button
                        onClick={onClose}
                        className="p-2 text-gray-400 hover:text-parchment hover:bg-white/5 rounded-full transition-colors"
                    >
                        <X size={20} />
                    </button>
                </div>

                {/* Tabs */}
                <div className="flex border-b border-white/5 bg-sidebar-dark/50 px-6">
                    {(["model", "retrieval", "output"] as const).map(tab => (
                        <button
                            key={tab}
                            onClick={() => setActiveTab(tab)}
                            className={cn(
                                "px-4 py-3 text-sm font-medium transition-colors border-b-2",
                                activeTab === tab
                                    ? "border-primary text-primary"
                                    : "border-transparent text-gray-400 hover:text-parchment"
                            )}
                        >
                            {tab === "model" && "Model Parameters"}
                            {tab === "retrieval" && "Retrieval & Knowledge"}
                            {tab === "output" && "Output Preferences"}
                        </button>
                    ))}
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-6 text-parchment space-y-6">

                    {activeTab === "model" && (
                        <div className="space-y-6">
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-gray-300">LLM Provider</label>
                                <select
                                    value={localSettings.llmProvider}
                                    onChange={(e) => handleChange("llmProvider", e.target.value)}
                                    className="w-full px-3 py-2 bg-background-dark border border-white/10 rounded-lg focus:border-primary/50 outline-none text-parchment transition-colors"
                                >
                                    <option value="alibaba">Alibaba Cloud (Qwen)</option>
                                    <option value="openai">OpenAI</option>
                                    <option value="anthropic">Anthropic</option>
                                    <option value="google">Google Gemini</option>
                                    <option value="openrouter">OpenRouter</option>
                                    <option value="together">Together AI</option>
                                </select>
                            </div>

                            <div className="space-y-2">
                                <label className="text-sm font-medium text-gray-300">Model ID (Optional)</label>
                                <input
                                    type="text"
                                    value={localSettings.llmModel}
                                    onChange={(e) => handleChange("llmModel", e.target.value)}
                                    placeholder="e.g. gpt-4o, qwen-max"
                                    className="w-full px-3 py-2 bg-background-dark border border-white/10 rounded-lg focus:border-primary/50 outline-none text-parchment transition-colors placeholder-gray-600"
                                />
                            </div>

                            <div className="space-y-4">
                                <div className="flex justify-between">
                                    <label className="text-sm font-medium text-gray-300">Temperature (Informational)</label>
                                    <span className="text-xs font-mono text-primary">{localSettings.informationalTemperature.toFixed(1)}</span>
                                </div>
                                <input
                                    type="range"
                                    min="0"
                                    max="1"
                                    step="0.1"
                                    value={localSettings.informationalTemperature}
                                    onChange={(e) => handleChange("informationalTemperature", parseFloat(e.target.value))}
                                    className="w-full accent-primary h-1 bg-white/10 rounded-lg appearance-none cursor-pointer"
                                />
                                <div className="flex justify-between text-xs text-gray-500">
                                    <span>Precise</span>
                                    <span>Creative</span>
                                </div>
                            </div>
                        </div>
                    )}

                    {activeTab === "retrieval" && (
                        <div className="space-y-6">
                            <div className="space-y-4">
                                <div className="flex justify-between">
                                    <label className="text-sm font-medium text-gray-300">Retrieval Depth (K)</label>
                                    <span className="text-xs font-mono text-primary">{localSettings.retrievalK} chunks</span>
                                </div>
                                <input
                                    type="range"
                                    min="1"
                                    max="20"
                                    step="1"
                                    value={localSettings.retrievalK}
                                    onChange={(e) => handleChange("retrievalK", parseInt(e.target.value))}
                                    className="w-full accent-primary h-1 bg-white/10 rounded-lg appearance-none cursor-pointer"
                                />
                                <div className="flex justify-between text-xs text-gray-500">
                                    <span>Faster</span>
                                    <span>More Context</span>
                                </div>
                            </div>

                            <div className="flex items-center justify-between p-4 bg-white/5 rounded-lg border border-white/5">
                                <div>
                                    <span className="text-sm font-medium block">Knowledge Graph</span>
                                    <span className="text-xs text-gray-400">Enable hybrid retrieval with KG facts</span>
                                </div>
                                <div className="relative inline-block w-12 mr-2 align-middle select-none transition duration-200 ease-in">
                                    <input
                                        type="checkbox"
                                        checked={localSettings.hybridRetrieval}
                                        onChange={(e) => handleChange("hybridRetrieval", e.target.checked)}
                                        className="toggle-checkbox absolute block w-6 h-6 rounded-full bg-white border-4 appearance-none cursor-pointer peer checked:right-0 right-6"
                                    />
                                    <label className="toggle-label block overflow-hidden h-6 rounded-full bg-gray-700 cursor-pointer peer-checked:bg-primary"></label>
                                </div>
                            </div>

                            {localSettings.hybridRetrieval && (
                                <div className="space-y-4 pl-4 border-l-2 border-primary/20">
                                    <div className="flex justify-between">
                                        <label className="text-sm font-medium text-gray-300">Graph Traversal Depth</label>
                                        <span className="text-xs font-mono text-primary">{localSettings.graphDepth}-hop</span>
                                    </div>
                                    <input
                                        type="range"
                                        min="1"
                                        max="3"
                                        step="1"
                                        value={localSettings.graphDepth}
                                        onChange={(e) => handleChange("graphDepth", parseInt(e.target.value))}
                                        className="w-full accent-primary h-1 bg-white/10 rounded-lg appearance-none cursor-pointer"
                                    />
                                </div>
                            )}
                        </div>
                    )}

                    {activeTab === "output" && (
                        <div className="space-y-6">
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-gray-300">Response Style</label>
                                <div className="grid grid-cols-3 gap-2">
                                    {(["concise", "detailed", "academic"] as const).map(style => (
                                        <button
                                            key={style}
                                            onClick={() => handleChange("responseStyle", style)}
                                            className={cn(
                                                "px-3 py-2 text-sm border rounded-lg transition-colors capitalize",
                                                localSettings.responseStyle === style
                                                    ? "bg-primary/20 border-primary text-primary"
                                                    : "bg-background-dark border-white/10 text-gray-400 hover:border-white/20"
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
                                    <label className="flex items-center gap-2 cursor-pointer group">
                                        <input
                                            type="radio"
                                            name="citationFormat"
                                            value="chapter"
                                            checked={localSettings.citationFormat === "chapter"}
                                            onChange={() => handleChange("citationFormat", "chapter")}
                                            className="accent-primary"
                                        />
                                        <span className="text-sm text-gray-400 group-hover:text-parchment transition-colors">Chapter/Verse</span>
                                    </label>
                                    <label className="flex items-center gap-2 cursor-pointer group">
                                        <input
                                            type="radio"
                                            name="citationFormat"
                                            value="section"
                                            checked={localSettings.citationFormat === "section"}
                                            onChange={() => handleChange("citationFormat", "section")}
                                            className="accent-primary"
                                        />
                                        <span className="text-sm text-gray-400 group-hover:text-parchment transition-colors">Modern Section</span>
                                    </label>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="p-6 border-t border-white/5 bg-sidebar-dark/50 flex justify-between items-center">
                    <button
                        onClick={handleReset}
                        className="flex items-center gap-2 px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors"
                    >
                        <RotateCcw size={16} /> Reset Default
                    </button>
                    <div className="flex gap-3">
                        <button
                            onClick={onClose}
                            className="px-4 py-2 text-sm text-parchment hover:bg-white/5 rounded-lg transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={handleSave}
                            className="flex items-center gap-2 px-6 py-2 bg-primary text-background-dark font-bold text-sm rounded-lg hover:bg-primary-dark transition-colors shadow-[0_0_15px_rgba(25,230,212,0.2)]"
                        >
                            <Save size={16} /> Save Configuration
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
