"use client";

import { useEffect, useState } from "react";
import { ARENA_MODEL_PRESETS, ARENA_SAMPLE_PROMPTS } from "@/lib/arenaPrompts";

export default function ArenaPage() {
  const [sessionId, setSessionId] = useState("");
  const [selectedModel, setSelectedModel] = useState("qwen-plus");
  const [inputValue, setInputValue] = useState("");
  const [showVoting, setShowVoting] = useState(false); // eslint-disable-line @typescript-eslint/no-unused-vars
  const [showReveal, setShowReveal] = useState(false); // eslint-disable-line @typescript-eslint/no-unused-vars

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSessionId(crypto.randomUUID());
  }, []);

  return (
    <div className="min-h-screen bg-[#1a1a2e] text-[#F3EFE0] flex flex-col">
      {/* Top bar */}
      <header className="flex items-center gap-4 px-6 py-3 border-b border-gray-700 bg-[#0d0d1a]">
        <h1 className="text-lg font-semibold text-[#19e6d4] shrink-0">TCM Arena</h1>
        <div className="flex gap-2 flex-wrap flex-1">
          {ARENA_MODEL_PRESETS.map((preset) => (
            <button
              key={preset.value}
              type="button"
              title={preset.description}
              onClick={() => setSelectedModel(preset.value)}
              className={`px-3 py-1 rounded-full text-sm font-medium border transition-colors ${
                selectedModel === preset.value
                  ? "border-[#19e6d4] text-[#19e6d4] bg-[#19e6d4]/10"
                  : "border-gray-600 text-gray-400 hover:border-gray-400 hover:text-gray-200"
              }`}
            >
              {preset.label}
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={() => setSessionId(crypto.randomUUID())}
          className="shrink-0 px-3 py-1 text-sm border border-gray-600 rounded-lg text-gray-300 hover:border-gray-400 hover:text-white transition-colors"
        >
          New Session
        </button>
      </header>

      {/* Panels */}
      <main className="flex flex-1 flex-col md:flex-row gap-4 p-4 min-h-0">
        {/* Panel A */}
        <div
          data-testid="arena-panel-a"
          className="flex-1 flex flex-col bg-[#0d0d1a] rounded-lg border border-gray-700 overflow-hidden"
        >
          <div className="px-4 py-2 border-b border-gray-700 bg-[#0a0a17]">
            <h2 className="text-sm font-semibold text-gray-300">Model A</h2>
          </div>
          <div className="flex-1 overflow-y-auto p-4">
            <p className="text-gray-500 text-sm italic">Awaiting response…</p>
          </div>
        </div>

        {/* Panel B */}
        <div
          data-testid="arena-panel-b"
          className="flex-1 flex flex-col bg-[#0d0d1a] rounded-lg border border-gray-700 overflow-hidden"
        >
          <div className="px-4 py-2 border-b border-gray-700 bg-[#0a0a17]">
            <h2 className="text-sm font-semibold text-gray-300">Model B</h2>
          </div>
          <div className="flex-1 overflow-y-auto p-4">
            <p className="text-gray-500 text-sm italic">Awaiting response…</p>
          </div>
        </div>
      </main>

      {/* Vote bar placeholder */}
      {showVoting && (
        <div className="px-6 py-3 border-t border-gray-700 bg-[#0d0d1a] text-center text-sm text-gray-400">
          Voting placeholder
        </div>
      )}

      {/* Input area */}
      <footer className="px-4 py-3 border-t border-gray-700 bg-[#0d0d1a] space-y-2">
        {/* Preset chips */}
        <div className="flex gap-2 flex-wrap">
          {ARENA_SAMPLE_PROMPTS.slice(0, 5).map((prompt) => (
            <button
              key={prompt}
              type="button"
              onClick={() => setInputValue(prompt)}
              className="text-xs px-2 py-1 rounded-full bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white transition-colors"
            >
              {prompt.length > 20 ? prompt.slice(0, 20) + "…" : prompt}
            </button>
          ))}
        </div>

        {/* Text input + controls */}
        <div className="flex gap-2">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="向两个模型提问…"
            className="flex-1 bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm text-[#F3EFE0] placeholder-gray-500 focus:outline-none focus:border-[#19e6d4]"
          />
          <button
            type="button"
            disabled={!inputValue.trim()}
            className="px-4 py-2 bg-[#19e6d4] text-[#0d0d1a] font-semibold text-sm rounded-lg disabled:opacity-40 disabled:cursor-not-allowed hover:bg-[#14c9b8] transition-colors"
          >
            提交
          </button>
          <button
            type="button"
            disabled
            className="px-4 py-2 border border-gray-600 text-gray-400 text-sm rounded-lg disabled:opacity-40 disabled:cursor-not-allowed"
          >
            End &amp; Reveal
          </button>
        </div>

        {/* Session debug (hidden in prod) */}
        <p className="text-xs text-gray-600 hidden">Session: {sessionId}</p>
      </footer>

      {/* Suppress unused state lint warnings */}
      {showReveal && null}
    </div>
  );
}
