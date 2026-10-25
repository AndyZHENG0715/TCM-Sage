# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and versions follow [SemVer 2.0.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Public release implementation in progress: six bilingual prerecorded examples with local source/graph navigation, isolated fixed-free BGE/OpenRouter runtime, admission and read quotas, source integrity checks, disabled public Arena writes and lean deployment requirements. Full-corpus index and deployed acceptance are pending.

- Actual free BGE-M3/reranker sample results: 256 semantic vectors and 32 fixed-model requests; positive hit@5 improves from 8/12 lexical to 11/12 dense and 12/12 reranked on the frozen development sample. Recorded independent review of incomplete cross-book labels; full-corpus build and production integration remain pending.

- Isolated free-retrieval validation preparation: fixed SiliconFlow CN BGE adapter, bounded opt-in API runner, 12 positive and 3 unsupported source-grounded cases, immutable sample/result fingerprints and offline network guard. Added 18 passing offline tests and recorded the lexical baseline; actual provider comparison awaits a local key. Existing presentation and corpus preserved.

- Phase 2a deployment plan: code/report/data audit, three hosting options, official pricing/eligibility evidence, reproducible HKD scenarios, proposed live-access controls and a static-first recommendation. HKO dependency released; implementation remains pending a named-option approval.

The earlier entries below describe pre-existing local work and historical context; the phase 2a planning delivery does not include those implementation changes.

- Coordinator recovery and explicit HKO phase 1 dependency hold for deployment planning; distinguished local retrieval from stub generation and documented the frontend/backend serving roles.

- Project-template files: `AGENTS.md` (agent rules, pinned standards, project-specific rules), `HANDOFF.md` (current state), `.editorconfig`, and `.gitattributes` (LF line endings, binary markers).
- Offline demo retrieval in `src/demo_mode.py`: a local character-bigram TF-IDF scorer over `data/processed/chunks.json` plus deterministic hashing embeddings and a local reranker, enabled with `DEMO_MODE=1`.
- `DEMO_LLM=1` fallback that replaces provider models with a deterministic local stub when no API key is available.
- README section pointing to the contributing guide, developer/test docs and the agent/handoff files.

### Changed

- Flattened the repository root: the project now lives directly at the repository root instead of a nested `TCM-Sage/` directory, and `venv/` was rebuilt so its recorded paths are correct.
- Generation switched to Google Gemini `gemini-3.8-flash` via `LLM_PROVIDER=google`; the arena model tiers are overridden with `ARENA_MODELS` to match the provider.
- `.gitignore` gains `.env.*` (with a `!.env.example` exception), `.DS_Store`, `Thumbs.db` and `.uv-cache/`.

### Fixed

- Distinguish formula cross-references from clause numbers in the public prompt; downgrade verification when generated Arabic clause numbers are absent from the supplied source metadata, including a reproduced false-positive verifier case.

- Public BGE index integrity validation now accepts Git line-ending normalization using a declared canonical JSON fingerprint while retaining the original raw SHA and rejecting source-content changes.

- Prevented the arena from calling the Google provider with DashScope model names, which failed with a 404.
