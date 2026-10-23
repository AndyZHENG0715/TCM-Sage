# Handoff

Current state only. Overwrite stale content instead of appending; history lives in git and CHANGELOG.md.

Checked at: `19e4f654855807b1df339e7fb75fcdc386d53b2a` on 2026-10-23 (Asia/Taipei), also origin/main after fetch. Existing presentation changes preserved: `.gitignore`, `README.md`, `src/embeddings.py`, `src/main.py`, `src/ui_backend.py`; untracked `.editorconfig`, `.gitattributes`, `src/demo_mode.py`. Phase 2a changes only `docs/DEPLOYMENT_PLAN.md`, HANDOFF.md and CHANGELOG.md on `codex/deployment-plan-2a`; prior local documentation context is retained. No presentation implementation is included in the planning commit.

## Coordination gate

- HKO phase 1 hold RELEASED on 2026-10-23 by an explicit coordinator instruction reporting Andy's completion and authorization. Phase 2a audit and plan are prepared in `docs/DEPLOYMENT_PLAN.md`; phase 2b remains HELD until Andy approves a named option.
- Coordinator chat: `01a167b5-4710-7191-8f97-2c3b7ed93048`. Canonical root remains `E:/Projects/tcm-sage`; stay on Windows. Preserve existing presentation work.
- Current scope: planning and draft PR only; no phase 2b implementation, provisioning, deployment, billing/card setup or paid provider/embedding calls. Free reversible branch, documentation edits, checks and draft PR delivery were explicitly authorized; no renewed blanket commit approval is required for that scope.
- Code audit confirms Next.js is the HTTP-serving frontend with server-side proxy routes on port 3000; FastAPI/Uvicorn is the HTTP-serving Python backend on port 8000. ChromaDB and NetworkX are in-process storage/retrieval components. Static replay needs an approved later export/adapter change.
- `DEMO_MODE` controls local retrieval; `DEMO_LLM` independently selects stub generation. The prior Gemini presentation is not evidence of a zero-cost prerecorded public mode.
- Arena's baseline bypasses the model factory, hardcodes DashScope and performs DDGS search. Even both flags do not establish full-app offline operation. Public Arena query/vote must stay disabled in the proposed initial release.

## Deployment planning outcome

- Recommend Option A: Cloudflare Pages prerecorded static showcase, no backend/vectorstore/provider key; estimated HK$0 at zero traffic or 500 replay sessions within official static limits. No-card offering verified on the official product page.
- Option B: static frontend plus Render Free optional local-retrieval live preview, conditional model cost HK$17.20 (Qwen) or HK$52.65 (Gemini) at 500 capped queries; memory fit and restart-safe allowance remain gates.
- Option C: static frontend plus Render paid 2 GB backend and assumed 1 GB disk: HK$196.95 idle; HK$222.50–257.95 at 500 vector-mode queries under declared token/model assumptions. Hosting overages, tax/FX and one-time vector recovery excluded; no unconditional bill cap verified.
- Provider eligibility is unresolved: Gemini's official supported-region list does not list Hong Kong. No residence/account eligibility is inferred. A proposed US$10/month API ceiling and server controls are future acceptance conditions, not existing protections.
- Corpus audit confirms 12,204 chunks, 388 Shanghan clause records, 489 Jingui clause records; 18,450 graph entities / 21,476 edges. Vectorstore absent; 12,592 checkpoint IDs must not be mistaken for recoverable vectors. No ingestion performed.
- Unresolved source conflicts: README 398 versus report/artifact 388; report April 3–6 versus its April 7 row. Graph-library audit found matching NetworkX backend descriptions, and React Flow mini-view versus Cytoscape explorer rather than a chapter-3 contradiction. Do not silently rewrite the source documents.

## Goals

Current goals, highest priority first. The long-term vision lives in README.md.

1. Ship a demo- and presentation-ready build of TCM-Sage: the Next.js UI and FastAPI backend answering TCM questions with verifiable citations from the 17-text classical corpus.
2. Restore real vector retrieval: build `vectorstore/chroma` with DashScope `text-embedding-v4` so `DEMO_MODE=0` works.

## State

- The repository root is `E:\Projects\tcm-sage` (it was nested one level deeper in `TCM-Sage/` until 2026-10-22).
- Backend runs with `venv\Scripts\python.exe src\api.py` on `http://127.0.0.1:8000`; frontend runs with `cd web && npm run dev` on `http://localhost:3000` (Next.js 16.1.6, Turbopack) and proxies `/api/backend/*` to the API.
- Python: `venv/` is a uv-built CPython 3.13.15 environment with all 169 pinned `requirements.txt` packages installed; the uv cache lives in `.uv-cache/` (gitignored).
- Retrieval currently runs in demo mode: `DEMO_MODE=1` scores `data/processed/chunks.json` (12,204 chunks from the 17 texts) with a local character-bigram TF-IDF scorer. `vectorstore/chroma` does not exist, so real ChromaDB retrieval and `src/ingest.py` have not been exercised recently.
- Generation uses Google Gemini `gemini-3.8-flash` (`LLM_PROVIDER=google`, classifier and verifier the same model) via `GOOGLE_API_KEY` in `.env`. Arena tiers are overridden with `ARENA_MODELS` in `.env`; the built-in defaults are still DashScope model names and would 404 under the Google provider.
- SymMap KG loads from `data/graph/symmap/symmap_entities.json`; hybrid retrieval reports `hybrid_available: true`.
- Known behavior: the verifier returns `UNSUPPORTED` when Gemini adds knowledge beyond the retrieved context (the UI badge reads 可能包含未支援的論述), and KG citations are sometimes off-topic (a 桂枝汤 question returned 天山雪莲 graph facts).

## Verification

Phase 2a: read-only code, report and JSON audit; official hosting/model/region/lifecycle evidence checked 2026-10-23; cost formulas reconciled across three LLM calls plus optional embedding/rerank. Project venv used only for read-only data counts and preservation hashes. No retrieval mode executed, no app imports or paid requests. Static build, Linux dependency install, memory/latency, provider access and live controls remain unverified. Existing October runtime results below remain historical reports, not re-tested evidence.

Recovery on 2026-10-23 (Asia/Taipei): read local AGENTS.md, README.md, HANDOFF.md and CHANGELOG.md; inspected git status, five latest commits and tracked working diffs. Fetched origin successfully; `HEAD..origin/main` has no later commits. No secrets opened. No application tests or retrieval modes run this turn. The checks below are inherited reports, not independently rerun; current configuration, vectorstore absence and runtime remain unverified.

Checks run on 2026-10-22, all in `DEMO_MODE=1` with Gemini:

- `venv\Scripts\python.exe -c "import api"` from the repo root: the FastAPI app imports (`TCM-Sage API 1.0.0`).
- Demo store: `_collection.count()` → 12,204; `similarity_search("桂枝汤", k=2)` returns corpus hits.
- Backend `GET /health` → `{"status":"ok"}`; `GET /config` → `provider: google`, `model: gemini-3.8-flash`.
- `POST /query` (SSE) → streamed Chinese answer with inline `[N]` citations and a metadata event carrying 10 citations (text + graph), `citation_bounds.is_valid: true`.
- `POST /arena/query` (SSE) → 84 data events including metadata; `GET /arena/stats` → 64 historical votes with t-test output.
- Browser check of `http://localhost:3000`: a question typed into the composer streamed a Markdown answer (tables, headings), rendered the 來源 panel with 10 sources, and showed the severity badge 知識性問題.
- Frontend after the root move: `npm run dev` served `http://localhost:3000` with HTTP 200 and title `TCM-Sage`.

Not verified: real ChromaDB retrieval (`DEMO_MODE=0`), `src/ingest.py`, `npm run build`, and the `src/test_*.py` / `scripts/` check scripts.

## Recent changes

- Applied the project-template bootstrap: added `AGENTS.md`, `HANDOFF.md`, `CHANGELOG.md`, `.editorconfig`, `.gitattributes`; merged secret/OS ignore rules into `.gitignore`; linked the agent and developer docs from `README.md`.
- Flattened the repository root, moving everything out of `TCM-Sage/` into `E:\Projects\tcm-sage` and rebuilding `venv/` so its launcher and `activate` paths are correct.
- Added the offline demo layer `src/demo_mode.py` (`DEMO_MODE` local retriever and embeddings, `DEMO_LLM` stub model, local reranker) and wired it into `src/embeddings.py`, `src/main.py` and `src/ui_backend.py` behind environment flags.
- Switched generation to Google Gemini `gemini-3.8-flash` and pointed `ARENA_MODELS` at it, after the DashScope key turned out to be unavailable.

## Next steps

- Await Andy's approval naming A, B or C. Spec ahead: the deployment plan proposes static replay and access/budget controls; code pending and phase 2b held. The older backlog below is deferred and does not authorize ingestion or implementation.

- Build the real vectorstore when a DashScope key is available: set `DEMO_MODE=0`, run `venv\Scripts\python.exe src\ingest.py`, then re-run the `/query` check against ChromaDB.
- Decide whether the demo layer stays in the repository or is only enabled during presentations; if it stays, document it under `docs/`.
- Investigate the verifier's `UNSUPPORTED` verdicts on otherwise well-cited answers (stricter prompt or a stronger verifier model).
- Run `npm run build` and the Python check scripts before the next release.

## Open questions and blockers

- No DashScope API key, so embeddings, reranking via `qwen3-rerank`, and ingestion are unavailable; only the local feature-hashing path is exercised.
- Existing template/presentation implementation remains uncommitted and outside the planning delivery. The phase 2a authorization permits the separate documentation commit and draft PR; it does not authorize merging, deployment or committing unrelated implementation.
- The one-time Bootstrap section in `AGENTS.md` (between the BOOTSTRAP markers) is intentionally retained: the user has not approved deleting it yet, so remove it, markers included, once the project-specific rules are signed off.
- Whether `.uv-cache/` should be kept, ignored (current state), or deleted after installs.
