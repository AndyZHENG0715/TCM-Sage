# Codebase Concerns

**Analysis Date:** 2026-04-01

## Prioritized Concern List

**P0 - API contract instability (`GET /config` duplicated):**
- Impact: Runtime settings bootstrap can drift or break between backend and frontend because two handlers exist for the same path, one typed and one partial.
- Files: `src/api.py`, `web/lib/api.ts`
- Near-term mitigation: Keep one `GET /config` implementation returning a single stable schema that matches `RuntimeConfigResponse` in `web/lib/api.ts`.

**P0 - Security baseline not deployment-safe (open CORS + no auth/rate limit):**
- Impact: Any reachable client can invoke model-backed query routes, creating abuse and cost exposure.
- Files: `src/api.py`, `web/app/api/backend/[...path]/route.ts`
- Near-term mitigation: Restrict `ALLOWED_ORIGINS`, add API auth and rate limits at app or reverse-proxy layer, and document deployment profiles (dev vs public).

**P1 - Data quality risk in KG loader/import path (silent drops and heuristic mapping):**
- Impact: Missing or malformed SymMap rows can be silently ignored, producing incomplete retrieval context with no visibility.
- Files: `src/graph_builder.py`, `scripts/import_symmap_kg.py`
- Near-term mitigation: Replace silent `except: pass` flows with counters/warnings, emit import quality metrics, and fail CI on high drop rates.

**P1 - Retrieval performance hot paths are O(N) in user-facing endpoints:**
- Impact: Context views and graph lookup latency grows with corpus and graph size.
- Files: `src/api.py`, `src/graph_builder.py`, `src/ui_backend.py`
- Near-term mitigation: Index chapter chunks by `(book, chapter)` in memory, add bounded search logic for graph name lookup, and instrument endpoint timing.

**P1 - Frontend/backend coupling is manual and brittle:**
- Impact: Response/event format changes can break chat stream parsing without compile-time guarantees.
- Files: `src/api.py`, `web/lib/api.ts`, `web/hooks/useChat.ts`
- Near-term mitigation: Add contract tests for SSE events and config payload; introduce shared schema validation at boundary.

## Tech Debt

**Duplicate route declarations in API module:**
- Issue: `src/api.py` defines `@app.get("/config", response_model=ConfigResponse)` and later defines another `@app.get("/config")` function returning fewer fields.
- Files: `src/api.py`
- Impact: Ambiguous route behavior, unstable OpenAPI output, and confusing maintenance.
- Fix approach: Remove duplicate handler and preserve one typed response contract.

**Retrieval logic duplicated across modules:**
- Issue: Graph fact formatting/search behavior exists in both `src/retriever.py` and `src/ui_backend.py` (`_format_graph_fact`, graph document assembly), increasing drift risk.
- Files: `src/retriever.py`, `src/ui_backend.py`
- Impact: Bug fixes can land in one path and not the other.
- Fix approach: Centralize graph-to-document transformation in one shared utility.

**Dependency hygiene issues in lock-like requirements file:**
- Issue: `langchain-community==0.3.31` is listed twice.
- Files: `requirements.txt`
- Impact: Harder reproducibility audits and noisy environment debugging.
- Fix approach: Deduplicate requirements and validate installs in the project venv only.

**Configuration/documentation drift around prompts and models:**
- Issue: `docs/CONFIG.md` still includes a legacy `SYSTEM_PROMPT` example that asks for a `"Sources:"` section, while `src/main.py` explicitly strips/forbids trailing Sources/References output.
- Files: `docs/CONFIG.md`, `src/main.py`
- Impact: Operators may configure prompts that conflict with UI rendering constraints.
- Fix approach: Align docs with enforced runtime behavior and current default model/provider values.

## Known Bugs

**Abort control in chat hook is not wired to request stream:**
- Symptoms: `AbortController` is created in `web/hooks/useChat.ts` but never passed to `streamQuery`, and `streamQuery` does not accept an abort signal.
- Files: `web/hooks/useChat.ts`, `web/lib/api.ts`
- Trigger: User navigates away or needs to stop a long response; stream keeps running until backend completion/error.
- Workaround: Refresh page or wait for stream completion.

**Legacy E2E script uses mismatched graph source type discriminator:**
- Symptoms: Script treats graph documents as `source_type == "knowledge_graph"` while runtime emits `"graph"`.
- Files: `scripts/e2e_test.py`, `src/retriever.py`, `src/ui_backend.py`
- Trigger: Running `scripts/e2e_test.py` under current hybrid retriever output.
- Workaround: Manual inspection of raw results instead of split counters in that script.

## Security Considerations

**Permissive default CORS policy:**
- Risk: `ALLOWED_ORIGINS` defaults to `*` while credentials are enabled.
- Files: `src/api.py`
- Current mitigation: Environment override exists.
- Recommendations: Set explicit origins per environment and disable credentialed wildcard patterns.

**No authentication/authorization for API endpoints:**
- Risk: Query and arena endpoints can be invoked by any reachable caller.
- Files: `src/api.py`
- Current mitigation: Implicit assumption of trusted/local network.
- Recommendations: Add token-based auth (or gateway auth), per-IP/per-key rate limiting, and request size guards.

**External data ingestion uses HTTP (not HTTPS):**
- Risk: SymMap fetch script pulls from `http://www.symmap.org/...`, susceptible to MITM/tampering on untrusted networks.
- Files: `scripts/fetch_symmap_v2.py`
- Current mitigation: None in script.
- Recommendations: Prefer HTTPS endpoints if available, checksum downloaded artifacts, and store provenance manifest.

## Performance Bottlenecks

**Chunk context endpoint performs repeated full-list scans:**
- Problem: `/source/{chunk_id}/context` filters `load_chunks_data()` in Python for each request and can return large chapter payloads.
- Files: `src/api.py`
- Cause: No precomputed index by chapter/book and always materializing chapter text.
- Improvement path: Build memoized chapter index map, paginate or cap full-text payload, and cache per-chapter reconstruction.

**Graph entity lookup is linear over all nodes per query:**
- Problem: `search_by_name` iterates every graph node and checks multiple string variants.
- Files: `src/graph_builder.py`
- Cause: No secondary index for names/aliases.
- Improvement path: Build normalized name index at load time and keep fallback fuzzy/substring matching bounded.

**Per-request model/runtime construction adds latency:**
- Problem: Query paths repeatedly construct runtime model chain components.
- Files: `src/ui_backend.py`, `src/main.py`
- Cause: Runtime model creation in request path, especially with multiple provider models.
- Improvement path: Cache provider/model clients keyed by runtime settings where safe.

## Fragile Areas

**Graph ingestion swallows malformed rows silently:**
- Files: `src/graph_builder.py`
- Why fragile: `except Exception: pass` in entity and relationship loads hides parser regressions and source data drift.
- Safe modification: Replace silent skipping with structured warnings and counters.
- Test coverage: Minimal; no regression test asserts acceptable skip/error thresholds.

**SymMap migration fallback can mask incorrect graph wiring:**
- Files: `src/ui_backend.py`, `src/config.py`
- Why fragile: Runtime falls back from SymMap path to legacy graph files when expected file is absent.
- Safe modification: Make fallback explicit per environment and emit startup warnings with resolved graph path.
- Test coverage: `scripts/verify_symmap_retrieval.py` validates one query path only.

## Scaling Limits

**Single-process local storage architecture:**
- Current capacity: Chroma at `vectorstore/chroma` and local JSON graph files under `data/graph/`.
- Limit: Horizontal scaling and multi-tenant usage are constrained by local disk + process memory.
- Scaling path: External vector DB/object storage and a dedicated graph service or precomputed graph index.

## Dependencies at Risk

**Fast-moving AI stack version coupling:**
- Risk: Tight and broad pin set (`langchain*`, `chromadb`, `torch`, `transformers`) can create upgrade conflicts.
- Impact: Longer stabilization cycles and breakage risk on dependency refresh.
- Migration plan: Introduce constraints workflow, periodic compatibility matrix checks, and smoke tests on fresh venv creation.

## Missing Critical Features

**Readiness checks for deployment correctness:**
- Problem: `/health` only returns process liveness, not vectorstore/KG/read-model readiness.
- Blocks: Reliable orchestration and auto-restart decisioning in production-like deployments.

**Versioned API contract governance:**
- Problem: No versioning or schema contract gate for frontend/backend changes.
- Blocks: Safe iterative rollout across UI and API independently.

## Test Coverage Gaps

**No automated frontend unit/integration tests:**
- What's not tested: SSE parsing, settings bootstrap mapping, citation panel interactions, and source page flows.
- Files: `web/package.json`, `web/lib/api.ts`, `web/hooks/useChat.ts`, `web/components/CitationPanel.tsx`, `web/app/source/[chunkId]/page.tsx`
- Risk: Regressions surface only in manual QA.
- Priority: High.

**No backend API contract tests for critical routes:**
- What's not tested: `/query` event semantics, `/config` schema stability, and `/source/{chunk_id}/context` response shape.
- Files: `src/api.py`, `web/lib/api.ts`
- Risk: Silent client breakage after backend changes.
- Priority: High.

**KG migration verification is narrow:**
- What's not tested: Multi-entity query coverage, edge-direction correctness across all relationship tables, and import-loss thresholds.
- Files: `scripts/verify_symmap_retrieval.py`, `scripts/import_symmap_kg.py`, `scripts/fetch_symmap_v2.py`
- Risk: Data migration appears successful while preserving hidden semantic errors.
- Priority: High.

---

*Concerns audit: 2026-04-01*
