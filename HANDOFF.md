# Handoff

Checked at: `fe22c7f` on 2026-10-24 (Asia/Taipei), plus the uncommitted work below. Fetched origin this continuation: origin/main remains `19e4f65`. Branch: `codex/free-retrieval-validation`. Canonical workspace: `E:/Projects/tcm-sage` on Windows.

## Authorization and current goal

Andy explicitly authorized autonomous implementation and deployment until TCM-Sage is fully online, superseding prior phase/sample/hosting holds. Use Vercel Hobby frontend and Render Free backend; no paid upgrade/card setup. Both accounts are logged in. SiliconFlow and OpenRouter keys are configured locally; never print or commit them. See `docs/RELEASE_EXECUTION.md` for release acceptance and ownership.

## Current state

- Full real BGE index build COMPLETE under retrieval worker: version `vectorstore/bge-m3/20261024T015651Z-223a0f34`, 12,204/12,204 verified records; session1679 exited0. Manifest complete. Builder preserves corpus and reuses 256 validated sample vectors. Portable exports packaged in deploy/bge-m3. Offline native restore passed all12,204 records and19 tests; Linux execution remains unverified.
- Fixed free models: SiliconFlow `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3`; OpenRouter `nvidia/nemotron-3-ultra-550b-a55b:free`. Account endpoint confirmed 1,000 free-model requests/day. Gemma free endpoints returned 429; no paid fallback. A small Nemotron classifier probe succeeded; full generated-answer quality remains unverified.
- New isolated backend: `src/release_api.py`, `release_runtime.py`, `release_limits.py`. Access-code gate, SQLite quotas, bounded requests, fixed providers, sanitized failures, source/graph reads, disabled Arena. SQLite only survives when its filesystem survives; Render ephemeral replacement is a remaining durable-quota gate before enabling live access.
- New frontend defaults to six labelled prerecorded bilingual examples with corpus/graph fixtures and local source navigation; English default. Live code held only in session memory. Historical presentation remains opt-in via NEXT_PUBLIC_PRESENTATION_MODE=1. No public Arena writes/raw vote exports.
- Frontend worker owns final build and port3001 preview. Audit worker owns full-index independent evaluation runner and tests. Retrieval worker owns full build and Linux restore. Root owns integration/deployment/docs. Shared browser operations are root-only.
- No new deployment, implementation commit or push yet. Planning draft PR #1 remains https://github.com/AndyZHENG0715/TCM-Sage/pull/1 .

## Retrieval evidence

Frozen 256-record actual sample: 32 fixed-free calls, 1,024 dimensions. Hit@5 lexical8/12, dense11/12, reranked12/12; required-source coverage .625/.875/.958333. These enriched sample metrics are not corpus or clinical accuracy. Strict cross-book label miss has a relevant unlabelled Qianjin alternative; preserve old labels and metrics. Independent new fixture frozen before full evaluation: `scripts/fixtures/retrieval_independent_cases.json`, SHA256 `25893dad77f99dd77ae01ba09266f7ed08f9945caca543c988483cd31f9d3e08`. Evaluate unfiltered and metadata-filtered lanes separately. All three unsupported controls retrieve neighbours; no abstention threshold validated. See retrieval validation/review documents.

## Verification

Current release tests reported by audit worker: runtime21PASS, API13PASS, limits16PASS. Runtime includes rejection of malformed classifiers, truncated/empty nonstream responses and empty DONE streams. All are mocked/no provider calls. Frontend lint and four fixture/replay tests PASS; final production rebuild pending the last graph-alias fixture change. Earlier production build passed compile/typecheck/prerender. Full corpus evaluation completed: lexical9/12, dense11/12, rerank12/12 hit@5; strict source coverage .958333 (one required passage ranks6). Two tiny generated-answer/out-of-scope smokes passed; browser390px DOM/source checks passed. Memory/Linux deployment and deployed checks remain pending. See docs/RELEASE_VERIFICATION.md. Tested retrieval mode is isolated BGE sample and in-progress full BGE build, not legacy DEMO_MODE=0/DashScope.

## Preserve

Original unrelated presentation changes: .gitignore, README.md, src/embeddings.py, src/main.py, src/ui_backend.py; untracked .editorconfig, .gitattributes, src/demo_mode.py. Do not revert or silently include them in release commits. Corpus and graph are read-only. Legacy vectorstore/chroma absent; legacy12592-ID checkpoint is stale and must not be reused/reset. Current corpus12204 chunks/17 books; graph18450 entities/21476 edges.

## Next

Finish full index and portable restore; independent evaluation and tiny live query; measure memory and package lean Linux dependencies. Verify replay/source/graph/mobile. Resolve durable quota storage before live enablement. Publish coherent reviewed changes, provision free hosts and environment, verify deployed HTTPS paths, no-code denial and off switches. Genuine provider authentication/terms or sensitive key transmission may require a specific human action; prepare concrete service configuration first.
