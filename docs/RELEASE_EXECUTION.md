# Public release execution

Authorized by Andy on 2026-10-24 (Asia/Taipei): continue autonomously until TCM-Sage is fully online. This supersedes the earlier phase 2a/hosting-choice and sample-build holds. Preserve historical results; resolve defects through review and further tests, without changing old scores. Account login, verification and any spending remain human actions; target recurring cash cost is zero.

## Goal and definition of done

A public HTTPS frontend presents six to eight labelled prerecorded source-grounded examples without a backend dependency. Optional access-code-protected live questions reach the full corpus index with citations, source context and graph navigation. English is the clean-session default with Chinese available. Safety notices describe a fallible research prototype. Arena writes are disabled. Backend access, rate/concurrency/global allowances, model allowlists and emergency disable behavior are enforced before provider calls.

Done requires local and deployed checks for replay, live streaming, citations, graph, 390 px layout, no-code denial, limits and fallback. Record actual URLs, selected free models, measured startup/resource limitations and operational off switches. Do not claim unsupported clinical accuracy or guarantee free-provider availability.

## Chosen implementation route

- Vercel Hobby serves the existing Next.js frontend and server proxy. Render Free is the initial FastAPI backend candidate, subject to dependency and 512 MB memory fit checks. Both accounts are logged in per Andy; verify the actual signed-in identities at deployment. No paid upgrade or card setup is authorized by the zero-cost implementation choice.
- SiliconFlow CN exact non-Pro `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3` serve retrieval. Andy reports account L0 limits of 2,000 RPM and 500,000 TPM for both. Treat these as user-observed limits; run well below them with bounded batches, conservative token reservations and handling of 429 responses. This is not permission to consume all shared capacity.
- Build a separate versioned corpus index; do not run the legacy ingestion script, rewrite chunks, or reuse its stale checkpoint. Reuse only validated sample vectors that match source IDs/content/corpus/model/preprocessing. Verify collection count and provenance before switching runtime paths.
- Generation uses a verified free model only. Andy configured an OpenRouter key; account inspection confirmed 1,000 free-model requests/day. The fixed candidate is nvidia/nemotron-3-ultra-550b-a55b:free, pending full generated-answer checks. Gemma free routes returned 429; there is no paid fallback. No arbitrary provider/model controls are exposed publicly; no paid fallback. Monthly paid inference ceiling is US$0 for this release, enforced by fixed free-only allowlists and fail-closed behavior. Free quotas and host suspension remain availability limits.
- Public reads of already-public corpus excerpts and graph facts are bounded/rate-limited. Live query requires a backend-held access-code hash, HTTPS, quotas and limited input/context/output. Preserve the presentation interface behind an explicit opt-in flag. No raw historical vote/comment/session export.

## Sequence and ownership

1. Evidence worker freezes new independent questions before viewing new results and records the prior incomplete-label review, preserving v2 metrics.
2. Retrieval worker tests a fixed-model client, throttled full-corpus builder and separate index manifests/checkpoints. Coordinator reviews before executing the build.
3. Frontend worker implements independent replay, local citation/graph fixtures, English default and public-mode restrictions, preserving the original presentation path.
4. Coordinator integrates backend provider/index settings, grounded public prompt, failure-aware verification, access gate and quotas; tests locally with mocked providers before tiny free live smoke tests. Measure backend process memory and reduce deployment-only requirements to already-used packages where needed.
5. Package public index artifacts with manifests, no secrets; publish coherent commits/draft PRs, then configure only free hosting through authenticated accounts. The user's explicit deployment authorization covers publishing the completed result; an additional blanket permission step is unnecessary.
6. Verify deployed behavior and update README/HANDOFF with live URLs, cost assumptions and shutdown instructions. Do not stop merely because a build or independent test fails; correct and recheck within the authorized free scope.

## Known risks to resolve

The 256-record probe has incomplete manual relevance labels, not a held-out accuracy benchmark. Independent source review found genuine second-book coverage despite one strict label miss; fresh independent and full-corpus checks are required. Unknown-query nearest neighbours are not evidence of answerability. Free Render sleeps after inactivity; replay must continue while it wakes. Persistent public quotas cannot rely on an automatically replenished in-memory ledger after restarts; either reserve a conservative process allowance from an externally tracked total and require manual rearming, or add a verified free durable ledger before enabling unrestricted live access.
