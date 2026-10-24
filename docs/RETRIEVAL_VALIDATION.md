# Free retrieval validation

Status: actual free-API sample comparison completed on 2026-10-24 (Asia/Taipei) after Andy configured the local key and released the offline-only constraint. Acceptance still needs review for one multi-source coverage gap. This supersedes the planning-only hold for this experiment alone. Deployment, account/billing setup, paid API use, frontend redesign and production retrieval replacement remain outside scope.

## Goal and acceptance

Establish whether free BGE semantic retrieval and reranking improve retrieval of the existing classical corpus enough to justify a complete new index. Completion means a reproducible comparison on source-grounded questions, with results and limitations recorded. Network adapter tests alone do not establish model quality.

Use `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3` through SiliconFlow CN only. Their exact non-Pro IDs are listed as free on the [official pricing page](https://www.siliconflow.cn/pricing) (checked 2026-10-23); confirm before actual calls. Do not fall back to Pro models, alternative endpoints or paid providers. No LLM generation is required for this experiment.

## Work split

- Coordinator: experiment runner, data preservation, comparison, acceptance and handoff.
- Provider worker: a standalone fixed-model adapter and mocked contract/failure tests, without production wiring.
- Evidence worker: 12 positive retrieval cases and 3 negative controls, exact corpus IDs/quotes, and documented manual relevance judgements.
- Independent review: inspect runner and adapter after integration, including secret redaction and interpretation of negative controls.

## Method

1. Read `data/processed/chunks.json`; verify IDs and quoted anchors. Preserve the file, original books, graph and ingestion checkpoint byte-for-byte. Do not run `src/ingest.py` or trust the old checkpoint as a vector index.
2. Select a deterministic 256-chunk sample containing expected positive passages plus seeded distractors. Record its IDs and source hash before API use. This is a development probe with known positives included, not an unbiased corpus-wide benchmark.
3. Run the existing `DemoVectorStore` character-bigram baseline against this same sample. Do not call a generic model or substitute hashing vectors for semantic embeddings.
4. After `SILICONFLOW_API_KEY` is configured locally, embed sample documents and all questions with the fixed free BGE-M3 model. Persist a separate model-labelled experiment index under ignored `vectorstore/retrieval-validation/`, never `vectorstore/chroma`. Rank by cosine similarity; rerank only a bounded top-20 set with the fixed free reranker.
5. Compare positive-case hit@5 and reciprocal rank across lexical, dense and dense-plus-rerank outputs. Preserve every top candidate ID/score. Report negative controls separately: retrieval always returns neighbours and is not an abstention mechanism. Labels are manually source-grounded and incomplete, not expert or clinical validation. No arbitrary “accuracy” claim.
6. Proposed go/no-go: both semantic stages must return finite, dimension-consistent vectors/scores with valid citation IDs; dense and reranked hit@5 should each be at least 10/12 with match-aware required-source coverage at least 0.75. Require the exact clause target in the top five and flag any missing required source on multi-source cases. Reranking should not lower aggregate hit@5 or reciprocal rank relative to dense. Report regressions individually. If thresholds fail or runtime/API access cannot be checked, do not build the full index automatically. This is an engineering checkpoint, not proof of superiority.
7. Only after reviewing successful sample evidence, prepare the full 12,204-chunk build in a separate versioned index without changing corpus texts or activating production retrieval. The user's approval covers the proposed free validation-to-index sequence; external deployment and paid calls still require separate authorization. A complete index must use the same model, dimensions and preprocessing at query time.

## Credentials and execution boundary

Use the existing Windows project venv for all Python. No dependency installs are planned. Credentials are read into the request process from local environment configuration and never printed, saved in results or committed. Output files contain public-corpus evidence and aggregate operational metadata only. The initial check found no SiliconFlow, Cloudflare or OpenRouter credential under the expected names; this is not proof that Andy has no provider account.

The runner defaults to offline preparation. Live API work requires an explicit mode, local key, fixed free IDs and a strict request budget; a 401, 429, malformed response or network failure stops the experiment without retries or model substitution. Cached model vectors may only be reused when corpus/sample/model/preprocessing fingerprints match. This standalone experiment does not import the API, main generation pipeline, Arena or `.env`-dependent production factories.

## Commands and results

Offline preparation from the project root:

```powershell
venv\Scripts\python.exe scripts\test_retrieval_validation_provider.py
venv\Scripts\python.exe scripts\test_free_retrieval_validation.py
venv\Scripts\python.exe scripts\validate_free_retrieval.py
```

The runner blocks Python socket connections and DNS in offline mode. It snapshots the cases, selected original chunks and corpus/cases/sample/scorer/runner hashes into a fresh run directory. It never overwrites a prior run or the production vectorstore. Vector reuse/resume is not implemented; each future live run is a fresh bounded experiment.

On 2026-10-24, **10 provider contract tests and 8 runner tests passed**; independent review passed for offline scope. The initial runner-test failure was a sandbox temporary-directory permission error; the test now uses an explicitly workspace-confined ignored directory, and passed. No dependencies were added.

The version 2 lexical baseline on 256 chunks: **8/12 hit@5**, 2/12 hit@1, MRR@20 **0.396795**, match-aware required-source coverage@5 **0.625**. These manually labelled sample results are not corpus accuracy. [The checked-in result summary](RETRIEVAL_VALIDATION_RESULTS.json) records every case, expected source rank and top-five candidates. Full immutable artifacts are at `vectorstore/retrieval-validation/20261024T012638Z-2972d59a/report.json`; generated vectors and artifacts are gitignored. Earlier development runs are retained.

All three unsupported controls returned passages; the retriever has no validated abstention mechanism. The pure-letter control in fixture version 2 has no corpus bigram overlap, yet gets the documented fallback. The test-construction change from version 1 is recorded in [case provenance](RETRIEVAL_VALIDATION_CASES.md); positive questions and labels were not tuned after observing baseline results.

Preservation hashes matched across **49 evidence and existing presentation/config files**. Tested mode was isolated local `DemoVectorStore` lexical retrieval, corresponding to the `DEMO_MODE=1` retrieval implementation; no environment flags were changed, application routes exercised, graph retrieval invoked or semantic vectors built.

## Actual free API comparison

Andy configured `SILICONFLOW_API_KEY` locally and authorized continuation on 2026-10-24. Presence was checked without exposing the secret; the official [CN price page](https://siliconflow.cn/pricing) still listed both exact non-Pro IDs as free. The `www` URL timed out during recheck; the official non-www price listing was accessible. No paid model substitutions were used.

The opt-in command completed **32 requests** against only the pinned free embedding/reranking models. It produced **256 real BGE-M3 vectors, each 1,024 dimensions**, in `vectorstore/retrieval-validation/20261024T014413Z-ecba9244/semantic_index.json`. These are actual semantic model vectors, not the demo hashing stand-in; the production Chroma index remains absent and unchanged.

Same cases, labels, sample and scorer as the offline baseline:

- Lexical: 8/12 hit@5, 2/12 hit@1, MRR@20 0.396795, required-source coverage@5 0.625.
- Dense BGE-M3: 11/12 hit@5, 6/12 hit@1, MRR@20 0.697222, required-source coverage@5 0.875.
- Dense plus BGE reranker: 12/12 hit@5, 9/12 hit@1, MRR@20 0.875000, required-source coverage@5 0.958333.

**Gate: needs review**, despite passing the aggregate hit/coverage thresholds and improving MRR. For `cross-book-xiaoqinglong`, the labelled Qianjin chunk 1548 ranks 6th in dense results and 8th after reranking. Qianjin chunk 850 does appear third in both lists, so the result is not simply missing that book: relevance of this unlabelled alternative needs independent inspection. Keep the original labels and report; do not retroactively turn an unjudged hit into a benchmark pass. Exact clause 13 improves to rank 5 dense / rank 1 reranked. All three unsupported controls still return neighbours; no abstention threshold is established.

Independent source review confirms that chunk 850 is relevant: Qianjin's volume 9 passage explicitly discusses Xiaoqinglong and the requested water-context wording, while the frozen expected chunk 1548 is a different volume 18 passage. Thus actual two-book coverage exists in this example; the strict labelled-source miss exposes incomplete annotations. Scores and labels remain unchanged. Other top-five results include a book outside the question's scope and a Daqinglong passage, so 12/12 any-hit does not mean every returned passage is relevant. Independent checks found 256 unique sample IDs, no exact duplicate content, and matching sample/corpus hashes. Case snapshots preserve the fixture's values but serialize line endings differently; the actual-result summary records a separate `cases_snapshot_sha256`.

See [actual result summary](RETRIEVAL_LIVE_RESULTS.json) for all case rankings and acceptance checks; full report and exact case/sample snapshots remain under the same ignored run directory. Preservation checks again matched **49 evidence and presentation/config files**. No corpus/chunk/checkpoint changes, dependency installs, LLM generation, graph calls, production wiring, account changes or deployment occurred. These enriched-sample labels establish neither general corpus accuracy nor clinical reliability. Formal latency/resource profiling and provider billing-ledger inspection were not performed.

## Next step after sample review

Inspect the cross-book passage alternatives and retrieval coverage before choosing whether to test book-aware selection on the same frozen candidates or broaden to an independent full-corpus evaluation. Do not build the full index automatically while the recorded gate remains `needs_review`. Any experimental change must keep this run and label version intact.

Provider connectivity and sample semantic retrieval are now verified. End-to-end generated answers, production integration, formal latency tests and a full-corpus index remain unverified. Production demo behavior is unchanged.

For any later repeat, reconfirm the free prices of the exact non-Pro IDs. The command below was executed for the sample comparison above; rerunning creates a new run and repeats API calls rather than reusing a cache:

```powershell
venv\Scripts\python.exe scripts\validate_free_retrieval.py --live --acknowledge-free-api
```

One adapter instance allows at most 64 attempted requests, embeddings in batches up to 16 texts, reranking up to 20 documents, 2,048 characters/text, 16,384 characters/request and 30 seconds/request. No redirects, retries, alternate endpoints or paid fallbacks. Failed or limited responses end the run. Input is never silently truncated; invalid vectors and source/index mismatches fail validation. Optional test transport is for trusted offline tests only. The API schema is based on the official [embedding](https://api-docs.siliconflow.cn/docs/api/embeddings-post) and [reranking](https://api-docs.siliconflow.cn/docs/api/rerank-post) references checked 2026-10-24.
