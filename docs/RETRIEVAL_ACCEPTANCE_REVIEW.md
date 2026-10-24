# Retrieval acceptance review and fresh validation

Prepared 2026-10-24 (Asia/Taipei). Scope: independent acceptance of recorded retrieval evidence, frozen fresh cases and read-only deployment dependency audit. Andy has expanded authorization to free full-corpus processing and online delivery; this document does not authorize paid services or treat an aggregate retrieval score as production readiness.

## Earlier sample: retain the result, clarify its meaning

Reviewed the immutable run `vectorstore/retrieval-validation/20261024T014413Z-ecba9244/report.json`, its case snapshot, sample and index without API calls or environment-file reads. Independently recomputed positive hit@5 and match-aware required coverage: lexical 8/12 and 0.625; dense 11/12 and 0.875; reranked 12/12 and 0.958333. The recorded `needs_review` gate remains correct for the frozen version 2 labels. The run reports 49 preserved files unchanged. Sample/corpus hashes matched current artifacts; all 256 sample/index IDs were unique and ordered consistently, with 1,024-dimensional vectors and no exact duplicate sample contents.

The remaining strict-label miss is not proof of missing requested book coverage. For `cross-book-xiaoqinglong`, expected `备急千金要方_chunk_1548` ranks 6 dense / 8 reranked. It is the short passage in 卷十八大肠腑方／痰饮第六: “病溢饮者，当发其汗，小青龙汤主之”. The alternative `备急千金要方_chunk_850` ranks 3 in both results and is genuinely responsive to the actual question: its 小青龙汤 entry in 卷九伤寒方上／发汗吐下后第九 says “治伤寒表未解，心下有水气，干呕发热而咳”. Together with Shanghan clause 40, this supplies both requested books and the water-related comparison. Both passages were checked in the processed corpus and original `data/source/备急千金要方.txt`; neither text was altered.

Therefore the exact-ID miss exposes incomplete development labels. Do not retrospectively add chunk 850 to version 2, recalculate a perfect score or change its gate. Add this qualitative adjudication beside the immutable numeric result. Related top-five results still include a different book (宣明论方100) and 大青龙汤 in Shanghan clause 39; do not call all five passages relevant. Negative controls continue to retrieve neighbours, and low reranker scores are not an established answerability threshold.

This evidence supports moving from the small engineering sample to a bounded **full-corpus development index and fresh validation**, rather than a claim that semantic retrieval is clinically accurate. The old exact-label gate should not permanently block that experiment solely because an unlabelled, source-verified alternative satisfied the original request. The larger build may proceed under Andy's expanded free-service authorization once fixture integrity, free-model identity, bounded requests and preservation/resume checks are satisfied. Selection of the public live path remains conditional on the fresh results and operational controls below. A failed fresh gate permits investigation, not relabelling the same results into a pass.

## Fresh frozen cases

`scripts/fixtures/retrieval_independent_cases.json` contains 12 new positive cases and three unsupported controls, with 14 distinct expected chunk IDs. None of these expected IDs appeared as a positive label in the earlier fixture. Cases were written and source anchors checked before seeing any semantic results for the new questions. This is a fresh development set, not a blinded independent clinical evaluation: the author knows the previous system and constructed the labels manually. No model selected these labels.

Frozen original file SHA-256: `25893dad77f99dd77ae01ba09266f7ed08f9945caca543c988483cd31f9d3e08`. Preserve its original bytes and separately hash any normalized JSON snapshot. Do not revise positives after inspecting results; if a label issue appears, retain the original run and document qualitative adjudication before a separately versioned future set.

Each case retains the version 2 fields: `id`, `question`, `category`, `expected_chunk_ids`, `expected_match`, `negative_control`, `rationale`, and literal evidence anchors. `expected_match` is `any`, `all` or `none`. Two explicit-clause cases also provide `metadata_filter` in the existing Chroma-compatible `$and` form. Evaluate their unrestricted rank and exact-filter result separately; a filter success must not replace a ranking miss in aggregate metrics.

- Exact Shanghan clause 82: book and clause number must both match.
- Shaoyin Zhenwu symptoms, plus a separate two-setting Zhenwu comparison: shared formula names alone cannot satisfy chapter/context coverage.
- Banxia Xiexin content in clause 149: inspect text despite the misleading single-formula metadata.
- Fumai alias/preparation in clause 177: distinguish dedicated formula evidence from generic Gancao entries.
- Zhuling comparison across Yangming and Shaoyin: require both chapter contexts.
- Lizhong comparison across Shanghan and Qianjin: require one source from each requested book. Quantities differ; do not imply clinical interchangeability.
- Dedicated Huanglian and Shigao entries in Shennong: distinguish entries from incidental ingredients.
- Exact Jingui clause 10 in 肺痿肺痈咳嗽上气病脉证治第七: include the chapter because clause numbering repeats across Jingui chapters.
- Danggui Shengjiang Yangrou comparison across two Jingui chapters: require both settings.
- Sweating version of Guizhi Jia Gegen: distinguish it from the no-sweat Gegen passage.
- Unsupported controls: modern vaccine-trial metadata, Shigao aerospace engineering data, and synthetic `QZVJXK`. The synthetic token was verified to have zero character-bigram overlap across all 12,204 corpus records. Controls are scored separately from positive retrieval.

## Fresh acceptance criteria fixed before results

1. Integrity: every labelled ID and anchor exists in unmodified evidence; all corpus/index IDs are unique and map to the original metadata; hashes, counts, model name, dimensions, preprocessing, sampling/index configuration and cases snapshot are recorded. Verify finite vectors, completeness and resume consistency before publishing an index. A missing index record cannot be treated as a relevance miss alone.
2. Full-corpus development gate: report any-hit@1/@5, MRR@20 and match-aware required coverage@5. The proposed positive gate is at least 10/12 hits@5 and coverage at least 0.75. For `any`, coverage is binary hit; for `all`, it is the fraction of labelled required sources retrieved. Publish all per-case ranks and failures, not only aggregates. Compare lexical, dense and reranked methods on the same candidate corpus and frozen cases.
3. Multi-source review: independently inspect every `all` case for requested book/chapter/context coverage, including responsive unlabelled passages. Retain exact-ID metrics. A genuine context gap remains a gap even if the global gate passes. An incomplete-label finding requires written source-based adjudication, never an automatic perfect score.
4. Exact-filter lane: Shanghan book+82 and Jingui book+chapter+10 must return only records satisfying every requested metadata value and contain the labelled target. Also report each target's unfiltered rank. Repeated numbers in other chapters must not leak into the exact result.
5. Unsupported controls: publish neighbours and scores without treating them as evidence for the unsupported fact. Positive gates cannot approve arbitrary-question answering while every request still returns apparent support. Calibrate abstention with a separate held-out set, or restrict public live use to curated supported questions until answerability handling is verified. Three controls cannot establish a general threshold or false-positive rate.
6. Production delivery is a separate acceptance step: source/citation reconstruction and graph provenance, code denial and rate limits, bounded free-provider requests, no paid fallback, restart-safe allowance, visible limitations, exact off switch and backend-off replay must pass. No clinical-accuracy, general-recall or no-hallucination claim follows from these cases.

The claimed L0 limits of 2,000 RPM / 500,000 TPM for both BGE models are user-reported operational context, not independently verified limits in this review. Apply conservative bounded concurrency and stop on throttling; a quota is neither unlimited capacity nor proof of future free pricing. Keep pinned model IDs, current-price verification and no paid fallback.

## Source/metadata limitations exposed before the fresh run

Read-only inspection found formula metadata captures only one formula even when a clause discusses several: Shanghan clause 149 is labelled 大陷胸汤 while containing the requested 半夏泻心汤 passage; clause 386 is labelled 五苓散 while also containing 理中丸. Formula-only filters would silently lose responsive evidence. Do not change these corpus records during index construction; retain the original metadata and report content-aware retrieval separately.

The Gegen text also exposes a clause-boundary issue: record `伤寒论_clause_30_6485` includes the start of clause 31 and its Gegen formula text beneath the preceding clause. An exact clause-31 metadata query can therefore fail despite that text being present. The fresh exact-filter cases deliberately use independently verified clause 82 and the chapter-qualified Jingui clause 10; this does not resolve or conceal the clause-31 defect. Index counts and unique IDs alone do not establish correct segmentation or comprehensive clause metadata. A later corpus-repair proposal must preserve the original evidence and hashes and receive a separate documented decision.

## Read-only Linux/free-host dependency audit

The current `requirements.txt` is a broad Windows development environment, not a verified Linux deployment lock. It includes Torch, transformers, sentence-transformers, Streamlit, pandas/pyarrow, nodejs-wheel-binaries, developer tooling and pyreadline3. A remote embedding/reranking architecture does not itself need local Transformer model execution, but do not delete these packages blindly: first establish the chosen backend entrypoint and transitive import closure.

The current `src/api.py` imports Arena and Arena statistics eagerly. Statistics import NumPy/SciPy; `ui_backend.py` and `main.py` eagerly import `langchain_chroma`, while `main.py` also imports LangChain and attempts all provider packages. Therefore merely turning off Arena in the UI or skipping Chroma retrieval does not remove those import dependencies. `source_context.py` imports the UI backend too. The graph path requires NetworkX/jieba; external generation adapters require their selected SDKs. No Linux install, import smoke test, dependency resolver or measured memory peak was run in this audit.

For a lean later runtime, prefer a dedicated retrieval HTTP entrypoint with only FastAPI/Uvicorn/Pydantic, the explicitly selected HTTP transport, an on-disk vector format/search implementation, and graph/source modules that do not import generation or Arena. The standalone free-provider adapter currently uses the standard library. Treat this as a proposed dependency boundary, not an already working minimal requirements file. Preserve the existing presentation entrypoint. Build a separate pinned Linux runtime lock and test imports without credentials before claiming free-host fit; do not use system Python or copy the Windows venv into a Linux image.

At 12,204 × 1,024 float32 values, vectors alone occupy 49,987,584 bytes. A Python list-of-floats JSON load uses substantially more memory; that number is not a 512 MB service-fit proof. Prefer measured binary storage with controlled loading/mapping and bounded query work rather than promising that the JSON experiment artifact scales unchanged. The 13 MB graph file and 14.6 MB chunks JSON also expand in memory. Measure process peak and startup time with the actual Linux lock and selected index; retain a backend-independent replay path if the free instance cannot fit.

## Off switches and safe failure recommendations

Enforce server-side switches before initializing clients: live generation off, retrieval network off, and build jobs off independently. Invalid access codes, exhausted allowances, unavailable index/rates or provider errors must make zero further provider calls. Disable both Arena query and vote endpoints server-side; the existing plain Arena branch can invoke DashScope/DDGS independently of demo flags. Pin allowed free embedding/reranker models and endpoint, disable retries/redirects/model fallback, and bound request/body/token/concurrency limits. Never silently switch to a paid model after quota or free-tier failure.

Make emergency shutdown concrete in the chosen runtime: set the server switches to off and restart/redeploy as required by the host; revoke/remove the provider key if the service is compromised. Static prerecorded answers remain available and clearly labelled. A frontend toggle, CORS, provider dashboard alert or `DEMO_MODE` alone is not an effective server spending/network off switch. Do not expose keys or access codes in URLs, logs, source or bundled frontend assets.

## Verification of this review

Only the new fixture and this document were written. Used project venv standard-library operations for source lookup, quote/ID integrity, previous-run metric recomputation and hash checks. No API calls, environment/key reads, source edits, dependency installs, application imports, deployment or git changes were performed. Fresh semantic/full-index results and Linux runtime fit remain unverified at case freeze.
