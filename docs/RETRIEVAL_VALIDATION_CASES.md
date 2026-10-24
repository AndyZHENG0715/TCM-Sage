# Local retrieval validation cases

Prepared 2026-10-24 (Asia/Taipei). Scope: free, isolated retrieval validation only. No deployment, provider calls, embedding rebuild, generation, installation or changes to the corpus are authorized by this fixture.

## Evidence and interpretation

`scripts/fixtures/retrieval_cases.json` contains 12 positive cases and three unsupported controls. Its 15 distinct expected chunk IDs and every quoted anchor were checked against the existing UTF-8 `data/processed/chunks.json`. Questions are synthetic document-search requests, not real patient histories. Historical treatment and longevity statements in quoted material are source content, not factual medical endorsements.

The expected IDs are **manual source-grounded relevance proposals, not expert-reviewed labels**. A verified ID and substring establish provenance; they do not establish clinical truth, corpus completeness, authoritative edition, retrieval quality or the correctness of a generated answer. Nonlisted chunks are unjudged, rather than automatically irrelevant. Do not present these deliberately small, constructed cases as a held-out benchmark or general accuracy estimate. Do not tune questions, labels or distractors after looking at results without retaining the original evaluation and declaring a new version.

The fixture is version 2. Each case carries `id`, `question`, `category`, `expected_chunk_ids`, `expected_match`, `negative_control`, `rationale` and `evidence` with exact `{chunk_id, quote}` anchors. `expected_match` is `any` for relevant alternatives, `all` for the Baihu multiple-context and two cross-book requests, and `none` for unsupported controls. An empty expected list on a negative case means no supporting classical evidence is expected for the requested modern or out-of-domain fact; it is not an exhaustive proof of absence from all possible sources.

Revision note: the first offline run revealed that the version 1 machine token `ZXQJVK987654321` shared numeric bigrams with clause text (top distance 0.6028), so it was not a zero-overlap control. Version 2 corrects the test construction to `ZQXJKV`, verified to have zero bigram overlap across all 12,204 current corpus records, and makes multi-source match semantics explicit. Positive questions, source labels and anchors are unchanged. This is a declared construction correction, not result-driven relevance relabelling. Keep the version 1 runs immutable and snapshot the new fixture with subsequent runs; results from the two versions have different fingerprints.

## Cases and intended distinctions

- `formula-guizhi-composition`: the complete formula and preparation in Shanghan clause 12, rather than later formula-name mentions.
- `symptom-mahuang`: a paraphrased no-sweat, breathlessness and body-pain cluster in clause 35; no clinical recommendation.
- `symptom-shaoyang`: alternating chills/heat, poor appetite and nausea in clauses 96/97; either is a relevant hit.
- `explicit-clause-13`: the exact book and clause target. Clause 12 shares vocabulary but does not satisfy the requested number.
- `guizhi-administration`: hot thin porridge and only slight sweating in clause 12, with a question about the passage's instructions.
- `symptom-wuling`: post-sweating thirst, difficult urination and mild heat in clause 71.
- `symptom-xiaoqinglong`: water/cough discussion in clauses 40/41; clause 40 is the more direct full-cluster passage.
- `symptom-baihu-renshen`: distinguish sweating and vomiting/purging settings in clauses 26/168; finding only one is incomplete coverage.
- `herb-gancao-entry`: the dedicated Shennong Gancao entry, not incidental prescription ingredients.
- `herb-renshen-entry`: the dedicated Shennong Renshen entry, not ingredient lists in other formulae.
- `cross-book-xiaoqinglong`: Shanghan clause 40 and Qianjin chunk 1548; both books are required to satisfy the comparison.
- `cross-book-xiongbi`: Jingui chapter 9 clause 3 and Qianjin chunk 1116; both passages concern chest/back discomfort and breathlessness in the historical terminology.
- `negative-modern-trial`: herb vocabulary is a hard distractor for an unsupported request for a 2026 trial's DOI and effect estimate.
- `negative-hosting`: modern Cloudflare pricing cannot be supported by these classical passages.
- `negative-machine-token`: a synthetic token tests zero-overlap fallback behavior.

## Isolated candidate sample

Use all 15 expected IDs, add nearby formula/clause and same-name herb mentions as deliberate distractors, then deterministic seeded random chunks to reach 256 unique records (within the agreed 200–400 range). Preserve original IDs, content and metadata byte-for-byte as values; write any sample only outside the evidence directories. The harness must record the random seed, selected IDs, original artifact hash, candidate count, retrieval settings and `sample_sha256`, and snapshot the exact cases used. Exact corpus IDs are the matching key, not generated row positions.

The sample is an engineering smoke test. A 256-record sample alters IDF weights and ranking versus all 12,204 records, and it intentionally includes the known positives. Results cannot establish full-corpus performance. Report a baseline before any retrieval changes. A later full-corpus run and independent reviewer labels would be separate validation stages.

## Proposed evaluation gates

These are reviewable engineering gates, not measured outcomes or clinical criteria:

1. Fixture integrity must be 100%: unique case IDs, all positive IDs present in the original and isolated sample, every quoted anchor a literal substring of its original chunk, and negative expected lists empty. Any failed integrity check invalidates scoring.
2. Report positive any-hit@1 and any-hit@5, match-aware macro recall@5 and reciprocal rank. For `any` cases, recall contribution is binary presence of an acceptable alternative; for `all` cases, contribution is the fraction of expected IDs found. A proposed smoke-test gate is at least 10/12 positive any-hits@5 and match-aware macro recall@5 of at least 0.75. Optionally report raw labelled recall separately, but do not penalize alternative labels as if all were required. Publish numerators/denominators and every miss; the small case count makes percentages unstable.
3. Separately require exact clause 13 in the top five for `explicit-clause-13`; report its actual rank. Test metadata-filtered retrieval separately from unfiltered lexical retrieval so an exact filter cannot hide a ranking weakness.
4. For both cross-book cases, report whether top five includes one labelled passage from each requested book. Report both Baihu settings separately. Any-hit@5 alone is insufficient for these multi-source requests. Missing coverage is a review finding even if the aggregate smoke gate passes.
5. Unsupported controls do not contribute to positive recall or MRR. Report returned IDs and scores/distances, zero lexical overlap where applicable, and whether the retriever abstained. A safety-oriented desired outcome is zero passages presented as support for the requested unsupported facts; this requires relevance/answerability handling beyond a top-k retriever. No score threshold is validated here.

## Known implementation limits

`DemoVectorStore` builds an inverted index of unique character bigrams and smoothed document-frequency weights (`src/demo_mode.py`, `_bigrams`, `DemoVectorStore.__init__`, `_score`). Its label of “TF-IDF” in prior documentation is approximate: within-document term frequency does not affect the unique-bigram postings. Scores are lexical overlap, not calibrated relevance probabilities or semantic similarity.

More materially, `similarity_search_with_score` returns the first matching candidate records at distance 1.0 when no bigrams overlap, explicitly to keep the UI populated. Thus the synthetic token can return arbitrary passages rather than abstaining. Record that behavior as a known negative-control failure; do not modify production code or silently weaken the negative labels to obtain a pass. A future answerability/relevance threshold needs separate calibration. Empty negative labels must not be interpreted as permission for an LLM to invent an answer.

Use the project Windows venv only. The validation harness may load the local demo retrieval code and explicit isolated sample without reading `.env`; it must block provider/network access. Record this as **isolated local demo retrieval**, not Chroma vector retrieval, graph retrieval, full application testing or validation of the proposed static showcase. `DEMO_MODE` and `DEMO_LLM` are independent, and Arena bypasses the LLM factory; this fixture neither invokes nor validates them.

## Verification performed while preparing the cases

Read local agent rules, the phase 2a deployment plan, demo retrieval implementation and selected corpus records. Used the project venv's standard-library JSON/file operations to validate 15 expected IDs and all literal anchors and write the fixture. Subsequently reviewed and ran the coordinator's offline-only harness against version 1: 256 unique sampled IDs matched manifest order and covered all expected chunks; the longest sampled chunk was 500 characters, and the conservative top-20 rerank character envelope was 10,015 including the longest question, below 16,384. Positive any-hit@5 was 8/12 and MRR@20 was 0.39325, so the proposed 10/12 gate was not met. Clause 13 ranked 10th; Guizhi administration ranked 12th; both dedicated herb entries ranked 7th. The Xiaoqinglong cross-book request found only one of two labelled sources in the top five. The run checked 49 preserved files and reported all unchanged. These are version 1 local lexical results, not version 2 results, free API results or clinical evaluation. Its numeric-token construction issue is documented above. No application or provider calls were made. Author edits remain confined to this document and the fixture; the authorized harness wrote ignored experiment artifacts. Existing presentation files, HANDOFF, CHANGELOG, environment files and evidence files were preserved.
