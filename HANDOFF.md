# Demo handoff

Checked at `6a849c2` (origin/main), 2026-11-23 HKT. Canonical checkout: E:/Projects/tcm-sage. The readiness branch contains documentation. The checkout also contains unrelated local edits to README, ignore rules, release verification and legacy Python files, plus untracked configuration, a root lockfile and src/demo_mode.py. Preserve those separately.

## Show the project

Open [TCM-Sage](https://tcm-sage-sandy.vercel.app/). Choose **A clause, with its source**, read the short English answer, then open citation 1 to compare it with the original Shanghan Lun passage. **Explore recorded relationships** opens the saved SymMap graph. The graph describes database relationships and the passage documents a historical claim. Neither establishes clinical efficacy.

The six bilingual examples, their excerpts, and saved graphs are public. Visitors may also open the original interface at /system. Sending a live research question requires the project access code and an available [Render backend](https://tcm-sage-api.onrender.com). Public Arena submissions and raw statistics are disabled. The release runtime uses the packaged BGE index, SiliconFlow embedding/reranking and a fixed free OpenRouter model. Keep provider credentials on the backend.

## Run the replay locally

From the canonical checkout:

```powershell
cd web
npm.cmd ci
npm.cmd run dev
```

Open http://localhost:3000 and select a recorded example. The replay runs from bundled data without a Python server, API keys or retrieval index. Leave NEXT_PUBLIC_PRESENTATION_MODE unset for the showcase. See [release operations](docs/RELEASE_OPERATIONS.md) for live-backend setup and shutdown. The Python release entry point is src/release_api.py. The original presentation entry point is src/api.py.

## Verification

- On 2026-11-23, all six public replays completed at desktop and phone widths. Citation panels, distinct source readers and all graph explorer pages opened. Mobile navigation wraps and most examples initially sit outside the visible horizontal strip. See the [screenshot review](https://github.com/AndyZHENG0715/TCM-Sage/pull/6).
- Local npm ci and dev startup succeeded. An isolated browser replayed clause 13 and displayed the saved citations and graph. All 28 captured requests went to loopback and returned HTTP 200. Tested retrieval mode: bundled recorded fixtures. Neither legacy DEMO_MODE path ran. On this Windows host the default npm cache returned EPERM; the install used a temporary cache.
- Direct Render GET /health at 17:36:54 HKT, after 16 min 28 s since the previous known probe, returned HTTP 200 in 42.636 s (42.632 s to first byte). The warm repeat took 149 ms. This measures the first health response after observed idle. Other public traffic and the instance lifecycle were not independently observable.
- Earlier release verification on 2026-10-25 covered authenticated generation, source bounds and gate enforcement. The previous authenticated query took 30.56 s end to end. See [release verification](docs/RELEASE_VERIFICATION.md) for historical results.

## Remaining work

Decide which first-visit UX proposal to try. Before broader live access, revisit source faithfulness and persistent quota storage: Render can sleep, free providers can rate-limit, and the SQLite ledger depends on the instance filesystem. The six-example showcase remains usable independently. The WSL Ubuntu filesystem contained no directory named application-preview during the cleanup check.

## Index and evidence

The full index has 12,204 records from 17 books, 1,024 dimensions and cosine distance. Local version: `vectorstore/bge-m3/20261024T015651Z-223a0f34`; collection: `tcm_bge_m3`. The builder reused 256 validated sample vectors. Portable artifacts are committed under `deploy/bge-m3`; Linux restored all records offline without provider calls.

Original raw corpus SHA256 `225559d3603c5de3bb72277b124e480d269806b8611cbae160cf677751e717c3` remains recorded. Git normalizes file CRLF to LF on Linux; a declared canonical JSON fingerprint verifies identical records without changing source values. Tests reject altered IDs, content, metadata, embedded newlines and undeclared fallback formats. Corpus and graph remain read-only. Legacy `vectorstore/chroma` is absent; its stale 12,592-ID checkpoint must not be reused or reset.

Frozen independent cases: 12 positive and 3 negative questions. Hit@5: lexical 9/12, dense 11/12, reranked 12/12; required-source coverage 0.958333. The strict gate remains `needs_review` because required Shanghan 223 ranks sixth. Actual six-source output includes 223, 319 and 224; clause 224 is a contraindication and cannot substitute for affirmative indications. Labels/scores remain unchanged. These are manual development metrics, not clinical accuracy. Negative controls retrieve neighbours; no calibrated abstention threshold exists. Details remain in retrieval documents and ignored run `full-20261024T033652Z-47a94a0a`.
