# Public release operations

Goal: a public HTTPS replay with code-restricted live research, original citations and graph navigation. Acceptance means replay/source/graph routes work, unauthorized inference is denied before provider calls, and authorized live output completes with valid source IDs. Free hosts and model quotas can interrupt availability.

## Current deployment

- Frontend: https://tcm-sage-sandy.vercel.app , Vercel Hobby project `tcm-sage`, root `web`, build `npm run build`. Server-only `BACKEND_URL=https://tcm-sage-api.onrender.com`. No provider credentials belong in Vercel or NEXT_PUBLIC variables.
- Backend: https://tcm-sage-api.onrender.com , Render Free service `srv-dbe2k47b9u1s73bvd3ng`, Python3.13.15, one worker. Build: `pip install -r requirements-release.txt && python scripts/restore_bge_index.py --source deploy/bge-m3 --destination vectorstore/release-bge`. Start: `uvicorn release_api:app --app-dir src --host 0.0.0.0 --port $PORT --workers 1`.
- Both follow main. Provider credentials were uploaded only to the explicitly authorized Render service using a service-only file; original .env was not uploaded. Local ignored `.env.release-access` contains the project access code. Share it privately with intended testers; never commit it or put it in a URL.

## Live controls

Backend environment requires `SILICONFLOW_API_KEY`, `OPENROUTER_API_KEY`, `BGE_INDEX_PATH=vectorstore/release-bge`, `LIVE_ENABLED=1` and `LIVE_ACCESS_CODE_HASH` (SHA256 of the access code). Fixed IDs are `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`, `nvidia/nemotron-3-ultra-550b-a55b:free`. OpenRouter request provider price ceilings are zero; no paid fallback or public model override exists.

Initial configured limits: one global concurrent query,2queries/minute per code/IP,10per code/IP/day,20global/day,100global/month. Admission is charged before provider calls, including failed calls. Counters persist only while the same SQLite ledger survives. Render Free replacement instances can reset it; these are not durable account-wide budgets. Unrestricted access or paid models require a durable external ledger first. Direct client IP is used rather than trusting arbitrary forwarded headers; a shared gateway may make IP limits more conservative.

To stop new live admission, set `LIVE_ENABLED=0` in Render and apply the environment deployment. Existing streams may finish. Removing either provider key also fails inference closed. Rotate a leaked tester code by replacing `LIVE_ACCESS_CODE_HASH` and privately distributing the new code. Never disable TLS or remove the gate to troubleshoot availability.

## Index portability

The committed portable artifact has12204float32vectors (1024dimensions), original source IDs/text/metadata and SHA256 manifests. Restore builds native Linux HNSW offline with network denied; it never uses a provider key. Do not copy Windows HNSW binaries. Original raw corpus hashes are preserved, with a declared canonical JSON hash for Git file-line-ending normalization. Content/metadata changes still fail integrity checks. If resuming a partially written destination, use `--resume` only for the matching portable manifest; it verifies existing records before adding missing ones.

## Expected limitations

Render Free sleeps after inactivity; first live requests can take50seconds or more. Frontend proxy allows240seconds; provider/stream limits are shorter and failures show a safe message. Replay stays independent. A verifier badge is fallible source-faithfulness feedback, not clinical validation. Six selected sources may contain contraindications and historical dosage language; preserve negation and inspect originals. Use only non-identifying historical research questions. See RELEASE_VERIFICATION.md for observed tests and remaining checks.
