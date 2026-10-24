# TCM-Sage Frontend

Next.js 16 + React 19 frontend for TCM-Sage.

## What it does

- Default public showcase: six clearly labelled, manually curated recorded answers with corpus citations and bounded SymMap snapshots. Replay, citation expansion, the two recorded full books and recorded graph navigation resolve from bundled data without a backend request.
- Optional single-turn live preview: a visitor supplies a project access code; `/api/backend/query` carries it as a Bearer header. The backend must enforce access and usage limits. The code is kept in memory only, never bundled or saved in local storage. No client-selected model or provider settings are sent by the public composer.
- `/arena` and `/arena/stats` show the final report's frozen 59 preference votes (38 / 15 / 6), with limitations and source attribution. Public voting and Arena generation are disabled in the UI; the backend must also deny those paths.
- English is the clean-session default, with Simplified/Traditional Chinese available. Fonts use OS fallback stacks and require no build-time font downloads.
- The existing presentation chat and Arena components are retained behind the explicit build-time `NEXT_PUBLIC_PRESENTATION_MODE=1` opt-in. Do not enable that flag on the public release; those historical components retain their original operational behaviour.

## Run locally

```bash
npm install
npm run dev
```

Open http://localhost:3000.

Done means the first cited recorded example is visible without a backend, replay is labelled as prerecorded, source/graph navigation works, and live generation is admitted only by the protected backend. The permanent research/not-medical-advice notice applies to replay and live outputs. Neither a citation nor a recorded graph relationship establishes clinical reliability.

## Backend connection

The frontend talks to FastAPI through `/api/backend/...`.

- `BACKEND_URL`: server-side backend URL for the proxy route
- `NEXT_PUBLIC_BACKEND_URL`: fallback URL read by the server-side proxy; prefer `BACKEND_URL`

Default backend: `http://127.0.0.1:8000`

## Key directories

- `app/` — routes and pages (`page.tsx`, `arena/`, `source/`, `kg/`, `api/backend/`)
- `components/` — UI building blocks
- `hooks/` — chat, arena, and settings state
- `i18n/` — `context.tsx`, `zh.json`, `en.json`
- `lib/` — API client, markdown, and shared types

## Useful commands

```bash
npm run lint
npm run build
node --test scripts/showcase.test.mjs
```

When Windows sandbox permissions block Next.js telemetry configuration, set `NEXT_TELEMETRY_DISABLED=1` for the build process. A sandbox prerender filesystem error is distinct from a TypeScript/build failure; validate in the intended clean build environment before release. No paid inference is required by the showcase tests.

`lib/showcase-data.json` records the original corpus and graph SHA-256 hashes, exact source excerpts and graph edge provenance. The explanatory answers were manually prepared, not generated live. Source passage views explicitly distinguish a cited excerpt from an entire reconstructed chapter. Graph expansion is limited to saved snapshots. Keep these qualifications when adding examples; never invent a source, graph edge or relevance probability to improve the presentation.

The backend proxy uses Node.js with Vercel Fluid compute and a 300-second maximum duration. It aborts upstream work after 240 seconds, cancels on browser disconnect, preserves incremental SSE, disables response caching and returns sanitized connection/timeout errors. Set server-only BACKEND_URL to the release backend; redirects are rejected so access codes cannot follow an upstream redirect. Include node --test scripts/proxy.test.mjs when checking proxy changes.
