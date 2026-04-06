# TCM-Sage Frontend

Next.js 16 + React 19 frontend for TCM-Sage.

## What it does

- Chat UI for the RAG backend with SSE streaming
- Citation panel and source drill-down pages
- KG Explorer at `/kg/[entityId]` using Cytoscape.js
- Arena blind A/B evaluation and stats pages
- Chinese/English UI switching via the shared i18n context

## Run locally

```bash
npm install
npm run dev
```

Open http://localhost:3000.

## Backend connection

The frontend talks to FastAPI through `/api/backend/...`.

- `BACKEND_URL`: server-side backend URL for the proxy route
- `NEXT_PUBLIC_BACKEND_URL`: fallback backend URL for browser-side helpers

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
```
