**Generated:** 2026-04-08 | **See also:** root AGENTS.md, web/components/AGENTS.md

# web/ — Next.js Frontend

## Overview

Next.js 16 + React 19 chat UI consuming FastAPI backend via SSE. Citation panel, KG visualization, settings.

## Structure

```
web/
├── app/                    # App Router pages/layouts
│   ├── page.tsx            # Main chat page
│   ├── arena/page.tsx      # Arena blind A/B evaluation page
│   ├── source/[chunkId]/   # Citation drill-down page
│   └── api/backend/[...path]/ # Proxy to FastAPI
├── components/             # React components (see web/components/AGENTS.md)
│   ├── MessageBubble.tsx   # Chat message + citation markers
│   ├── ArenaPanel.tsx      # Arena response panel (reuses shared markdown)
│   ├── ArenaVoteBar.tsx    # Arena voting UI (A/B/Tie)
│   ├── ArenaReveal.tsx     # Arena reveal overlay with citations
│   ├── ArenaModelSelector.tsx # Model preset chip selector
│   ├── CitationPanel.tsx   # Source context panel
│   └── KGViewer.tsx        # Knowledge graph visualization
├── hooks/                  # React hooks (useX.ts)
│   ├── useChat.ts          # SSE streaming + message state
│   ├── useArena.ts         # Arena dual-stream SSE + voting state
│   └── useSettings.ts      # Runtime settings
└── lib/                    # Utilities
    ├── api.ts              # Backend API client (+arena functions)
    ├── markdown.ts         # Shared markdown rendering (citations, formatting)
    ├── types.ts            # TypeScript types (mirror src/citation_types.py)
    └── citations.ts        # Display helpers
```

## Key Patterns

**API Proxy:** `app/api/backend/[...path]/route.ts` forwards to `BACKEND_URL` (default `http://127.0.0.1:8000`)

**SSE Streaming:** `useChat.ts` handles `text` events + final `metadata` event with citations/verification

**Path Aliases:** `@/*` → `web/*` (tsconfig paths). Use `@/lib/...`, `@/components/...`

## Conventions

- **Files:** `PascalCase.tsx` components, `camelCase.ts` utilities/hooks
- **Imports:** External → `@/` aliases → relative
- **Types:** `export type` in `web/lib/types.ts`; strict mode enabled
- **Lint:** `npm run lint` (ESLint 9 + eslint-config-next)

## Where to Add Code

| New Feature | Location |
|-------------|----------|
| Page/route | `app/` following App Router conventions |
| UI component | `components/` as `PascalCase.tsx` |
| Hook | `hooks/` as `useX.ts` |
| API call | `lib/api.ts` (add function, mirror backend route) |
| Shared types | `lib/types.ts` |

## Commands

```bash
npm install              # Install deps
npm run dev              # Dev server at :3000
npm run build            # Production build
npm run lint             # ESLint
```

## Anti-Patterns

- **Direct backend calls:** Always use `/api/backend/` proxy, not direct `localhost:8000`
- **Citation type drift:** Keep `lib/types.ts` in sync with `src/citation_types.py`
- **Barrel files:** No `index.ts` pattern — import from concrete paths

## Environment

- `BACKEND_URL` / `NEXT_PUBLIC_BACKEND_URL` — Backend URL (default `http://127.0.0.1:8000`)
- Turbopack enabled via `next.config.ts`
