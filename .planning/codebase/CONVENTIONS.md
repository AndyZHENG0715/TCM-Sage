# Coding Conventions

**Analysis Date:** 2026-04-01

## Naming Patterns

**Python (`src/`):**
- Files use `snake_case.py` (examples: `src/ui_backend.py`, `src/graph_builder.py`, `src/citation_types.py`).
- Functions and variables use `snake_case` (examples: `format_docs_with_citations`, `verify_citation_bounds` in `src/main.py`).
- Constants use `UPPER_SNAKE_CASE` (examples: `GRAPH_DATA_PATH`, `DEFAULT_RETRIEVAL_K` in `src/config.py`).
- Local tests are named `test_*.py` and co-located in `src/` (examples: `src/test_citations.py`, `src/test_graph.py`).

**Next.js + TypeScript (`web/`):**
- App Router pages follow Next defaults: `web/app/page.tsx`, `web/app/layout.tsx`, dynamic route `web/app/source/[chunkId]/page.tsx`.
- React component files use `PascalCase.tsx` (examples: `web/components/CitationPanel.tsx`, `web/components/MessageBubble.tsx`).
- Hooks use `useX.ts` and export `useX` functions (examples: `web/hooks/useChat.ts`, `web/hooks/useSettings.ts`).
- Shared modules use lower camel/snake-like utility names in `web/lib/` (examples: `web/lib/api.ts`, `web/lib/citations.ts`, `web/lib/utils.ts`).

**Scripts (`scripts/`):**
- Script files are mostly `snake_case.py` and action-oriented (examples: `scripts/import_symmap_kg.py`, `scripts/verify_symmap_retrieval.py`, `scripts/check_health.py`).
- Diagnostic scripts commonly start with `check_`, `verify_`, or `test_` and are run directly from repo root.

**Planning (`.planning/`):**
- Canonical planning docs use uppercase names in `.planning/codebase/` (examples: `.planning/codebase/CONVENTIONS.md`, `.planning/codebase/TESTING.md`).
- Phase folders are numeric slug format (examples: `.planning/phases/01-stabilization-bug-fixes`, `.planning/phases/02-standard-kg-integration`).
- Per-phase files follow numeric prefixes and role suffixes (examples: `.planning/phases/02-standard-kg-integration/2-01-PLAN.md`, `.planning/phases/02-standard-kg-integration/2-03-SUMMARY.md`).

## Code Style

**Formatting:**
- Python style is consistent 4-space indentation with module docstrings in larger modules (examples: `src/main.py`, `src/api.py`, `src/test_citations.py`).
- No repo-level `black`/`ruff` formatting config detected; preserve local style per file when editing.
- Frontend has no dedicated Prettier config file; style is enforced primarily via ESLint and TypeScript strictness.

**Linting / Type Checking:**
- Frontend lint config is `web/eslint.config.mjs` using `eslint-config-next/core-web-vitals` and `eslint-config-next/typescript`.
- Frontend command is `cd web && npm run lint` (defined in `web/package.json`).
- Python static type settings are defined in `pyrightconfig.json` (targets `src` and `scripts`, Python 3.13 environment).

## Import Organization

**Python:**
1. Standard library imports first.
2. Third-party imports second.
3. Local project imports after path bootstrap when needed.

**Path bootstrap convention (important):**
- `src/` is used as flat modules, not an installed package.
- Entry/test scripts prepend or append to `sys.path` before local imports (examples: `src/api.py`, `src/test_citations.py`, `scripts/verify_symmap_retrieval.py`).
- For new runnable Python files under `src/` or `scripts/`, follow existing `Path(...).resolve()` + `sys.path.insert/append` pattern to keep imports stable from repo root.

**TypeScript/React:**
1. External packages first.
2. `@/` aliases second.
3. Relative imports last (for colocated modules).
- Alias mapping is `@/* -> ./*` in `web/tsconfig.json`; use `@/components/...`, `@/hooks/...`, `@/lib/...` over deep relative paths.

## Types and Data Contracts

**Backend contracts:**
- Citation schema is defined in `src/citation_types.py` via `TypedDict` + `Literal`.
- API request/response models are `pydantic.BaseModel` classes in `src/api.py` (for example `QueryRequest`, `ConfigResponse`).

**Frontend parity:**
- Mirror citation and message types in `web/lib/types.ts`.
- Keep `src/citation_types.py` and `web/lib/types.ts` synchronized whenever citation fields change.

## Error Handling and Logging

**Backend:**
- API path uses explicit `HTTPException` for request validation and runtime failures in `src/api.py`.
- Generator/streaming code wraps failures and emits structured error payloads in `src/api.py`.

**Frontend:**
- Async fetch handling checks `instanceof Error` before accessing error messages (example: `web/components/CitationPanel.tsx`).
- Streaming and API parse failures are logged with `console.error` and surfaced in user-visible message fallbacks (example: `web/hooks/useChat.ts`, `web/lib/api.ts`).

**Logging baseline:**
- Python tests/scripts are largely `print`-driven for operator feedback (examples: `src/test_graph.py`, `scripts/check_health.py`).
- No centralized production logging framework is currently enforced.

## Config and Environment Safety

**Configuration sources:**
- Environment variables are loaded by `load_dotenv()` in `src/main.py` and `src/api.py`.
- Central path/default constants live in `src/config.py`; prefer importing these constants over duplicating literals.
- Key variable names and comments are documented in `.env.example` and `docs/CONFIG.md`.

**Safety rules for contributors:**
- Never read from or commit secrets in `.env`; treat `.env.example` as the only shareable template.
- Keep `.env` and `.env.backup` excluded (already ignored in `.gitignore`).
- Add new env keys to both `.env.example` and `docs/CONFIG.md` with safe placeholder values.
- Do not hardcode API keys or provider tokens in `src/`, `web/`, or `scripts/`.

## Practical Contributor Rules

- Keep backend/frontend contract changes paired: `src/citation_types.py` + `web/lib/types.ts` + targeted tests in `src/test_citations.py`.
- When adding API fields, update both server model definitions in `src/api.py` and mapping logic in `web/lib/api.ts`.
- For new scripts, prefer explicit `main()` entry points and clear stdout diagnostics, matching patterns in `scripts/import_symmap_kg.py` and `scripts/verify_symmap_retrieval.py`.
- For planning updates, follow established naming (`*-PLAN.md`, `*-SUMMARY.md`, `*-VERIFICATION.md`) under `.planning/phases/`.

---

*Convention analysis: 2026-04-01*
