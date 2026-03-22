# Coding Conventions

**Analysis Date:** 2025-05-14

## Naming Patterns

**Files:**
- **Python:** Snake Case (e.g., `src/api.py`, `src/graph_builder.py`).
- **React Components:** PascalCase (e.g., `web/components/ChatArea.tsx`, `web/components/MessageBubble.tsx`).
- **TS Hooks:** camelCase with `use` prefix (e.g., `web/hooks/useChat.ts`).
- **TS Utilities/Types:** Snake Case or camelCase (e.g., `web/lib/api.ts`, `web/lib/types.ts`).

**Functions:**
- **Python:** Snake Case (e.g., `def run_query_stream(...)`).
- **TypeScript:** camelCase (e.g., `export function ChatArea(...)`, `async function* streamQuery(...)`).

**Variables:**
- **Python:** Snake Case (e.g., `vectorstore_path = "..."`). Constants use SCREAMING_SNAKE_CASE (e.g., `EMBEDDING_MODEL_NAME = "..."`).
- **TypeScript:** camelCase (e.g., `const [messages, setMessages] = useState(...)`). Constants often use SCREAMING_SNAKE_CASE (e.g., `BACKEND_URL = "..."`).

**Types:**
- **Python:** PascalCase for classes (e.g., `class QueryRequest(BaseModel)`). Use type hints for variables and function signatures (e.g., `question: str`).
- **TypeScript:** PascalCase for interfaces and types (e.g., `interface ChatAreaProps`, `type StreamEvent`).

## Code Style

**Formatting:**
- **Python:** Standard PEP 8 (implied). Uses `Path` for file system operations.
- **TypeScript:** Prettier (implied by Next.js defaults). Tailwind CSS for styling.

**Linting:**
- **Python:** Not explicitly configured (no `.flake8` or `pyproject.toml` with lint settings found).
- **TypeScript:** ESLint with `eslint-config-next` (`core-web-vitals` and `typescript`). Configured in `web/eslint.config.mjs`.

## Import Organization

**Order (Python):**
1. Standard library imports (e.g., `os`, `re`, `sys`).
2. Third-party library imports (e.g., `fastapi`, `pydantic`, `dotenv`).
3. Internal module imports (e.g., `from ui_backend import ...`).

**Order (TypeScript):**
1. React/Next.js imports.
2. External libraries (e.g., `lucide-react`).
3. Path Aliases (e.g., `@/lib/types`, `@/components/MessageBubble`).
4. Local relative imports.

**Path Aliases:**
- `@/*` maps to `web/*` (configured in `web/tsconfig.json`).

## Error Handling

**Patterns:**
- **Python:** Use `try...except` blocks. FastAPI `HTTPException` for API errors. `traceback.print_exc()` for logging server-side errors.
- **TypeScript:** `try...catch` blocks for API calls. Rethrowing errors with descriptive messages (e.g., `throw new Error("API Error: ...")`).

## Logging

**Framework:** `print` (Python) and `console.log`/`console.error` (TS).

**Patterns:**
- Use `print` statements in test scripts and during setup.
- Use `console.error` for failed API calls or parsing errors in the frontend.

## Comments

**When to Comment:**
- Module docstrings at the top of Python files.
- Function docstrings (triple quotes) for complex logic in Python.
- Inline comments to explain specific logic or regex patterns.
- Section separators in config files (e.g., `src/config.py`).

**JSDoc/TSDoc:**
- Minimal usage. Mostly relies on TypeScript types for documentation.

## Function Design

**Size:** Most functions are focused and small (10-50 lines). Some utility functions (like `extract_paragraph_context` in `src/api.py`) are more complex.

**Parameters:** Use typed parameters and default values (e.g., `Field(default_factory=list)` in Pydantic models).

**Return Values:** Explicitly typed (e.g., `-> Dict[str, Any]` in Python, `Promise<ChunkContext>` in TS).

## Module Design

**Exports:**
- **Python:** Standard module exports.
- **TypeScript:** Named exports for components and utility functions. Use of `AsyncGenerator` for streaming API responses.

**Barrel Files:**
- Not extensively used. `web/lib/api.ts` and `web/lib/types.ts` serve as central points for their respective domains.

---

*Convention analysis: 2025-05-14*
