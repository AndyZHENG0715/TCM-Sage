# Coding Conventions

**Analysis Date:** 2026-03-23

## Naming Patterns

### Files
- **Backend (Python):** `snake_case` (e.g., `src/api.py`, `src/graph_builder.py`).
- **Frontend (Next.js):** `PascalCase` for React components (e.g., `web/components/ChatArea.tsx`), `camelCase` for hooks (e.g., `web/hooks/useChat.ts`).
- **Scripts:** `snake_case` (e.g., `scripts/e2e_test.py`).

### Variables/Functions
- **Python:** `snake_case` (e.g., `def get_retrieval_context():`).
- **TypeScript:** `camelCase` (e.g., `const handleSendMessage = () => {}`).

## Architectural Patterns

- **Backend:** Uses **FastAPI** with **Pydantic** for request/response validation. Logic is organized into modular Python files under `src/`.
- **Frontend:** **Next.js 16 (React 19)** with **App Router** and **TypeScript**. Uses custom hooks for state management (`useChat.ts`, `useHistory.ts`).
- **Styling:** **Tailwind CSS 4** for styling and **Lucide React** for icons.

## Data Patterns

- **JSON:** Used for intermediate data storage (e.g., `data/processed/chunks.json`, `data/graph/entities_partial.json`).
- **Imports:** Python uses manual `sys.path` manipulation in scripts, while the web app uses TypeScript aliases (`@/lib/...`).
