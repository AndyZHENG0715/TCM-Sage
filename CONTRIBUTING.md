<!-- generated-by: gsd-doc-writer -->
# Contributing to TCM-Sage

TCM-Sage is an FYP academic project, currently single-developer, but contributions and feedback are welcome. This guide covers how to report issues, propose changes, and follow the project's conventions.

## Development Setup

See [README.md](README.md) for prerequisites and installation steps. In short:

1. Clone the repo and create a Python venv
2. `venv\Scripts\python.exe -m pip install -r requirements.txt`
3. `cd web && npm install`
4. Copy `.env.example` → `.env` and configure at least one LLM provider key

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for all environment variables.

## Reporting Issues

Open an issue on [GitHub Issues](https://github.com/AndyZHENG0715/TCM-Sage/issues). Please include:

- A clear description of the problem or request
- Steps to reproduce (for bugs): query text used, provider/model, relevant `.env` settings (omit API keys)
- Expected behaviour vs. actual behaviour
- Python version, OS, and Node.js version if applicable

## Submitting a Pull Request

1. Fork the repository and create a branch from `main`
2. Use descriptive branch names, e.g. `feat/add-provider-x` or `fix/citation-panel-scroll`
3. Make focused commits — one logical change per commit
4. Verify your changes manually (see [Running Tests](#running-tests))
5. Open a PR against `main` with a description of what changed and why

No CI is currently configured. The reviewer (project author) will test PRs locally.

## Coding Standards

**Python (`src/`, `scripts/`):**
- `snake_case` for files, functions, and variables
- Type hints on public functions; Pydantic models for API request/response types
- `sys.path` bootstrap before local imports — `src/` is not a package
- Never include `"Sources:"` or `"References:"` sections in LLM output — the UI strips them
- Never use `"KG"` or `"Knowledge Graph"` in text sent to the LLM (leaks RAG identity in Arena)
- Use `HybridRetriever` from `src/retriever.py` — do not create parallel retrieval implementations
- DashScope API batch limit is 10 (embeddings and reranker) — respect this in any new ingestion code

**TypeScript/Next.js (`web/`):**
- `PascalCase.tsx` for components, `camelCase.ts` for utilities and hooks
- Path alias `@/*` resolves to `web/*` — use `@/lib/...`, `@/components/...`
- Always route backend calls through the `/api/backend/` Next.js proxy — never call `localhost:8000` directly
- Keep `web/lib/types.ts` in sync with `src/citation_types.py` TypedDicts
- No barrel `index.ts` files — import from concrete paths

**Linting:**
```bash
# Frontend
cd web && npm run lint
```

No Python linter is currently configured. Follow the conventions above.

## Running Tests

There is no pytest setup. Tests are standalone scripts run directly with the project venv:

```bash
# Python
venv\Scripts\python.exe src/test_citations.py
venv\Scripts\python.exe src/test_graph.py
venv\Scripts\python.exe src/test_hybrid_retriever.py
venv\Scripts\python.exe scripts/verify_symmap_retrieval.py

# Frontend build check
cd web && npm run build
```

Run relevant test scripts before submitting a PR to confirm nothing is broken.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
