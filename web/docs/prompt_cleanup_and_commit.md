# Prompt: Commit Uncommitted Work + Fix Graph Citation Bug

Send to: **any backend agent** with access to the repo.

---

## Context

We're on branch `feature/premium-ui`. Several agents have made changes to the 
working tree but none were committed. There's also one small bug to fix.

## Task 1: Fix graph citation source_type mismatch

In `src/retriever.py`, the graph search sets `source_type = "knowledge_graph"` 
on doc metadata. But in `src/main.py`, `format_docs_with_citations()` checks 
for `source_type == "graph"` (line 333). These never match, so **all graph 
citations are silently dropped**.

**Fix:** In `src/retriever.py`, find where `source_type` is set to 
`"knowledge_graph"` and change it to `"graph"` to match what 
`format_docs_with_citations()` expects.

**Verify:** Run `python src/test_citations.py` — graph citations should now 
appear in formatting output.

## Task 2: Commit all uncommitted work (3 separate commits)

After fixing the bug, make these commits in order:

### Commit 1: Local LLM support
```bash
git add src/main.py .env.example docs/CONFIG.md
git commit -m "feat(llm): add Ollama and LM Studio local LLM providers" \
  -m "Both use OpenAI-compatible API via ChatOpenAI with custom base_url.
No new dependencies required. Enables free local inference for
development and as a cost-effective option for end users."
```

### Commit 2: Shared vectorstore + /source endpoint
```bash
git add src/api.py src/ui_backend.py
git commit -m "perf(api): reuse shared vectorstore in /source endpoint" \
  -m "The /source/{chunk_id}/context endpoint was creating a duplicate
HuggingFace embedding model. Refactored to share the vectorstore from
ui_backend.py since the endpoint only needs Chroma for ID lookups."
```

### Commit 3: Graph citation fix
```bash
git add src/retriever.py
git commit -m "fix(retriever): correct graph source_type to match citation formatter" \
  -m "Graph documents used source_type='knowledge_graph' but 
format_docs_with_citations() checked for 'graph', silently dropping
all KG citations from API responses."
```

## Task 3: Commit the frontend

The entire `web/` directory is untracked. Commit it:

```bash
git add web/
git commit -m "feat(web): add Next.js frontend with premium chat UI" \
  -m "Components: ChatArea, ChatInput, MessageBubble, CitationPanel,
Sidebar, SettingsModal. Hooks: useChat (streaming SSE), useHistory
(localStorage), useSettings, useKeepAlive. Tailwind + Noto Serif SC
typography. Mobile-responsive parchment theme."
```

## Task 4: Clean up stray files

```bash
# Remove files that shouldn't be tracked
git rm --cached endpoint_test.json 2>/dev/null
echo "endpoint_test.json" >> .gitignore
rm endpoint_test.json 2>/dev/null

# Commit cleanup
git add .gitignore
git commit -m "chore: clean up stray files and update gitignore"
```

## Verification

After all commits:
```bash
git log --oneline -6
git status
```

Expected: clean working tree, 4 new commits on `feature/premium-ui`.
