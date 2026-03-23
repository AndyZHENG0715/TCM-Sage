# Prompt 2: Fix Rel Score, Markdown Bold, and Source Stripping (Frontend)

Send to: **frontend agent**

## 3 quick fixes in the chat UI, all frontend-only.

---

### Fix A: Rel score always shows 100.0%

**File**: `web/components/CitationPanel.tsx` line 190

**Bug**: Formula `Math.exp(-citation.score / 1000) * 100` divides by 1000, but
ChromaDB L2 distances are typically 0.3–2.0. `exp(-0.5/1000) ≈ 99.95%`.

**Fix**: Replace with a linear mapping that makes sense for L2 distances:
```
Math.max(0, (1 - citation.score / 4) * 100).toFixed(1)
```
This maps: score 0 → 100%, score 1 → 75%, score 2 → 50%, score 4+ → 0%.

---

### Fix B: Markdown bold fails around Chinese quotes

**File**: `web/components/MessageBubble.tsx`, `processedContent` memo (line 38)

**Bug**: `**"阴阳和调"**` renders as literal asterisks. The curly/smart quotes
break `remark-gfm`'s bold boundary detection.

**Fix**: In the `processedContent` useMemo, after the existing `[n]` replacement,
add a second `.replace()` that normalizes smart quotes inside bold markers:
- Replace `**\u201c` with `**"` and `\u201d**` with `"**`
- Or more robustly: `replace(/\*\*\s*([""「」『』])/g, '**$1')` to remove any
  whitespace between `**` and the quote character

---

### Fix C: Strip LLM-generated "Sources:" block from response

**File**: `web/components/MessageBubble.tsx`, `processedContent` memo (line 38)

**Bug**: The LLM sometimes appends its own `Sources:` or `**Sources:**` section
at the end of the response, duplicating the structured citation footer we render.

**Fix**: In `processedContent`, add a regex to strip trailing source blocks:
```ts
.replace(/\n+\*{0,2}Sources?\*{0,2}:?[-\s][\s\S]*$/i, '')
```
This removes everything from the last "Sources" heading onward.

---

## Verification
1. Ask a question → check Rel in citation panel shows realistic values (not 100%)
2. Look for bold text with Chinese quotes → should render as bold, no raw `**`
3. Check that the footer only has styled citation badges, no duplicate text list

## Commit
```
fix(web): correct Rel score formula, bold parsing, and source stripping
```
