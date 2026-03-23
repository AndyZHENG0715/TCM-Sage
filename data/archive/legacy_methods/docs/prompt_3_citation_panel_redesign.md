# Prompt 3: Redesign Citation Panel — Full Paragraph + Full Source Page

Send to: **frontend agent**

## Goal
Redesign the citation panel to use a **two-tier** approach:

### Tier 1: Passage Content (default view in side panel)
Currently shows a 100-char snippet. Change it to show the **full paragraph**
where the cited chunk appears, with the cited text highlighted.

**Changes**:
- `web/components/CitationPanel.tsx` → `TextCitationContent`: auto-fetch context
  via `fetchChunkContext()` when a text citation is selected (instead of requiring
  a button click)
- Display the chunk's `page_content` fully (remove the 100-char truncation — this
  truncation happens in `src/main.py` line 344, `SNIPPET_LENGTH = 100`)
- If chunk context loads, show the surrounding paragraph with `<mark>` highlight
- The Source Chapter field should also be cleaned — currently it repeats content
  from the passage (the `source` metadata contains raw text like `卷一第三。）`
  which overlaps with passage content)

### Tier 2: View Full Context → New Page/Route
The "View Full Context" button currently opens inline in the panel. Change it to
**navigate to a new page** that shows the entire source text.

**Changes**:
- Create `web/app/source/[chunkId]/page.tsx` — a new Next.js route
- This page calls `fetchChunkContext()` and renders the complete `full_chapter_text`
  with the cited section highlighted and auto-scrolled to
- Include a back button to return to the chat
- Style with the parchment theme (match existing design)
- Update the "View Full Context" button in `CitationPanel.tsx` to use
  `window.open()` or Next.js `<Link>` to the new route

## Files to create/modify
- `web/components/CitationPanel.tsx` — auto-fetch, full paragraph display
- `web/app/source/[chunkId]/page.tsx` — **NEW** full source reader page
- `src/main.py` line 344 — increase or remove `SNIPPET_LENGTH` truncation

## Commit
```
feat(web): redesign citation panel with full paragraph and source page
```
