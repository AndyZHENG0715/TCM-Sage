# Phase 1: Stabilization & Bug Fixes

**Phase Goal:** Resolve frontend tech debt, fix UI inconsistencies (markdown, sources trailing, context toggle), and repair the chat history sidebar delete button.

## Technical Strategy
The goal of this phase is to deliver a stable, bug-free MVP UI without touching the underlying core logic or standard knowledge graph (which is reserved for Phase 2). The approach for each task is surgical:
- **Markdown & Bolding:** Update the `MessageBubble` regex/markdown parser to gracefully handle Chinese characters (`“` `”` `「` `」`) around `**` markers.
- **Source Stripping:** Enhance the regex in `MessageBubble` to completely remove LLM-hallucinated trailing sources.
- **Full-Text Toggle:** Use existing state in `CitationPanel.tsx` to call the `/source/{chunk_id}/context` backend endpoint and expand the context passage.
- **KG Visualizer Check:** Validate the integration of `@xyflow/react` in `KGViewer.tsx`.
- **Sidebar Delete Button:** Ensure the `onDeleteSession` callback properly removes sessions from `localStorage` and cleans up current state variables. Often `onClick` events need `e.preventDefault()` alongside `e.stopPropagation()` when nested in clickable parent elements.

## Step-by-Step Execution Plan

### Task 1.1: Fix Markdown bolding and Chinese quote rendering
- **File:** `web/components/MessageBubble.tsx`
- **Action:** Update `normalizeQuotedBoldMarkdown` to correctly normalize spaces and smart quotes around bold markers. Verify that `remarkGfm` correctly interprets standard markdown afterwards.

### Task 1.2: Implement trailing "Sources:" regex stripper
- **File:** `web/components/MessageBubble.tsx`
- **Action:** Improve the `stripTrailingReferenceSection` logic. Make sure it strips out "Sources:", "**Sources:**", and any numbered list trailing at the end of the `message.content`. 

### Task 1.3: Wire up Full-Text Context toggle
- **File:** `web/components/CitationPanel.tsx`
- **Action:** Add a "View Full Paragraph" / "View Snippet" toggle button in the `TextCitationContent` component. Utilize the `fetchChunkContext` to load the full text dynamically.

### Task 1.4: Verify `KGViewer.tsx` rendering
- **File:** `web/components/KGViewer.tsx`
- **Action:** Ensure `@xyflow/react` handles dummy or existing `GraphCitation` facts properly. Make sure `fitView` is called to keep the nodes visible within the small panel width.

### Task 1.5: Fix the delete button in the chat history sidebar
- **Files:** `web/components/Sidebar.tsx` and `web/hooks/useHistory.ts`
- **Action:** 
  1. Add `e.preventDefault()` in `handleDelete` in `Sidebar.tsx`.
  2. Check if the parent `div` click event is capturing the pointer event. Ensure the `Trash2` button receives standard click events.
  3. Validate that `page.tsx` properly clears the chat if the currently active chat is deleted.

## Verification Strategy
- **Empirical Check:** Run the Next.js dev server. Ask the LLM to output bolded Chinese quotes and verify rendering.
- **End-to-End Test:** Ask a prescriptive query that results in a trailing "Sources:" list and verify it is stripped.
- **UI Interaction:** Open the citation panel, toggle the full-text view. Delete a past chat from the sidebar and confirm it disappears completely even after a page refresh.
