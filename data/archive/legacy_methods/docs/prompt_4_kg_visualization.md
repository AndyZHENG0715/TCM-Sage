# Prompt 4: KG Citation Visualization with React Flow

Send to: **frontend agent**

## Goal
Replace the plain-text KG citation display with an interactive graph visualization
using **React Flow** (`@xyflow/react`).

## What to change

### Install dependency
```
npm install @xyflow/react
```

### Create `web/components/KGViewer.tsx`
- Accept a `GraphCitation` (fact string like `陰 --CONTAINS--> 月`, depth, source_ref)
- Parse the fact string to extract source, relationship, and target
- Render 2-3 nodes connected by labeled edges using React Flow
- Style nodes to match the parchment theme (rounded, warm colors)
- Node types: entity nodes (herb, symptom, etc.) with distinct colors by type
- Edge labels show the relationship type (e.g., TREATS, CONTAINS)
- Keep it small and focused — this goes in the citation side panel

### Update `web/components/CitationPanel.tsx`
- In `GraphCitationContent`, replace the plain `<p>{citation.fact}</p>` with
  the new `<KGViewer citation={citation} />` component
- Keep the depth and provenance metadata below the graph

## Design notes
- Panel width is ~400-450px, so the graph should be compact
- Use `fitView` to auto-zoom
- Dark node backgrounds with parchment text to match theme
- Animated edges for visual polish

## Commit
```
feat(web): add interactive KG visualization with React Flow
```
