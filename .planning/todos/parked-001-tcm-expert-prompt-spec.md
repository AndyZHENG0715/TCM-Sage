# Parked: TCM expert prompt & answer contract

**Status:** PARKED — waiting for friend / domain feedback
**Created:** 2026-03-28
**Resume when:** Expert returns spec (structure, tone, safety, few-shots)
**Owner:** zianz
**Target feedback date:** TBD (set when friend confirms)
**Resume command:** `/gsd-resume-work`

## What this is

Block prompt redesign and `src/main.py` template changes until we have the TCM expert’s definition of a *good* answer given our RAG inputs.

## Handoff context

- Technical + domain brief to send the expert: `.planning/phases/01-stabilization-bug-fixes/1.6-HANDOFF-APIS-PROMPT.md`
- Conversation copy for the expert model: see latest thread (domain-led prompt brief, SymMap KG, multi-book corpus)

## On resume

1. Merge expert output into `DEFAULT_SYSTEM_PROMPT` / `build_prompt_template()` in `src/main.py`.
2. Align `MessageBubble` / citation UI if answer structure changes.
3. Re-run smoke tests (chat, citations, mobile proxy).
4. Clear **Parked** section in `.planning/STATE.md` or move this file to `done/`.

## Parallel work (not blocked)

- Phase 2 SymMap KG migration (`.planning/phases/02-standard-kg-integration/`)
- Ingest more classical text sources (extend `data/source/` + re-ingest)
- UI/UX polish (Phase 3 roadmap items)
