# Next Codex Handoff

This note is for the next Codex session working in CRANBERRY/cranberry.

## Current State

- The repo has two new uncommitted developer notes: docs/dev/roadmap.md and docs/dev/next-codex-handoff.md.
- The latest committed change at handoff is 6238265 Add MD restart metadata checks.
- The package is already scaffolded and installable as cranberry-rna.
- The current public surface includes cranberry inspect, cranberry prepare, cranberry cg, cranberry energy, and cranberry md.
- Phase 3 MD restart behavior has been implemented and documented.
- Phase 4 now includes the coarse-graining entry point for atomistic inputs plus the canonicalization and terminal phosphate insertion slice.

## What To Read First

Read these files before changing anything non-trivial:

- AGENTS.md
- docs/dev/program.md
- docs/dev/cranberry-v1-plan.md
- the latest report under docs/dev/progress/
- docs/dev/migration-from-openmm-cgrna.md when touching model or fixture provenance

## Important Conventions

- Keep docs/dev/ for developer-only design and operating notes.
- Keep public docs under docs/ and exclude docs/dev/ from the public Sphinx narrative.
- Use the workshop reference runner for regression generation when needed.
- Treat ../OpenMM-CGRNA/ as a read-only reference unless the user explicitly changes that.
- Keep cranberry-rna as the distribution name and cranberry as the import name.

## What Is Already Done

- Phase 1 scaffold, CI, and contributor docs.
- Phase 2 packaged model assets, validation, and energy decomposition.
- Phase 3 MD runner, outputs, restart logic, and regression tests.
- Phase 4 coarse-graining and canonicalization slice, plus terminal phosphate insertion support, is complete.
- Developer reports for the main phase slices under docs/dev/progress/.
- Code review reports should include the essential code path and algorithm snippets, not just high-level summaries.

## What Is Next

The next major user-facing work is Phase 5 REMD:

- add `remd` as an optional feature
- keep `openmmtools` out of the base install
- define REMD success criteria, fixtures, and tests before implementation

After that, the likely order is:

- JAX, training, and PySAGES as optional extras
- release hardening, benchmarks, and v1 finalization

This handoff is intended for a fresh Codex session. Start by re-reading `AGENTS.md`, `docs/dev/program.md`, `docs/dev/cranberry-v1-plan.md`, and the latest report under `docs/dev/progress/`, then begin Phase 5 with a REMD design pass before coding.

## Practical Warnings

- Do not reintroduce D_CGRNA.
- Do not expand the public API with legacy tuning knobs unless they are explicitly promoted.
- Preserve the OpenMM-style API and naming conventions.
- Keep tests CPU-first unless a specific task requires otherwise.
