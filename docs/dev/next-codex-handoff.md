# Next Codex Handoff

This note is for the next Codex session working in CRANBERRY/cranberry.

## Current State

- The repo has two new uncommitted developer notes: docs/dev/roadmap.md and docs/dev/next-codex-handoff.md.
- The latest committed change at handoff is 6238265 Add MD restart metadata checks.
- The package is already scaffolded and installable as cranberry-rna.
- The current public surface includes cranberry inspect, cranberry prepare, cranberry cg, cranberry energy, and cranberry md.
- Phase 3 MD restart behavior has been implemented and documented.
- Phase 4 has started with the first preparation and canonicalization slice, including terminal phosphate insertion support.

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
- Phase 4 preparation and canonicalization slice, plus terminal phosphate insertion support.
- Developer reports for the main phase slices under docs/dev/progress/.

## What Is Next

The next major user-facing work is the rest of Phase 4:

- fuller coarse-graining workflow
- canonical CG input handling polish
- optional terminal phosphate insertion tuning

After that, the likely order is:

- REMD as an optional extra
- JAX, training, and PySAGES as optional extras
- release hardening, benchmarks, and v1 finalization

## Practical Warnings

- Do not reintroduce D_CGRNA.
- Do not expand the public API with legacy tuning knobs unless they are explicitly promoted.
- Preserve the OpenMM-style API and naming conventions.
- Keep tests CPU-first unless a specific task requires otherwise.
