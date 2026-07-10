# Next Codex Handoff

This note is for the next Codex session working in CRANBERRY/cranberry.

## Current State

- Current HEAD should include the Phase 5 REMD closeout commit.
- Working tree should be clean apart from this handoff note when a new session starts.
- Phase 5 REMD is closed.
- The next implementation phase is Phase 6 PBC, with double-stranded melting/REMD correctness as the first priority.
- The terminal-phosphate placement heuristic is treated as fixed v1 behavior.
- The package is already scaffolded and installable as cranberry-rna.
- The current public surface includes `cranberry inspect`, `cranberry prepare`, `cranberry cg`, `cranberry energy`, `cranberry md`, `cranberry remd`, and `cranberry remd-extract`.
- Phase 3 MD restart behavior is implemented and documented.
- Phase 4 coarse-graining, canonicalization, terminal-phosphate support, and the real `1zih` regression are complete.
- Phase 5 REMD provides optional OpenMMTools-backed parallel tempering, NetCDF restart/provenance metadata, extra start structures, interval controls, and DCD extraction by replica or by thermodynamic temperature.
- Current Cranberry can preserve or carry topology box vectors, but force-level PBC behavior has not been implemented/audited and should not be claimed as supported.

## What To Read First

Read these files before changing anything non-trivial:

- AGENTS.md
- docs/dev/program.md
- docs/dev/cranberry-v1-plan.md
- docs/dev/remd-design.md
- docs/dev/progress/phase-5-remd-report.html
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
- The Phase 4 phosphate-placement heuristic is frozen as part of the v1 contract.
- Phase 5 REMD is complete for the intended first slice: optional dependency boundary, CLI/API sketch, OpenMMTools NetCDF restart, provenance `args.json`, no-overwrite default, `--overwrite`, `--extra-start-pdb`, `--n-record`, `--n-analysis`, and `remd-extract`.
- Developer reports for the main phase slices under docs/dev/progress/.
- Code review reports should include the essential code path and algorithm snippets, not just high-level summaries.

## What Is Next

The next major user-facing work is Phase 6 PBC:

- make PBC an explicit opt-in workflow, not inferred accidentally from a PDB box
- prioritize double-stranded melting and REMD correctness; normal single-chain runs usually do not need PBC
- audit the legacy `OpenMM-CGRNA/core/openRNA.py` PBC path before editing Cranberry force behavior
- separate three concepts in design and code: box metadata, force-periodic physics, and trajectory wrapping
- port force-level periodic behavior carefully for WCA, spline, Debye-Huckel, stacking, pairing, and sugar-pucker/custom forces
- keep pre-commit tests small; use unit-level force audits and minimal CPU smoke tests, with heavier REMD/PBC integration tests outside the fast pre-commit subset

This handoff is for a fresh Codex session. Re-read `AGENTS.md`, `docs/dev/program.md`, `docs/dev/cranberry-v1-plan.md`, `docs/dev/remd-design.md`, `docs/dev/progress/phase-5-remd-report.html`, and the latest report under `docs/dev/progress/`, then start the Phase 6 PBC design/audit pass.

## Practical Warnings

- Do not reintroduce D_CGRNA.
- Do not expand the public API with legacy tuning knobs unless they are explicitly promoted.
- Preserve the OpenMM-style API and naming conventions.
- Treat the fixed terminal-phosphate heuristic as part of the current model contract.
- Keep tests CPU-first unless a specific task requires otherwise.
- Do not claim PBC support until force methods, sampler/box state, and DCD output semantics are tested together.
