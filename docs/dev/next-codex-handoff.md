# Next Codex Handoff

This note is for the next Codex session working in CRANBERRY/cranberry.

## Current State

- Current HEAD should include the Phase 5 REMD closeout commit.
- Working tree should be clean apart from this handoff note when a new session starts.
- Phase 5 REMD is closed.
- Phase 6 PBC has started with an initial explicit-PBC implementation slice; double-stranded melting/REMD correctness remains the next priority.
- The terminal-phosphate placement heuristic is treated as fixed v1 behavior.
- The package is already scaffolded and installable as cranberry-rna.
- The current public surface includes `cranberry inspect`, `cranberry prepare`, `cranberry cg`, `cranberry energy`, `cranberry md`, `cranberry remd`, and `cranberry remd-extract`.
- Phase 3 MD restart behavior is implemented and documented.
- Phase 4 coarse-graining, canonicalization, terminal-phosphate support, and the real `1zih` regression are complete.
- Phase 5 REMD provides optional OpenMMTools-backed parallel tempering, NetCDF restart/provenance metadata, extra start structures, interval controls, and DCD extraction by replica or by thermodynamic temperature.
- Current Cranberry has explicit opt-in PBC wiring for MD/REMD and legacy-compatible force periodicity for WCA, spline, Debye-Huckel, stacking, pairing, and sugar pucker. Broader melting workflow validation is still pending.

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
- Phase 6 initial PBC slice is implemented: `periodic=True`, `--periodic`, generated cubic boxes with configurable padding, explicit REMD sampler-state box vectors, PBC provenance, requested/actual platform provenance, separate MD DCD wrapping via `--enforce-periodic-output`, REMD OpenMMTools platform selection, early periodic cutoff-vs-box validation, and force-level periodic switches matching the legacy audit.
- Developer reports for the main phase slices under docs/dev/progress/.
- Code review reports should include the essential code path and algorithm snippets, not just high-level summaries.

## What Is Next

The next major user-facing work is continuing Phase 6 PBC:

- keep PBC as an explicit opt-in workflow, not inferred accidentally from a PDB box
- prioritize double-stranded melting and REMD correctness; normal single-chain runs usually do not need PBC
- legacy `OpenMM-CGRNA/core/openRNA.py` PBC force behavior has been audited for the initial slice; re-audit only when expanding beyond the current force set
- separate three concepts in design and code: box metadata, force-periodic physics, and trajectory wrapping
- extend validation around the ported force-level periodic behavior for WCA, spline, Debye-Huckel, stacking, pairing, and sugar-pucker/custom forces
- keep pre-commit tests small; use unit-level force audits and minimal CPU smoke tests, with heavier REMD/PBC integration tests outside the fast pre-commit subset

This handoff is for a fresh Codex session. Re-read `AGENTS.md`, `docs/dev/program.md`, `docs/dev/cranberry-v1-plan.md`, `docs/dev/remd-design.md`, `docs/dev/progress/phase-5-remd-report.html`, and the latest report under `docs/dev/progress/`, then continue Phase 6 with double-stranded melting validation, restart checks, and trajectory wrapping semantics.

## Practical Warnings

- Do not reintroduce D_CGRNA.
- Do not expand the public API with legacy tuning knobs unless they are explicitly promoted.
- Preserve the OpenMM-style API and naming conventions.
- Treat the fixed terminal-phosphate heuristic as part of the current model contract.
- Keep tests CPU-first unless a specific task requires otherwise.
- CUDA REMD with online analysis enabled (`--n-analysis` > 0) can fail in PyMBAR/JAX GPU allocation even when OpenMM CUDA is fine; use `--n-analysis 0` or force JAX to CPU for production CUDA runs unless online MBAR is explicitly needed.
- Do not claim the full melting workflow complete until double-stranded fixtures, sampler/box state, restart behavior, and DCD output semantics are tested together.
