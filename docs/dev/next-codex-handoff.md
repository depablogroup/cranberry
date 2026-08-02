# Next Codex Handoff

This note is for the next Codex session working in CRANBERRY/cranberry.

## Current State

- Current HEAD is `813b0a5 Update REMD Langevin defaults`.
- Working tree was clean after that commit when this handoff was updated.
- Phase 5 REMD is closed.
- Phase 6 PBC has started with an initial explicit-PBC implementation slice; double-stranded melting/REMD correctness remains the next priority.
- The terminal-phosphate placement heuristic is treated as fixed v1 behavior.
- The package is already scaffolded and installable as cranberry-rna.
- The current public surface includes `cranberry inspect`, `cranberry prepare`, `cranberry cg`, `cranberry energy`, `cranberry md`, `cranberry remd`, and `cranberry remd-extract`.
- Phase 3 MD restart behavior is implemented and documented.
- Phase 4 coarse-graining, canonicalization, terminal-phosphate support, and the real `1zih` regression are complete.
- Phase 5 REMD provides optional OpenMMTools-backed parallel tempering, NetCDF restart/provenance metadata, extra start structures, interval controls, and DCD extraction by replica or by thermodynamic temperature.
- REMD defaults were updated after benchmarking: `swap_steps=5000`, `n_analysis=0`, and OpenMMTools `LangevinDynamicsMove` with `reassign_velocities=True`. Cranberry records the move name and per-temperature Langevin collision rates in `args.json`.
- Current Cranberry has explicit opt-in PBC wiring for MD/REMD and legacy-compatible force periodicity for WCA, spline, Debye-Huckel, stacking, pairing, and sugar pucker. It also records requested/actual OpenMM platforms, records REMD `JAX_PLATFORM_NAME` provenance, warns about CUDA online-analysis JAX memory risk, and fails early when a periodic box is too small for a force cutoff. Broader melting workflow validation is still pending.

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
- MPI REMD restart and post-processing issues were fixed after regression tests exposed them. The current test suite includes MPI-shaped tests using `mpirun --oversubscribe` when MPI is available.
- Phase 6 initial PBC/platform slice is implemented: `periodic=True`, `--periodic`, generated cubic boxes with configurable padding, explicit REMD sampler-state box vectors, PBC provenance, requested/actual platform provenance, REMD `JAX_PLATFORM_NAME` provenance, separate MD DCD wrapping via `--enforce-periodic-output`, REMD OpenMMTools platform selection, CUDA online-analysis warning, early periodic cutoff-vs-box validation, and force-level periodic switches matching the legacy audit.
- The local CUDA benchmark harness `benchmarks/benchmark_cuda_modes.py` can compare single CUDA MD, multi-process MPS MD, and MPI+MPS REMD on `ggcGCAAgcc`.
- Developer reports for the main phase slices under docs/dev/progress/.
- Code review reports should include the essential code path and algorithm snippets, not just high-level summaries.


## Latest REMD Performance Findings

The most recent local profiling focused on why 8-temperature MPI+MPS REMD seemed much slower than 8 independent MPS MD runners on the local RTX 2060. The updated conclusion is that short benchmark runs were dominated by startup and should not be interpreted as steady-state REMD overhead.

- Profiling used a scratch monkeypatch script at `/tmp/profile_openmmtools_remd.py`; it did not modify repository code.
- Scratch outputs were written under `/tmp/cranberry_openmmtools_profile/`.
- The production-shaped profile used `OPENMMTOOLS_ENABLE_MPI=1`, `JAX_PLATFORM_NAME=cpu`, `mpirun --oversubscribe -n 8`, CUDA, MPS, `ggcGCAAgcc`, 8 temperatures, and `swap_steps=5000`.
- OpenMMTools steady-state path is `MultiStateSampler.run()` -> `_mix_replicas()` -> `_propagate_replicas()` -> `_compute_energies()` -> `_report_iteration()` -> `_update_analysis()`.
- Inside `BaseIntegratorMove.apply()`, OpenMMTools creates a fresh integrator object, asks `ContextCache` for a compatible context, applies sampler state, optionally reassigns velocities, calls `integrator.step(n_steps)`, reads `context.getState(getPositions=True, getVelocities=True, getEnergy=True)`, and updates the sampler state.
- The 100k-step, 20-iteration profile showed first CUDA `get_context()` cost of about 9.2 s per rank, but steady `get_context()` was effectively zero.
- Steady `integrator.step(5000)` was about 3.38 s per call on average, `getState(pos, vel, energy)` about 0.033 s, and steady total move apply about 3.41 s per call.
- Exchange/mixing, energy matrix evaluation, reporting, and analysis bookkeeping were all small at this scale: roughly 0.024 s, 0.006 s, 0.001 s, and 0.012 s per iteration respectively.
- The remaining end-to-end gap in short runs is mostly first CUDA context creation plus MPI/MPS load imbalance and synchronization around the slowest rank, not swap logic, NetCDF writes, online analysis, or velocity reassignment.
- For production-length runs, the steady estimate is close to the direct 8-process MPS MD baseline: roughly 4400-4500 ns/day aggregate for this tiny system on the local RTX 2060.
- Future REMD benchmark comparisons should either warm up before timing or run enough iterations that the first-context cost is negligible. Keep `n_analysis=0`; OpenMMTools may not write real-time YAML for very short runs, so use sufficiently long runs when relying on YAML throughput.

## What Is Next

The next major user-facing work is continuing Phase 6 PBC:

- keep PBC as an explicit opt-in workflow, not inferred accidentally from a PDB box
- prioritize double-stranded melting and REMD correctness; normal single-chain runs usually do not need PBC
- legacy `OpenMM-CGRNA/core/openRNA.py` PBC force behavior has been audited for the initial slice; re-audit only when expanding beyond the current force set
- separate three concepts in design and code: box metadata, force-periodic physics, and trajectory wrapping
- extend validation around the ported force-level periodic behavior for WCA, spline, Debye-Huckel, stacking, pairing, and sugar-pucker/custom forces
- keep pre-commit tests small; use unit-level force audits and minimal CPU smoke tests, with heavier REMD/PBC integration tests outside the fast pre-commit subset
- include the canonical Cranberry citation in the top-level README before public v1

REMD performance follow-up should be lower priority than correctness unless the user asks for it. If continuing performance work, use warmed or long MPI+MPS runs, compare against the existing direct MPS MD baseline, and avoid drawing conclusions from 2-5 iteration runs.

This handoff is for a fresh Codex session. Re-read `AGENTS.md`, `docs/dev/program.md`, `docs/dev/cranberry-v1-plan.md`, `docs/dev/remd-design.md`, `docs/dev/progress/phase-5-remd-report.html`, and the latest report under `docs/dev/progress/`, then continue Phase 6 with double-stranded melting validation, restart checks, and trajectory wrapping semantics.

## Practical Warnings

- Do not reintroduce D_CGRNA.
- Do not expand the public API with legacy tuning knobs unless they are explicitly promoted.
- Preserve the OpenMM-style API and naming conventions.
- Treat the fixed terminal-phosphate heuristic as part of the current model contract.
- Keep tests CPU-first unless a specific task requires otherwise.
- CUDA REMD with online analysis enabled (`--n-analysis` > 0) can fail in PyMBAR/JAX GPU allocation even when OpenMM CUDA is fine; use `--n-analysis 0` or force JAX to CPU for production CUDA runs unless online MBAR is explicitly needed.
- Latest committed-code validation before `813b0a5`: `conda run -n cranberry-dev python -m pytest -q tests/test_remd.py` passed with 12 tests in 111.23 s, then focused post-rebase `tests/test_remd.py::test_run_remd_and_translate_real_openmmtools` passed. Pre-commit also ran pytest and passed during commit.
- After the profiling session, MPS was stopped with `echo quit | nvidia-cuda-mps-control`. Check `nvidia-smi` if a future session suspects stray GPU processes.
- Do not claim the full melting workflow complete until double-stranded fixtures, sampler/box state, restart behavior, and DCD output semantics are tested together.
