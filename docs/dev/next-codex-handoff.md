# Next Codex Handoff

This note is for the next Codex session working in CRANBERRY/cranberry.

## Current State

- As of 2026-08-04, a substantial publication-readiness change set is present in the worktree but is not committed or pushed. The exact local status is recorded below; do not assume these changes are on `main`.
- The local publication-readiness report is `docs/dev/progress/publication-readiness-report.html`.
- The final local validation passed: 77 tests, Sphinx build, privacy hook, citation parsing, final wheel/source build, and wheel asset checks. The test run took 308.27 s and emitted one pre-existing NumPy binary-ABI warning during OpenMMTools import.
- GitHub-side gates remain: confirm account email privacy and command-line push blocking, commit/push, obtain passing Actions including Gitleaks, make the repository public, and protect `main`.

- Current HEAD is `813b0a5 Update REMD Langevin defaults`.
- The worktree is not clean: the publication-readiness change set described below is uncommitted and unstaged.
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


## Publication-Readiness Worktree

The current uncommitted change set includes:

- Researcher-facing `README.md`, `CITATION.cff`, `docs/citation.md`, and `THIRD_PARTY_NOTICES.md`.
- Expanded `cranberry/data/README.md` with model/fixture hashes and provenance, including PDB and Web 3DNA attribution.
- Python metadata narrowed to `>=3.11,<3.12`; package data now includes the nested atomistic 1ZIH fixture and README.
- Benchmark snapshot sanitization for absolute paths and hostnames, with current committed YAML snapshots normalized.
- `scripts/check-publication-privacy.sh`, a pre-commit hook, and CI checks for privacy, distribution builds, package assets, and Gitleaks.
- REMD shared-box initialization for multiple starting conformations, bounding-box-midpoint centering, exact ordered atom/bond topology validation, common box-vector provenance, and restart metadata compatibility checks.
- A CLI regression fix for `cranberry energy --json`, which previously accessed a nonexistent timestep argument.
- Tests covering the new behavior in `tests/test_cli.py`, `tests/test_md.py`, and `tests/test_remd.py`.
- Developer path/hostname sanitization in the existing reports and reference-generation notes.

The local repository identity is configured as `Yiheng Wu <39614623+yihengwuKP@users.noreply.github.com>`. Historical commits still contain old university and internal service-style author addresses. They are not credentials; history rewriting was intentionally deferred. Current files do not expose those paths or benchmark hostnames.

The final artifacts were built outside the repository under `/tmp/cranberry-publication-final-20260803/`. They are validation outputs, not release artifacts to commit.

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
- The canonical Cranberry citation is now in the top-level README and `CITATION.cff`; verify the rendered links after push.

REMD performance follow-up should be lower priority than correctness unless the user asks for it. If continuing performance work, use warmed or long MPI+MPS runs, compare against the existing direct MPS MD baseline, and avoid drawing conclusions from 2-5 iteration runs.
Before any public release, review the publication-readiness report and the complete uncommitted diff, then commit and push only after explicit approval. After the first successful GitHub Actions run, make the repository public and require the CI checks through branch protection or a ruleset. Do not rewrite history unless the historical author metadata is judged unacceptable.


This handoff is for a fresh Codex session. Re-read `AGENTS.md`, `docs/dev/program.md`, `docs/dev/cranberry-v1-plan.md`, `docs/dev/remd-design.md`, `docs/dev/progress/phase-5-remd-report.html`, and the latest report under `docs/dev/progress/`, then continue Phase 6 with double-stranded melting validation, restart checks, and trajectory wrapping semantics.

## Practical Warnings

- Do not reintroduce D_CGRNA.
- Do not expand the public API with legacy tuning knobs unless they are explicitly promoted.
- Preserve the OpenMM-style API and naming conventions.
- Treat the fixed terminal-phosphate heuristic as part of the current model contract.
- Keep tests CPU-first unless a specific task requires otherwise.
- CUDA REMD with online analysis enabled (`--n-analysis` > 0) can fail in PyMBAR/JAX GPU allocation even when OpenMM CUDA is fine; use `--n-analysis 0` or force JAX to CPU for production CUDA runs unless online MBAR is explicitly needed.
- Latest committed-code validation before `813b0a5`: `conda run -n cranberry-dev python -m pytest -q tests/test_remd.py` passed with 12 tests in 111.23 s, then focused post-rebase `tests/test_remd.py::test_run_remd_and_translate_real_openmmtools` passed. Pre-commit also ran pytest and passed during commit. The newer uncommitted publication-readiness tree passes 77 tests; do not describe that result as a committed validation until the change set is committed.
- After the profiling session, MPS was stopped with `echo quit | nvidia-cuda-mps-control`. Check `nvidia-smi` if a future session suspects stray GPU processes.
- Do not claim the full melting workflow complete until double-stranded fixtures, sampler/box state, restart behavior, and DCD output semantics are tested together.


## Timestep Validation Handoff (2026-08-11)

### Operational State

- Branch: `timestep-validation`. No timestep-validation process is running. The local CUDA queue was deliberately stopped before the workstation was disconnected.
- Do not restart long trajectories locally without checking with the user; remaining microsecond studies are better suited to the cluster.
- The former 5 fs NVE recommendation is withdrawn. No NVE timestep has passed 1 microsecond for 1l2x. Use 1 fs only as a provisional diagnostic timestep with explicit drift monitoring.
- The 8 fs Langevin recommendation remains supported at 298 K by a completed 1 microsecond 1l2x run, prior multi-seed configurational comparisons against 1 fs, and constrained velocity/kinetic-energy checks.

### Long NVE Results

- 5 fs failed at 135.75 ns.
- A corrected 3 fs run removed initial COM velocity and localized progressive heating at 293.999706 ns. Total energy had risen by 8812.03 kJ/mol and instantaneous kinetic temperature from about 307 K to 2630 K. The first physical bond excursion above 0.1 nm was residue 10 P-S3 at 293.067807 ns.
- 2.75 fs became nonfinite between 420 and 421 ns. The result serializer originally rejected NaN; `benchmarks/nve_survival_bracket.py` now checks finiteness before recording.
- 2.5 fs reached 60 ns without failure before the user stopped the run. It is not validated. 2.25, 2.0, 1.5, and 1.0 fs microsecond runs were not performed.
- Detailed 3 fs output is local and ephemeral: `/tmp/cranberry-nve-adaptive-3fs/result.json`.

### Failure Localization

- The failure is secular numerical heating followed by distributed backbone-bond excursions, not an initial COM artifact or a single defective bond.
- Five-nanosecond 1l2x drift slopes at 3 fs were: full +13.21, no-pucker -0.13, no-spline +5.87, no-pairing +6.13, no-WCA -2.66, and no-stacking +10.48 kJ/mol/ns. These alter the Hamiltonian and thermalized state, so treat them as localization evidence, not additive attribution.
- C3-only in the six-particle compound force still drifted +11.61 kJ/mol/ns. An equivalent split into harmonic bonds/angles and CBT torsions drifted +14.91 kJ/mol/ns. Therefore C2/C3 switching and `CustomCompoundBondForce` itself are not the identified root cause.
- The strongest sugar coordinate is S2-B1 with k=27000 kJ/mol/nm^2; ordinary P-S3 is about 32000. A short 2ntCG Reference decomposition also ranked S2-B1 above the remaining C3 angles/torsions.
- The constrained mass-weighted Hessian gives a fastest period near 115 fs, so a period-only heuristic incorrectly predicts 3 fs should be easy. Nonlinear coupling and nonsmooth spline/pairing/WCA behavior also contribute. Exact term-by-term attribution remains incomplete.

### Langevin Evidence

- The completed 8 fs velocity figure and JSON are `docs/dev/progress/velocity-distribution-8fs.png` and `.json`. Correct DOF subtract explicit constraints and three COM degrees.
- 1l2x standardized velocity statistics at 8 fs: std 1.0076, skew -0.0098, excess kurtosis 0.0350; kinetic `2K/RT` mean 463.06 versus chi-square mean 456 and variance 950.4 versus 912.
- A Maxwell/chi-square check is necessary but insufficient. LFMiddle can maintain momentum statistics while configurational distributions are timestep-biased. Primary selection evidence is the existing state-conditioned Wasserstein comparison against 1 fs in `benchmarks/results/timestep-validation-rtx2060-summary.json`.
- The attempted extended 5 ns, three-seed 298/420 K NVT and GHMC rerun did not execute after the original periodic-box harness error. The harness now prepares a 4 nm padded periodic system and passes a CUDA smoke test, but the long matrix was stopped.

### PBC/OpenMMTools Fix

- Custom bonded forces do not define OpenMM molecule connectivity. Cranberry now adds zero-energy HarmonicBondForce entries for topology edges otherwise represented only by custom sugar forces.
- Pucker remains nonperiodic as a bonded internal-coordinate force. With complete connectivity, `enforcePeriodicBox=True` images the full RNA as one molecule and no longer changes pucker energy.
- This was a Cranberry System-connectivity bug exposed by ordinary OpenMMTools state imaging, not an OpenMMTools replica-exchange bug.

### Validation and Next Steps

- Focused validation: `tests/test_timestep_study.py` passes 5 tests; the periodic pucker regression and MD periodic-force expectations are included. `git diff --check` passed before commit.
- Corrected narrative: `docs/dev/timestep-validation.md`. It prominently supersedes the historical short-run 5 fs NVE conclusion.
- On a cluster, first run matched pucker subterm ablations from a shared equilibrated state, then test whether constraining S2-B1 and P-S3 removes drift without unacceptable ensemble changes. Only after the Hamiltonian is settled should 1/1.5/2/2.25/2.5 fs be bracketed to 1 microsecond.
- For Langevin, rerun the fixed 298/420 K multi-seed matrix and GHMC curve, then regenerate velocity figures across 1/5/8/10/12 fs. Do not use a universal 95% GHMC threshold.
