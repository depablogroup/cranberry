# Phase 5 REMD Design

Status: implemented and closed for the Phase 5 first slice

## Goal

Add replica exchange MD as an optional `remd` extra without pulling `openmmtools` into the base install.

Phase 5 is now closed. The remaining PBC/box-force correctness work is intentionally moved to Phase 6 because it affects MD and REMD force semantics, not only the REMD workflow.

## What Is Already Settled

- REMD is Phase 5 and should follow the same OpenMM-style packaging and CLI conventions as the rest of Cranberry.
- The base install must continue to work without `openmmtools`.
- REMD should reuse the current `CranberryForceField` and the existing canonical CG validation path.
- The `remd` extra should stay narrow: `openmmtools` plus the minimum reader/writer support needed for NetCDF restart and optional DCD translation. Do not pull MDAnalysis into the public release path.
- PySAGES and REMD remain separate optional extras; we should not promise a single all-in-one environment unless we validate that exact matrix.
- REMD output conventions must stay separate from the MD overwrite/restart contract.
- The terminal-phosphate placement heuristic is frozen as part of the v1 model contract and is not part of the REMD design question.
- The first REMD ladder should be derived from `t_min`, `t_max`, and `n_replicas`, with an override path for an explicit custom temperature-ladder sequence. The initial default replica count is 8.
- REMD restart should treat the NetCDF file as authoritative, and the ladder must match exactly on resume.
- REMD should default to no-overwrite, with `--overwrite` as an explicit opt-in that differs from MD's default overwrite behavior.

## Scope

- Provide a `cranberry remd` workflow for replica exchange MD.
- Keep the CLI thin and map flags directly to a run configuration.
- Follow the same dataclass-driven pattern used by `md`.
- Keep REMD restart and output bookkeeping explicit and separate from ordinary MD.
- Use a compact config object internally, with a default ladder plus an optional custom `temperature_ladder` override.
- Provide a helper that translates the REMD NetCDF output to DCD for inspection, similar to the legacy `nc2dcd` workflow.

## Non-goals

- No base-install REMD implementation.
- No alternative ensemble methods beyond replica exchange in this phase.
- No new force-field model changes.
- No expert tuning surface beyond what is required to make the workflow functional.

## Dependency Boundary

- Base install remains `cranberry-rna` without `openmmtools`.
- The `remd` extra adds `openmmtools` and the smallest direct dependencies needed for NetCDF restart and DCD translation.
- Importing `cranberry` without the extra must continue to work on CPU-only test environments.
- NetCDF-to-DCD translation should use OpenMM and the optional REMD stack, not MDAnalysis, so the base release stays lighter.
- Document the output-contract difference between OpenMM (MD overwrite) and OpenMMTools (REMD no-overwrite) in the user-facing reference docs.

## Proposed First Slice

- Start with a single CPU-friendly REMD smoke workflow.
- Use `1zih` and `2ntCG` as the first canonical fixture pair.
- Keep the first implementation narrowly focused on one default temperature ladder, one restart path, and a tiny exchange schedule.
- Do not couple REMD to MDAnalysis in the default release; that dependency cost is not worth carrying forward.
- Add a small NetCDF-to-DCD translation helper as part of the workflow so users can inspect trajectories without learning the storage format first.

## Decision Checklist

### 1. API Shape

Recommended default: expose a compact run config first, not a free-form low-level wrapper. The default ladder should come from `t_min` / `t_max` / `n_replicas`, with an override path for a custom temperature scale when the user needs it. The default replica count should be 8.

Decision to lock:
- Whether the custom override should be exposed as a literal temperature-ladder sequence or a small temperature-scale object. For the first slice, prefer the literal ladder sequence.
- Whether the Python API should mirror the CLI config exactly or expose a slightly lower-level helper for advanced callers.

Why this matters:
- A too-flexible API risks duplicating OpenMMTools concepts before the workflow is stable.
- A too-small API can make restart and fixture testing awkward.

### 2. Fixture Set

Recommended default: use `1zih` and `2ntCG` as the first regression pair.

Decision to lock:
- Which two canonical systems should be committed first. This is now settled as `1zih` and `2ntCG`.
- Whether the smoke test should prefer the smallest system or the shortest wall-clock path on CPU. This is now settled as a quick CPU path.

Why this matters:
- REMD behavior is more sensitive to exchange count, temperature spacing, and runtime than plain MD.
- The fixture choice determines how quickly we can validate restart and exchange bookkeeping in CI.

### 3. Restart Contract

Recommended default: mirror the existing docs-first output discipline, but keep REMD checkpoint semantics separate from MD. The NetCDF file is the source of truth for resume, and the ladder must match exactly on restart. By default REMD should not overwrite existing outputs; `--overwrite` should be an explicit opt-in.

Decision to lock:
- What metadata, if any, we keep in auxiliary JSON for provenance only.
- Whether the implementation should raise immediately on any ladder mismatch or try to normalize equivalent inputs first. For the first slice, raise immediately.
- Whether checkpoint files should be considered authoritative over any auxiliary run metadata. For REMD, the NetCDF checkpoint should be authoritative.
- Whether the DCD translation helper should live in the runtime API or remain a separate utility command.

Why this matters:
- Replica exchange restart failures are easy to hide if we only check that a file exists.
- The restart rules need to be strict enough to prevent silent ensemble drift.

### 4. Smoke Test Scope

Recommended default: one tiny CPU smoke test in the normal test suite, with the first implementation tuned to stay quick enough for CI.

Decision to lock:
- Whether CI should run a tiny REMD path or leave REMD to an optional test marker. This is now settled as CPU CI.
- How many replicas and steps are enough to prove exchange bookkeeping without creating flaky runtime. Keep this minimal and adjust only if runtime is too noisy.

Why this matters:
- REMD is heavier than the current MD path, so the boundary between CI and manual validation needs to be explicit.

### 5. Helper Reuse

Recommended default: share validation and system construction with `md`, but keep the REMD runner isolated.

Decision to lock:
- Whether any run-state or reporter helpers should be shared directly with `md`.
- Whether a common `run_simulation` helper would reduce duplication without coupling the two workflows too tightly.
- Whether the NetCDF-to-DCD translator should live beside REMD or in a shared analysis utility.

Why this matters:
- Shared helpers reduce maintenance cost, but over-sharing can make the MD and REMD contracts blur together.

## Success Criteria

- `cranberry remd --help` exists and the command is gated behind the optional extra.
- A tiny REMD smoke run can start, advance, and write the expected checkpoint/output files.
- Restart from the REMD checkpoint works.
- The base install still passes import and non-REMD tests without `openmmtools`.
- The REMD implementation uses the same canonical input validation path as the rest of Cranberry.
- The smoke test itself is the first regression contract; no separate benchmark series is required for Phase 5.
- A DCD translation helper exists for inspection of REMD trajectories.

## Initial Shape

- Keep the CLI thin and map flags directly to the run configuration.
- Follow the same dataclass-driven pattern used by `md`.
- Use the current output-contract split so REMD files stay distinct from MD files.

## Practical Recommendation

If we want the smallest workable Phase 5 slice, the right order is:

1. Lock the REMD config shape and restart rules.
2. Add the `remd` extra and dependency boundary in packaging.
3. Implement a tiny CPU smoke path on `1zih`.
4. Add `2ntCG` as the second regression fixture.
5. Document the output contract and restart semantics.

That sequence keeps the design narrow and gives us a working contract before we expand the fixture coverage.


## 2026-07-10 Interval And Metadata Update

- `cranberry remd --steps` is total MD integration steps; OpenMMTools iterations are derived as `max(1, steps // swap_steps)`.
- `--n-record` controls NetCDF checkpoint density through the legacy-style derived checkpoint interval `max(1, steps // (swap_steps * n_record))`.
- `--n-analysis 0` disables OpenMMTools online analysis. Positive values derive an online-analysis interval as `max(1, iterations // n_analysis)`.
- `--extra-start-pdb` adds a second canonical CG coordinate source for alternating initial replica states.
- REMD writes `args.json` for provenance, while `output.nc` remains the authoritative restart state.
- Temperature-organized extraction writes `output_T0.dcd`, `output_T1.dcd`, and `output_temperature_labels.txt`.
- PBC/box force behavior needs a separate audit phase before Cranberry claims full PBC support for MD or REMD. Current REMD can carry topology box vectors into sampler states, but force-level periodic correctness has not been audited.

## Phase 5 Closeout

- Closed after the first REMD slice: optional OpenMMTools dependency boundary, `cranberry remd`, `remd-extract`, NetCDF restart as the authoritative state, provenance `args.json`, no-overwrite default with explicit `--overwrite`, `--extra-start-pdb`, `--n-record`, `--n-analysis`, and DCD extraction by replica or by temperature.
- PBC is explicitly not part of the Phase 5 closure. Phase 6 starts with PBC because double-stranded melting and REMD need correct box/force behavior, while ordinary single-chain runs usually remain nonperiodic.
