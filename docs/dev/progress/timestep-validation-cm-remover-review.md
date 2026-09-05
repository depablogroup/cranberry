# Timestep validation and COM-removal review

Date: 2026-09-05  
Branch: `timestep-validation-cm-remover`  
Base commit: `601bd29` (`Validate timesteps and fix pucker imaging`)

## Scope

This public change makes center-of-mass removal explicit for timestep studies
and records the deferred force-field sensitivity findings from the old alpha.1
paper-era H5 validation. It does not change the alpha.1 force expressions.

## Changes

- Added `remove_cmmotion` to `CranberryForceField.createSystem`, retaining the
  compatibility default while allowing strict NVE callers to disable the
  `CMMotionRemover`.
- Updated `benchmarks/timestep_study.py` so Langevin production retains COM
  removal and strict NVE production does not; Langevin equilibration retains
  COM removal.
- Added regression coverage for default and explicit strict-NVE COM behavior.
- Documented stacking radial sensitivity, pairing sensitivity, and the
  unresolved pairing cutoff-continuity risk in `docs/dev/timestep-validation.md`.
- Kept long-run checkpoint runners, one-off COM diagnostics, raw CUDA results,
  and scratch analysis artifacts out of the release commit.

## Validation evidence

The old alpha.1 2ntCG CUDA mixed-precision NVE study used one mass-weighted
COM-velocity projection at the Langevin-to-NVE handoff and no production
`CMMotionRemover`. The 2 fs run completed 100 ns without non-finite values and
bounded total energy; the 2.25 fs run developed repeated large one-step errors.
The exact 2.25 fs event was reproduced on both CUDA and OpenMM Reference, so it
was not attributed to CUDA precision or residual COM translation.

The public markdown records the quantitative force-sensitivity and pairing
cutoff observations and lists the required follow-up scans. Those risks remain
explicitly deferred rather than silently fixed in this commit.

## Tests

Run before commit:

```text
PYTHONPATH=/tmp/cranberry-timestep-validation-worktree pytest tests/test_md.py tests/test_energy.py
git diff --check
```

Result: `33 passed`.

## Risks and follow-up

- Existing callers that rely on the default `createSystem` behavior continue to
  receive a `CMMotionRemover`.
- Strict NVE callers must explicitly remove the initial mass-weighted COM
  velocity once at the ensemble handoff.
- Pairing requires a targeted energy/force scan through its 0.8 nm cutoff
  before it should be considered fully cleared for long NVE validation.
- A future smooth-shell or soft-absolute-value force-field change requires new
  energy, structure, and timestep regression validation.
