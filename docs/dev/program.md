# Cranberry Development Program

This developer-only operating guide tells future agent sessions how to work on Cranberry. It complements `AGENTS.md`, which should stay short, and `docs/dev/cranberry-v1-plan.md`, which records the scientific and packaging design.

## Read First

For non-trivial work, read these files before editing:

- `AGENTS.md`
- `docs/dev/program.md`
- `docs/dev/cranberry-v1-plan.md`
- the latest relevant report under `docs/dev/progress/`
- `CONTRIBUTING.md` when changing workflow, tests, or CI

Read only the additional files needed for the task. Prefer targeted code reading over broad rewrites.

## Current Objective

Build `cranberry-rna` as a clean installable Python package with import name `cranberry`.

The package should provide a small OpenMM-style public API, packaged runtime data, validation, CLI entry points, tests, and documentation. Advanced JAX, training, PySAGES, and REMD functionality should remain optional extras.

## Scope Rules

- Treat `../OpenMM-CGRNA/` as a read-only code reference unless the user explicitly asks otherwise.
- Treat `../CRANBERRY_workshop/cranberry-workshop-919771f4-gpu/` as the temporary executable reference for generating regression outputs.
- Do not depend on `D_CGRNA` or other local environment variables for installed-package behavior.
- Do not edit generated artifacts such as `docs/_build/`, `dist/`, build caches, or egg-info unless the task is specifically about build output inspection.
- Do not silently change packaged model assets. Any change to force-field XML, H5 parameters, examples, or reference outputs needs provenance notes and tests.

## Working Principles

State assumptions before implementation when a task has scientific, API, packaging, or output-format consequences. If several interpretations are plausible and the wrong choice would be expensive, ask the user.

### Simplicity First

Use the minimum code and public surface that solves the current task. Do not add speculative features, abstractions, compatibility layers, configuration formats, or expert switches before the project has a concrete need for them.

For Cranberry, this means:

- prefer OpenMM-native objects and conventions over custom framework layers
- keep `CranberryForceField` small until repeated workflows justify helpers
- do not expose legacy knobs just because they existed in `OpenMM-CGRNA`
- keep JAX, training, PySAGES, and REMD functionality behind optional extras
- prefer explicit package resources over environment-variable plumbing
- simplify code that grows faster than the behavior it supports

When a simpler design and a more flexible design both satisfy the current phase, choose the simpler design and document the extension point only if it is likely to matter soon.

Prefer the smallest stable API that matches OpenMM conventions. Do not expose legacy tuning knobs, scaling factors, or experimental options unless they are intentionally promoted into the v1 contract.

Make surgical changes. Every changed line should trace to the task, required tests, packaging metadata, or required documentation. Leave unrelated cleanup for a separate task.

Preserve scientific behavior while refactoring. Packaging cleanup, path removal, and API design are welcome; accidental model drift is not.

Use explicit validation and clear errors for user-facing scientific workflows. In this project, unlikely malformed inputs can still deserve good diagnostics because silent mistakes can corrupt simulations.

## Development Loop

For non-trivial implementation work:

1. Check repository state and identify any unrelated local changes.
2. State the local assumptions and success criteria.
3. Make the smallest coherent change.
4. Add or update focused tests when behavior changes.
5. Run the relevant local checks.
6. Update a developer-facing code review report under `docs/dev/progress/`.
7. Wait for explicit user approval before committing.
8. Commit intentionally and push when requested.
9. Confirm CI status when the change is pushed.

Typical local checks:

```bash
conda run -n cranberry-dev python -m pytest -q
conda run -n cranberry-dev sphinx-build -b html docs docs/_build/html
```

Run narrower checks first when iterating, then the full checks before review or commit.

## Code Review Report

Before committing non-trivial code, package, CLI, validation, or documentation-architecture changes, create or update an HTML report under `docs/dev/progress/`.

The report should include:

- what changed and why
- important code paths touched
- command outputs or summarized results
- tests and docs checks
- known risks and follow-up work

The report is a developer artifact and should remain excluded from the public Sphinx build unless the user decides otherwise.

## Experiment And Benchmark Mode

Use a stricter experiment loop only for benchmark, performance, or force-field tuning work. Normal package development should not use automatic keep/discard loops.

When benchmark work begins, define:

- fixed input fixtures
- fixed command lines
- fixed metrics
- hardware and platform metadata
- machine-readable results

Do not commit noisy raw logs by default. Commit curated summaries, scripts, and reproducible instructions.

## Phase Success Criteria

Each phase should have concrete success criteria. For example, a force-field construction phase should define:

- which examples are supported
- which reference energies must match
- which platforms are tested locally or in CI
- which CLI and API contracts are stable
- which docs explain the workflow

Avoid accepting vague goals like "make it work" without converting them into checks.
