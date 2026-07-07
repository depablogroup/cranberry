# Cranberry Roadmap

This note summarizes the working plan for the `cranberry-rna` package. It is the short version of the longer design record in `docs/dev/cranberry-v1-plan.md`.

## Where We Are

The repository already has:

- a clean installable package layout
- the `cranberry` import package and `cranberry-rna` distribution identity
- packaged canonical alpha model assets
- `CranberryForceField` with an OpenMM-style API
- CPU energy decomposition
- a basic MD runner with restart handling and output bookkeeping
- developer-facing docs, tests, and phase reports

## Roadmap

### Phase 1: Package Foundation

Done. This covered the new repo, editable development setup, docs skeleton, CI, contributor guidance, and developer operating notes.

### Phase 2: Canonical Force Field

Done. This covered packaged model assets, canonical validation, `inspect`, and energy decomposition on the canonical fixtures.

### Phase 3: Basic MD

Done. This covered `cranberry md`, default outputs, checkpoint restart behavior, argument metadata, and restart compatibility checks.

### Phase 4: Preparation And Coarse-Graining

Next. This should add the preparation workflow, canonical coarse-graining, and the option to add a 5'-terminal phosphate when needed.

### Phase 5: REMD

After Phase 4. Add `remd` as an optional feature, with `openmmtools` kept out of the base install.

### Phase 6: Advanced Extras

After the core workflows are stable. Add optional JAX, training, and PySAGES/enhanced sampling support.

### Phase 7: Hardening

In parallel with the later phases. Finish developer-facing force-field construction notes, benchmark scaffolding, release discipline, and any documentation needed for a public v1.

### Phase 8: v1 Release

Finalize the package versioning sequence, freeze the public scientific/API contract, publish the stable v1 model, and release `1.0.0`.

## v1 Functional Scope

The v1 release should provide:

- installation by `pip` and conda-friendly packaging
- OpenMM-native force-field loading through `CranberryForceField`
- canonical coarse-grained RNA input validation
- preparation/coarse-graining workflow
- basic MD
- energy decomposition
- restart support
- CPU-first tests and CI
- optional extras for REMD, JAX, training, and PySAGES

## Beyond v1

Later work should focus on:

- broader benchmark infrastructure
- more complete publication-quality documentation
- any additional scientific model revisions as versioned model updates
- optional workflow expansion around enhanced sampling and training

