# Cranberry v1 Plan

This developer-only document captures the agreed design for refactoring the legacy `OpenMM-CGRNA` project into a clean installable Python package.

## Package Shape

- Create a new project at `CRANBERRY/cranberry/`, parallel to `OpenMM-CGRNA/`.
- Distribution name: `cranberry-rna`.
- Import package name: `cranberry`.
- Use a flat package layout:

  ```text
  cranberry/
    pyproject.toml
    cranberry/
      __init__.py
      data/
    tests/
    docs/
    benchmarks/
  ```

- Runtime package data must live inside the inner `cranberry/` import package so it is installed by pip/conda.
- Use Sphinx with MyST Markdown for documentation.

## Scientific Scope

- v1 supports the canonical CRANBERRY model only.
- 5SPN and RACER are not public v1 scope.
- Canonical input requires coarse-grained RNA PDBs with virtual sites and `CONECT`.
- Coarse-graining is part of the workflow.
- Public force-field object: `CranberryForceField`.
- Follow OpenMM style where appropriate, including `createSystem`.

## Packaging And Dependencies

- Base install supports package import, structure preparation, force-field loading, plain MD, energy decomposition, and tests.
- Optional extras:
  - `remd`: `openmmtools` and replica exchange support.
  - `jax`: JAX-related functionality.
  - `train`: training dependencies.
  - `pysages`: enhanced sampling dependencies.
  - `dev`: testing, docs, build tooling.
- REMD/PT is not part of the base dependency set because `openmmtools` is heavy.
- Remove `D_CGRNA`; use package resources for bundled data.

## Runtime Data

- Put packaged runtime data under `cranberry/cranberry/data/`.
- Rename canonical XML to `cranberry.xml`.
- Build a canonical `cranberry-v1.h5` parameter bundle. This should absorb finalized angular terms and other model constants so angle scaling is not exposed as API.
- Ship these fixtures as both examples and tests:
  - `157d`
  - `1l2x`
  - `2ntCG`
  - `1zih`
- Full training datasets remain external. Data fetching can be added later.

## Versioning

- Use alpha, beta, and release-candidate versions before the paper-final model:
  - `cranberry-rna 1.0.0a1` with `cranberry-v1-alpha.1`
  - `cranberry-rna 1.0.0b1` with `cranberry-v1-beta.1`
  - `cranberry-rna 1.0.0rc1` with `cranberry-v1-rc.1`
  - `cranberry-rna 1.0.0` with `cranberry-v1`
- Scientific parameter/model changes happen only in minor or major releases, not patch releases.
- Patch releases are for software fixes that do not change the default scientific model.

## Public API

- Primary API is OpenMM-native:

  ```python
  from openmm import app
  from cranberry import CranberryForceField

  pdb = app.PDBFile("input_cg_vs_conect.pdb")
  ff = CranberryForceField("cranberry-v1")
  system = ff.createSystem(
      pdb.topology,
      positions=pdb.positions,
      temperature=298,
      salt_concentration=150,
  )
  ```

- Also provide path-based helpers for convenience.
- `CranberryForceField` constructs `System` objects only; it does not directly run MD.
- High-level run helpers use dataclass configs.
- CLI fields map directly to config fields.
- Defer YAML configs.
- Do not support old `core.*` imports in the new package.
- Provide migration docs rather than a legacy compatibility API.

## CLI

Planned commands:

- `cranberry prepare`
- `cranberry cg`
- `cranberry md`
- `cranberry remd`
- `cranberry energy`
- `cranberry inspect`

Rules:

- `cranberry md` requires prepared canonical CG input.
- Provide an option in the preparation workflow to add a 5'-terminal phosphate when needed. Current terminal nucleotides are C3'-endo-only and the sugar-pucker model requires phosphate context.
- Strict validation is default.
- `--outdir` is supported for run commands and defaults to `.`.
- Platform is selectable; default lets OpenMM choose. Tests use CPU.
- No seed option in v1.

## Force-Field Semantics

- Use `CranberryForceField`.
- Public force-group names:
  - `bond`
  - `angle`
  - `dihedral`
  - `pucker`
  - `stacking35`
  - `stacking55`
  - `stacking33`
  - `pairing`
  - `wca`
  - `spline`
  - `electrostatic`
- Force-group integer IDs may be consistent internally but are not guaranteed as public API.
- Default force terms:
  - bond
  - angle
  - dihedral
  - sugar pucker
  - stacking35
  - stacking55
  - stacking33
  - pairing
  - wca
  - spline
  - Debye-Huckel/electrostatic
- Temperature is required at system construction because electrostatics are temperature dependent.
- High-level APIs should pass a single temperature to both system construction and integrator setup.
- A temperature mismatch between system parameterization and simulation is an error.
- Defaults:
  - temperature: `298 K`
  - salt concentration: `150 mM`
  - phosphate charge: internal default, not public v1 API
- Do not expose scaling factors, angle scaling, or `base_form` as public v1 API.
- Expert force disabling is allowed; disabled forces are omitted from `detailed.log`.

## Simulation Defaults And Outputs

- Use `LangevinMiddleIntegrator`.
- Preserve the existing length-dependent friction heuristic:

  ```text
  D = 4.58e-10 * N_residues^-0.39 m^2/s
  gamma = R*T / (D * total_mass)
  ```

- MD default outputs:
  - `output.dcd`
  - `log`
  - `detailed.log`
  - `args.json`
  - `checkpoint.chk`
  - `final.pdb`
- `detailed.log` is written by default at the same interval as `log` and `output.dcd`.
- `detailed.log` includes total potential energy plus force-group component energies.
- Keep OpenMM-style quoted headers with units, for example:

  ```text
  #"Step","Time (ps)","Potential Energy (kJ/mole)","bond (kJ/mole)",...
  ```

- `cranberry md` follows OpenMM-style overwrite behavior.
- Phase 3 `cranberry md` accepts explicit `--steps` first. Defer friendlier `--time` plus `--timestep` CLI duration parsing until the basic MD path is stable; the Python API may still accept OpenMM unit quantities.
- `cranberry remd` follows OpenMMTools-style no-overwrite behavior by default; `--overwrite` should be an explicit opt-in.
- REMD default outputs:
  - `output.nc`
  - `output_checkpoint.nc`
- Restart is supported.
- Expose both low-level OpenMM `Simulation` creation and high-level `run_md`.

## Reference Output Generation

During migration, use `CRANBERRY_workshop/cranberry-workshop-919771f4-gpu/` as the temporary executable reference implementation for generating regression outputs. Treat `OpenMM-CGRNA/` as code reference unless explicitly needed. See `docs/dev/reference-output-generation.md`.

## Tests

- Default CI is CPU-only.
- Default tests should be fast and installable-package friendly.
- Include:
  - import without `D_CGRNA`
  - package data lookup
  - topology validation
  - energy regression tests for `157d`, `1l2x`, `2ntCG`, and `1zih`
  - tiny CPU MD smoke test
- Mark heavier or optional tests separately:
  - REMD/openmmtools
  - JAX
  - training
  - PySAGES
  - manual/HPC
- Before public alpha/beta release, enable GitHub branch protection or a ruleset that requires the `CI` workflow to pass on `main`. This may require making the repository public or using a GitHub plan that supports required checks on private repositories.

## Benchmarks

- Benchmarks are separate from tests.
- CI may run only a tiny benchmark smoke check.
- Full CPU/GPU benchmarks should be manual or scheduled, because GitHub runners are noisy and GPU availability is unreliable.
- Benchmark results should be machine-readable and suitable for publishing in docs.
- Use the four bundled canonical systems (`157d`, `1l2x`, `2ntCG`, `1zih`) as the first baseline set.
- Publish a benchmark tab under `docs/benchmarks/` with a committed snapshot, raw JSON, and a simple speed-vs-system-size plot.
- Capture the local RTX 2060/CUDA machine as an initial baseline, then add later CPU, CUDA, and MPS series for cross-cluster comparison.
- Add multi-process MPS demonstrations later as a distinct benchmark kind rather than mixing them into the first one-process MD baseline.
- Add REMD throughput once the REMD workflow exists, but keep that as a separate series from MD.
- Before v1 release, add appropriate developer-facing documentation and comments for the force-field construction internals, especially `add_bond`, `add_angle`, `add_dihedral`, `add_sugar_pucker`, stacking, pairing, WCA, spline, and electrostatics. These notes should explain the scientific formula mapping from `OpenMM-CGRNA` to Cranberry without making the public docs too implementation-heavy.

## Documentation

The top-level README should include the canonical Cranberry citation before public v1, alongside the OpenMM/openmmtools citations where relevant.

Use Markdown-first docs that can be built into a Sphinx website:

```text
docs/
  index.md
  installation.md
  quickstart.md
  tutorials/
    prepare-and-run-md.md
    energy-decomposition.md
    remd.md
  reference/
    api.md
    cli.md
    outputs.md
    forcefield.md
  design/
    cranberry-v1-plan.md
    migration-from-openmm-cgrna.md
  benchmarks/
    index.md
```
