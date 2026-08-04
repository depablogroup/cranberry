# cranberry-rna

`cranberry-rna` is the installable Python package for CRANBERRY coarse-grained RNA simulations with OpenMM. The distribution name is `cranberry-rna`; the Python import package is `cranberry`.

CRANBERRY is currently in alpha development while the stable v1 workflow is migrated from the legacy `OpenMM-CGRNA` project. The current package already ships the canonical CRANBERRY v1 alpha force-field assets, an OpenMM-native force-field API, CPU energy decomposition, a basic MD runner with restart support, and the first Phase 4 preparation workflow slice.

This repository is intended to be usable by active developers from a fresh clone, including developers who want to run on GPU hardware. The package itself is intentionally small: it does not manage GPU drivers, CUDA toolkits, or OpenMM platform installation. Those come from your local Python environment and system setup.

## Current Scope

Implemented:

- package import as `cranberry`
- packaged model assets: XML plus `cranberry-v1-alpha.1.h5`
- canonical input validation
- `cranberry cg` coarse-graining workflow for atomistic RNA inputs, and `cranberry prepare` canonicalization workflow for already coarse-grained CRANBERRY PDBs
- `CranberryForceField.createSystem()`
- `cranberry energy` and `cranberry.energy.compute_energy()`
- `cranberry md` and `cranberry.md.run_md()`
- default MD outputs: `output.dcd`, `log`, `detailed.log`, `args.json`, `checkpoint.chk`, `final.pdb`
- restart from OpenMM checkpoints

Planned:

- coarse-graining workflow from atomistic RNA inputs
- preparation/canonicalization workflow for already coarse-grained inputs
- optional 5'-phosphate insertion during preparation
- REMD through optional `openmmtools` support, with NetCDF restart and OpenMM-native DCD translation; no MDAnalysis dependency in the public release
- optional JAX/training/PySAGES workflows

## Fresh Setup

The recommended development flow is:

1. Create a dedicated conda environment.
2. Install the package in editable mode with dev extras.
3. Verify the import, CLI, and tests.

For a new environment, clone the repository first, then install from inside that cloned folder:

```bash
conda create -n cranberry-dev python=3.11
conda activate cranberry-dev
git clone https://github.com/yihengwuKP/cranberry.git
cd cranberry
python -m pip install -e ".[dev]"
```

For GPU development, prefer installing OpenMM with conda before the editable Cranberry install so that OpenMM's CUDA platform plugin is resolved by conda:

```bash
conda create -n cranberry-dev -c conda-forge python=3.11 openmm
conda activate cranberry-dev
git clone https://github.com/yihengwuKP/cranberry.git
cd cranberry
python -m pip install -e ".[dev]"
```

If you already cloned the repository, skip `git clone` and replace `cd cranberry` with `cd /absolute/path/to/your/cranberry`. The `cd` step must put your shell in the repository checkout before running `python -m pip install -e ".[dev]"`.

Verify the editable install:

```bash
python -c 'import cranberry; print(cranberry.__file__)'
cranberry --help
python -m pytest -q
sphinx-build -b html docs docs/_build/html
```

The import check should print a path ending in `cranberry/__init__.py` inside your checkout. If it prints `None`, return to the cloned repository folder and rerun `python -m pip install -e ".[dev]"` in the activated environment.

For install troubleshooting, see the [FAQ](https://github.com/yihengwuKP/cranberry/blob/main/docs/faq.md).

If you prefer a CPU-only workflow, this is enough. If you want GPU execution, keep reading.

## GPU Development

The current workflow is sufficient for GPU-enabled development if your environment already exposes a GPU-capable OpenMM build. In practice, that means:

- your driver/runtime stack is installed on the machine
- OpenMM can see the accelerator platform you want to use
- you select that platform explicitly or allow OpenMM to choose it

CRANBERRY does not add an extra GPU-specific install layer. Once OpenMM is installed correctly, the same editable install works for CPU and GPU runs. If `python -m pip install -e ".[dev]"` installs OpenMM for you, that may be sufficient for CPU use, but GPU users should prefer preinstalling OpenMM from conda-forge:

```bash
conda create -n cranberry-dev -c conda-forge python=3.11 openmm
conda activate cranberry-dev
python -m pip install -e ".[dev]"
```

Before running Cranberry on CUDA, confirm that OpenMM registered the CUDA platform in this environment:

```bash
python - <<'PY'
from openmm import Platform
print([Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())])
PY
```

If that list does not include `CUDA`, debug the OpenMM installation before debugging Cranberry.

To run on GPU hardware, pass the OpenMM platform you want:

```bash
cranberry energy cranberry/data/examples/2ntCG_cg_vs_conect.pdb --platform CUDA
cranberry md cranberry/data/examples/2ntCG_cg_vs_conect.pdb \
  --steps 1000 \
  --output-dir md-out \
  --platform CUDA
```

If you want OpenMM to pick the available platform itself, use:

```bash
cranberry md cranberry/data/examples/2ntCG_cg_vs_conect.pdb \
  --steps 1000 \
  --output-dir md-out \
  --platform default
```

Important current limitations:

- CI is CPU-only, so GPU behavior is not validated in GitHub Actions.
- The package defaults to CPU for predictable local and CI behavior, and the MD default timestep is 5 fs.
- GPU restart behavior still uses the same OpenMM checkpoint contract, so the restart checkpoint must be compatible with the selected platform and model.

## Quickstart

Inspect the package and default model:

```bash
cranberry inspect
cranberry inspect forcefield
cranberry inspect data
```

Prepare a canonical coarse-grained input. `cranberry cg` coarse-grains an atomistic RNA PDB into canonical CRANBERRY CG form, while `cranberry prepare` canonicalizes an already coarse-grained input. The default `prepare` path is a validation-only check: if no terminal-phosphate insertion is requested, Cranberry reports that nothing needs to be changed and does not write a new file. Use `--add-terminal-phosphate` only when you want Cranberry to add missing 5'-terminal phosphate context near a chain end for sugar-puckering analysis in the coarse-grained model:

```bash
cranberry prepare cranberry/data/examples/2ntCG_cg_vs_conect.pdb --add-terminal-phosphate
```

Validate a canonical coarse-grained input PDB:

The example filenames follow the pattern `*_cg_vs_conect.pdb`: `cg` means coarse-grained, `vs` means virtual sites, and `conect` means the PDB includes `CONECT` bond records.

```bash
cranberry inspect input cranberry/data/examples/2ntCG_cg_vs_conect.pdb
```

Compute total and decomposed energies on CPU:

```bash
cranberry energy cranberry/data/examples/2ntCG_cg_vs_conect.pdb --platform CPU
```

Run a short CPU MD simulation:

```bash
cranberry md cranberry/data/examples/2ntCG_cg_vs_conect.pdb \
  --steps 1000 \
  --n-record 10 \
  --output-dir md-out \
  --platform CPU
```

Restart from the generated checkpoint. Restart appends to `output.dcd`, `log`, and `detailed.log` when they exist; if any are missing, Cranberry warns and creates them starting from the checkpoint step. `args.json` records latest run metadata, and distinct previous metadata is archived under `args_history/`.

```bash
cranberry md cranberry/data/examples/2ntCG_cg_vs_conect.pdb \
  --steps 1000 \
  --restart-from md-out/checkpoint.chk \
  --output-dir md-out \
  --platform CPU
```

On Slurm GPU nodes, pack several independent MD runs into one GPU allocation with CUDA MPS:

```bash
scripts/hpc/submit_md_mps_pack_example_slurm.sh \
  --pdb system_a_cg_vs_conect.pdb \
  --pdb system_b_cg_vs_conect.pdb \
  --steps 100000 \
  --platform CUDA \
  --output-root runs/md-pack \
  --account my_account \
  --partition gpu
```

See `docs/tutorials/packed-md-hpc.md` for the full prepare-and-submit workflow.

## Python API

The API follows OpenMM conventions and accepts OpenMM unit quantities where appropriate.

```python
from openmm import app, unit
from cranberry import CranberryForceField

pdb = app.PDBFile("input_cg_vs_conect.pdb")
forcefield = CranberryForceField()
system = forcefield.createSystem(
    pdb.topology,
    positions=pdb.positions,
    temperature=298 * unit.kelvin,
    salt_concentration=150 * unit.millimolar,
)
```

For the high-level MD runner:

```python
from cranberry.md import run_md

result = run_md(
    "input_cg_vs_conect.pdb",
    steps=1000,
    output_dir="md-out",
    platform="CPU",
)
print(result.checkpoint_path)
```

## Tests And Docs

Run the default CPU test suite:

```bash
python -m pytest -q
```

Build the Sphinx docs:

```bash
sphinx-build -b html docs docs/_build/html
```

Build package artifacts:

```bash
python -m build
```

## Documentation

Primary documentation lives under `docs/`. Developer-only design notes and code-review reports live under `docs/dev/` and are intentionally excluded from the public documentation narrative.

Start here:

- [Installation](https://github.com/yihengwuKP/cranberry/blob/main/docs/installation.md)
- [Quickstart](https://github.com/yihengwuKP/cranberry/blob/main/docs/quickstart.md)
- [API reference](https://github.com/yihengwuKP/cranberry/blob/main/docs/reference/api.md)
- [CLI reference](https://github.com/yihengwuKP/cranberry/blob/main/docs/reference/cli.md)
- [Outputs reference](https://github.com/yihengwuKP/cranberry/blob/main/docs/reference/outputs.md)
- [Force-field reference](https://github.com/yihengwuKP/cranberry/blob/main/docs/reference/forcefield.md)
- [Benchmarks](https://github.com/yihengwuKP/cranberry/blob/main/docs/benchmarks/index.md)

Tutorials:

- [Energy decomposition](https://github.com/yihengwuKP/cranberry/blob/main/docs/tutorials/energy-decomposition.md)
- [Prepare and run packed MD on HPC](https://github.com/yihengwuKP/cranberry/blob/main/docs/tutorials/packed-md-hpc.md)
- [Prepare and run MD](https://github.com/yihengwuKP/cranberry/blob/main/docs/tutorials/prepare-and-run-md.md)
- [REMD](https://github.com/yihengwuKP/cranberry/blob/main/docs/tutorials/remd.md)

Developer notes:

- [Development environment](https://github.com/yihengwuKP/cranberry/blob/main/docs/dev/development-environment.md)
- [Project program](https://github.com/yihengwuKP/cranberry/blob/main/docs/dev/program.md)
- [Roadmap](https://github.com/yihengwuKP/cranberry/blob/main/docs/dev/roadmap.md)
- [v1 plan](https://github.com/yihengwuKP/cranberry/blob/main/docs/dev/cranberry-v1-plan.md)
- [Migration from OpenMM-CGRNA](https://github.com/yihengwuKP/cranberry/blob/main/docs/dev/migration-from-openmm-cgrna.md)
- [Reference output generation](https://github.com/yihengwuKP/cranberry/blob/main/docs/dev/reference-output-generation.md)
- [Next Codex handoff](https://github.com/yihengwuKP/cranberry/blob/main/docs/dev/next-codex-handoff.md)
- [ADR index](https://github.com/yihengwuKP/cranberry/blob/main/docs/dev/adr/README.md)

## License

MIT.
