# cranberry-rna

`cranberry-rna` provides the CRANBERRY coarse-grained RNA model and OpenMM-based workflows for structure preparation, energy evaluation, molecular dynamics (MD), and replica-exchange molecular dynamics (REMD).

The package is currently an alpha release tied to the published CRANBERRY model assets. Python 3.11 is supported.

## Capabilities

- Coarse-grain atomistic RNA PDB structures into the canonical CRANBERRY representation.
- Validate and canonicalize prepared coarse-grained structures.
- Build OpenMM `System` objects with `CranberryForceField`.
- Compute total and force-group-resolved energies.
- Run CPU or GPU MD with checkpoints, restart, detailed logs, and DCD output.
- Run OpenMMTools parallel tempering with optional MPI, periodic boundaries, restart, and replica- or temperature-indexed DCD extraction.
- Use packaged force-field assets and reference structures.

## Installation

The recommended installation is a GPU-capable conda environment. OpenMM's conda-forge package selects a CUDA build compatible with the available NVIDIA driver; you do not need to install a separate system CUDA toolkit. An up-to-date vendor driver is still required. See the [OpenMM installation guide](https://docs.openmm.org/latest/userguide/application/01_getting_started.html) for CUDA-version-specific options.

```bash
conda create -n cranberry -c conda-forge python=3.11 pip openmm h5py
conda activate cranberry
git clone https://github.com/yihengwuKP/cranberry.git
cd cranberry
python -m pip install --no-deps -e .
python -m openmm.testInstallation
```

Use `--platform CUDA` for GPU runs after the self-test reports a CUDA platform:

```bash
cranberry energy cranberry/data/examples/2ntCG_cg_vs_conect.pdb --platform CUDA
cranberry md cranberry/data/examples/2ntCG_cg_vs_conect.pdb --steps 1000 --platform CUDA --output-dir md-out
```

For a CPU-only installation, use pip's OpenMM package and select `--platform CPU`:

```bash
conda create -n cranberry-cpu python=3.11 pip
conda activate cranberry-cpu
python -m pip install openmm
git clone https://github.com/yihengwuKP/cranberry.git
cd cranberry
python -m pip install -e .
python -m openmm.testInstallation
cranberry energy cranberry/data/examples/2ntCG_cg_vs_conect.pdb --platform CPU
```

For REMD, install OpenMMTools and MPI in the active environment, then install Cranberry's optional dependency boundary:

```bash
conda install -c conda-forge openmmtools openmpi mpi4py h5py
python -m pip install --no-deps -e ".[remd]"
```

REMD can run on CPU or GPU. Use `--platform CPU` for a CPU-only run, `--platform CUDA` for CUDA, and read the [REMD tutorial](docs/tutorials/remd.md) before using periodic or melting-style workflows. The [MD tutorial](docs/tutorials/prepare-and-run-md.md) covers CPU/GPU MD, checkpoints, restarts, and output files. See [installation](docs/installation.md) and the [FAQ](docs/faq.md) for platform troubleshooting.

## Quick Start

Inspect the installed model and a prepared structure:

```bash
cranberry inspect forcefield
cranberry inspect input cranberry/data/examples/157d_cg_vs_conect.pdb
```

Coarse-grain an atomistic RNA structure and evaluate its energy:

```bash
cranberry cg atomistic-rna.pdb
cranberry energy atomistic-rna_cg_vs_conect.pdb --platform CPU
```

Run MD:

```bash
cranberry md atomistic-rna_cg_vs_conect.pdb \
  --steps 100000 \
  --platform CPU \
  --output-dir md-run
```

On Slurm GPU nodes, run several independent MD jobs concurrently in one GPU allocation with CUDA MPS:

```bash
scripts/hpc/submit_md_mps_concurrent_example_slurm.sh \
  --pdb system_a_cg_vs_conect.pdb \
  --pdb system_b_cg_vs_conect.pdb \
  --steps 100000 \
  --platform CUDA \
  --output-root runs/md-concurrent \
  --account my_account \
  --partition gpu
```

See `docs/tutorials/concurrent-md-hpc.md` for the full prepare-and-submit workflow.

Run periodic REMD with folded and extended starting conformations:

```bash
cranberry remd cranberry/data/examples/ggcGCAAgcc_cg_vs_conect.pdb \
  --extra-start-pdb cranberry/data/examples/ggcGCAAgcc_extended_cg_vs_conect.pdb \
  --periodic \
  --steps 100000 \
  --output-dir remd-run
```

Extract trajectories by thermodynamic temperature:

```bash
cranberry remd-extract remd-run/output.nc \
  cranberry/data/examples/ggcGCAAgcc_cg_vs_conect.pdb \
  --output-dir remd-run \
  --by-temperature
```

## Python API

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

## Input Model

Simulation commands require canonical CRANBERRY coarse-grained PDB files containing supported RNA residues, the expected beads and virtual sites, and `CONECT` records. Use `cranberry cg` for atomistic input and `cranberry inspect input` before simulation.

Single-chain workflows are nonperiodic by default. Enable explicit periodic boundary conditions with `--periodic` for workflows such as duplex melting.

## Outputs

MD runs write a DCD trajectory, state and force-group logs, checkpoint, final PDB, and `args.json`. REMD runs use OpenMMTools NetCDF storage as the restart source of truth and can extract trajectories by replica or temperature.

The [output reference](docs/reference/outputs.md) describes filenames, restart behavior, and provenance metadata.

## Documentation

- [Quick start](docs/quickstart.md)
- [Prepare and run MD](docs/tutorials/prepare-and-run-md.md)
- [Prepare and run concurrent MD on HPC](docs/tutorials/concurrent-md-hpc.md)
- [Energy decomposition](docs/tutorials/energy-decomposition.md)
- [REMD](docs/tutorials/remd.md)
- [API reference](docs/reference/api.md)
- [CLI reference](docs/reference/cli.md)
- [Force-field reference](docs/reference/forcefield.md)
- [Benchmarks](docs/benchmarks/index.md)
- [Data provenance](cranberry/data/README.md)

## Citation

Please cite:

> Y. Wu, R. Alessandri, A. E. Coraor, X. Peng, P. F. Zubieta Rico, K. Liebl, K. Trinh, T. R. Sosnick, and J. J. de Pablo, "CRANBERRY: An RNA Dynamics Model with Sugar Puckering and Noncanonical Base Pairing," bioRxiv (2026). https://doi.org/10.64898/2026.01.12.699131

Machine-readable citation metadata are provided in [CITATION.cff](CITATION.cff).

## License

CRANBERRY code, model assets, and project-authored fixtures are released under the MIT License. PDB-derived inputs originate from the wwPDB archive under CC0. See [LICENSE](LICENSE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and the [data provenance record](cranberry/data/README.md).

Development instructions are in [CONTRIBUTING.md](CONTRIBUTING.md).
