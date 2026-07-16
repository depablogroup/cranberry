# REMD

REMD support lives behind the optional `remd` extra because it depends on `openmmtools`. The public release does not require MDAnalysis; the NetCDF-to-DCD helper uses OpenMM-native writing instead.

REMD defaults to no-overwrite behavior, matching OpenMMTools. Use `--overwrite` only when you intentionally want to replace an existing NetCDF output set.

A tiny CPU run can be started with total MD integration steps plus the number of MD steps between exchange attempts:

```bash
cranberry remd cranberry/data/examples/2ntCG_cg_vs_conect.pdb   --steps 1000   --swap-steps 100   --n-record 10   --temperature-ladder 298 318   --output-dir remd-out
```

`--steps` is the total MD integration step count. Cranberry derives the OpenMMTools iteration count as `max(1, steps // swap_steps)`. `--n-record` controls the NetCDF checkpoint density by deriving a checkpoint interval in REMD iterations. `--n-analysis 0` disables OpenMMTools online analysis; values greater than zero derive an online-analysis interval. For example, `--steps 500000 --swap-steps 5000 --n-analysis 10` produces an online-analysis interval of 10 REMD iterations.

Online analysis is an OpenMMTools/PyMBAR convenience path, not part of the MD propagation itself. PyMBAR uses JAX, so online analysis can add noticeable overhead and may allocate GPU memory separately from OpenMM when JAX chooses a CUDA backend. For the lowest-risk CUDA REMD run, use `--n-analysis 0`. To reproduce the old melting-style path with JAX GPU online analysis available, use a nonzero `--n-analysis` and leave `JAX_PLATFORM_NAME` unset so JAX can choose its default backend. If you want online analysis but do not want JAX competing with OpenMM on the GPU, launch with `JAX_PLATFORM_NAME=cpu`. Cranberry records `online_analysis_interval`, `jax_platform_name_env`, JAX version/backend metadata when online analysis is active, MPI metadata, CUDA/MPS environment values, and any OpenMM platform properties in `args.json`.

Both `cranberry md` and `cranberry remd` accept repeated OpenMM platform properties:

```bash
cranberry remd input.pdb --steps 500000 --swap-steps 5000 --platform CUDA --platform-property Precision=mixed
```

For cluster launch helpers that mirror the old CUDA MPS/MPI melting workflow, see `scripts/hpc/README.md`.

Use `--extra-start-pdb` to provide an additional canonical CG PDB whose coordinates seed alternating initial replicas, useful for melting-style starts where the main PDB and an extra starting structure should both be represented.

The primary REMD restart artifact is `output.nc`. Cranberry also writes `args.json` for provenance. To inspect the trajectory as DCD files:

```bash
cranberry remd-extract remd-out/output.nc cranberry/data/examples/2ntCG_cg_vs_conect.pdb --by-replica --output-dir remd-out
cranberry remd-extract remd-out/output.nc cranberry/data/examples/2ntCG_cg_vs_conect.pdb --by-temperature --output-dir remd-out
```

`--by-replica` writes `output_0.dcd`, `output_1.dcd`, and so on. `--by-temperature` writes `output_T0.dcd`, `output_T1.dcd`, and a companion `output_temperature_labels.txt` mapping each T index to the stored thermodynamic-state temperature.
