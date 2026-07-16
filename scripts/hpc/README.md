# Cranberry HPC Launchers

These scripts are convenience launchers for CUDA MPS and MPI runs. They assume you have activated an environment with `cranberry-rna`, OpenMM, OpenMMTools, and for REMD/MPI, `mpi4py`.

## REMD With MPI And CUDA MPS

Minimal first run:

```bash
PDB_PATH=/path/to/structure_cg_vs_conect.pdb \
OUTPUT_DIR=runs/melting_example \
N_REPLICAS=8 \
N_MPI_RANKS=8 \
NUM_GPUS=1 \
scripts/hpc/run_cranberry_remd_mpi_mps.sh
```

Useful controls:

- `NSTEPS`, `SWAP_STEPS`, `N_RECORD`, `N_REPLICAS`, `T_MIN`, `T_MAX`, `SALT`, and `STEP_SIZE` map directly to `cranberry remd` options.
- `N_RECORD` defaults to `10000`, matching the old `melting_v2_cedric` benchmark wrapper. For the old 500000-step / 5000-swap-step runs, and for short 50000-step / 1000-swap-step smoke runs, this yields a checkpoint interval of 1 REMD iteration.
- `N_ANALYSIS` defaults to `10`, matching the old 500000-step / 5000-swap-step melting runs where OpenMMTools selected an online-analysis interval of 10. Set `N_ANALYSIS=0` to disable online PyMBAR/JAX analysis.
- `JAX_PLATFORM_NAME` is not forced. Leave it unset to let JAX use its default backend, or set `JAX_PLATFORM_NAME=cpu` if online analysis should avoid the GPU.
- `ALT_PDB_PATH` seeds alternating initial replicas on a fresh run.
- `RESTART=1` restarts automatically from `OUTPUT_DIR/output.nc` when it exists.
- `PLATFORM_PROPERTIES="Precision=mixed"` passes repeated `--platform-property KEY=VALUE` options to Cranberry.
- `MPI_LAUNCHER=srun` uses `srun -n`; otherwise the launcher uses `mpirun --oversubscribe --bind-to none`.
- `STREAM_LOG=0` is the default and captures MPI output as `stdout.log` and `stderr.log`, matching the old worker harness. Set `STREAM_LOG=1` for live terminal output through `tee`.

The rank wrapper maps local MPI ranks to GPUs using `NUM_GPUS` and `RANKS_PER_GPU`. If `CUDA_VISIBLE_DEVICES` is already set by the scheduler, the wrapper treats it as the parent GPU list and gives each rank one entry from that list.

Each run records `launcher_env.txt`, `preflight.log`, `nvidia_smi_start.txt`, and if the wrapper exits normally, `nvidia_smi_end.txt` and `timing_summary.txt`. Use these to compare GPU model, JAX backend, MPI rank layout, and Slurm CPU allocation across nodes.

The legacy 500000-step C ACAG L40S benchmark was submitted with `--cpus-per-task=32` for 8 MPI ranks. An interactive allocation with only 4 CPUs for 8 ranks can be much slower even when CUDA MPS and GPU JAX are working.

If a bare `python -c "from mpi4py import MPI"` hangs or fails inside an interactive Slurm GPU shell, source `scripts/hpc/mpi_runtime_env.sh` before testing. The helper disables the missing munge PMIx security component and clears stale PMI/PMIx/OMPI rank variables from the parent interactive shell before Cranberry launches its own `mpirun`.

Legacy C ACAG comparison:

```bash
PDB_PATH=/projects/rps/jjd8110/depablolab/cedric/04-cranberry/melting/seed/cacag_cg_vs_conect.pdb \
ALT_PDB_PATH=/projects/rps/jjd8110/depablolab/cedric/04-cranberry/melting/seed/cacag_extended_cg_vs_conect.pdb \
OUTPUT_DIR=runs/cacag_legacy_match_500k \
NSTEPS=500000 \
SWAP_STEPS=5000 \
N_RECORD=10000 \
N_ANALYSIS=10 \
N_REPLICAS=8 \
N_MPI_RANKS=8 \
NUM_GPUS=1 \
USE_MPS=1 \
USE_PBC=1 \
BOX_PADDING=2 \
T_MIN=280 \
T_MAX=380 \
SALT=1000 \
STREAM_LOG=0 \
MPI_EXTRA_ARGS="--mca pml ob1 --mca mtl ^ofi --mca shmem_mmap_enable_nfs_warning 0" \
scripts/hpc/run_cranberry_remd_mpi_mps.sh
```

For an interactive benchmark allocation on the same node class, request enough CPU for the rank count:

```bash
salloc --partition=l40s_courant --gres=gpu:1 --cpus-per-task=32 --mem=32G --time=2:00:00
```

## C ACAG 5 us Production

Submit the 5 us C ACAG PBC production run:

```bash
scripts/hpc/submit_cacag_5us_pbc_l40s.sh
```

The submitter targets `1000000000` steps at 5 fs, or `200000` REMD iterations with `SWAP_STEPS=5000`. It submits to the Torch account by default:

- `ACCOUNT=torch_pr_109_courant`
- `CONSTRAINT=l40s`
- `PARTITION=l40s_courant`
- `--gres=gpu:1`
- `--cpus-per-task=32`
- `--time=48:00:00`

Each batch attempt runs through `timeout -k 10m 47h`, leaving roughly one hour for checkpoint flushing and Slurm-side restart handling before the 48 h limit. On timeout exit code `124` or kill exit code `137`, the script reads `output.nc`, computes remaining REMD iterations, adjusts the per-attempt `N_RECORD` and `N_ANALYSIS` to preserve the full-run logging cadence, and resubmits itself only if the stored iteration count is still below the target. If the stored iteration is at or above the target, it logs completion and does not submit another job.

When `N_ANALYSIS` is not set, the production submitter mirrors the old melting v2 `pt_online_analysis_interval=null` behavior:

```text
target_analysis_interval = max(min(target_iterations / 10, 10 * target_checkpoint_interval), 1)
```

For the default 5 us run, that gives `target_checkpoint_interval=20`, `target_analysis_interval=200`, and `full_run_n_analysis=1000`, or one online-analysis update about every 5 ns per replica. Set `N_ANALYSIS=0` to disable online analysis, set `N_ANALYSIS=<count>` to request a full-run analysis count, or set `TARGET_ANALYSIS_INTERVAL=<iterations>` to control the interval directly.

The submitter exports the original Cranberry checkout path into the Slurm job as `CRANBERRY_REPO_ROOT`, so restarts call the helper in this repo even when Slurm executes a spooled copy of the sbatch script.

The submitter also resolves `CONDA_ENV=cranberry-dev` to `$HOME/.conda/envs/cranberry-dev` by default, prepends that environment's `bin` directory to `PATH` inside the batch job, and passes its Python as `PYTHON_BIN`. Each batch attempt runs a submit-side preflight for `openmmtools`, `mpi4py`, and `netCDF4` before reading `output.nc` or launching MPI. Override `CONDA_ENV`, `CONDA_ENV_PATH`, or `PYTHON_BIN` if the environment lives somewhere else.

Useful overrides:

```bash
OUTPUT_DIR=/path/to/run \
ACCOUNT=torch_pr_109_courant \
CONSTRAINT=l40s \
PARTITION=l40s_courant \
CONDA_ENV=cranberry-dev \
scripts/hpc/submit_cacag_5us_pbc_l40s.sh
```

Restart test with a 10 minute internal timeout and one automatic resubmission:

```bash
OUTPUT_DIR=runs/cacag_restart_test_10min \
JOB_NAME=cacag_restart_test \
TARGET_STEPS=2000000 \
RUN_TIMEOUT=10m \
KILL_AFTER=2m \
MAX_RESUBMITS=1 \
scripts/hpc/submit_cacag_5us_pbc_l40s.sh
```

This short target is intentionally longer than one 10 minute batch attempt, so the first job should exit through `timeout`, archive `std.out` to `std.out.prev`, and submit one follow-up job. Remove `MAX_RESUBMITS=1` or increase it if you want the test run to continue until completion.

## ggcGCAAgcc 5 us Production Without PBC

Submit the matched no-PBC production run:

```bash
scripts/hpc/submit_ggcGCAAgcc_5us_nopbc_l40s.sh
```

This wrapper uses the same restart-tested L40S production machinery as the C ACAG helper, but sets:

- `JOB_NAME=ggcGCAAgcc_5us_nopbc`
- `OUTPUT_DIR=runs/ggcGCAAgcc_nopbc_5us_l40s`
- `PDB_PATH=/projects/rps/jjd8110/depablolab/cedric/04-cranberry/melting/seed/ggcGCAAgcc_cg_vs_conect.pdb`
- `ALT_PDB_PATH=/projects/rps/jjd8110/depablolab/cedric/04-cranberry/melting/seed/ggcGCAAgcc_extended_cg_vs_conect.pdb`
- `USE_PBC=0`
- `T_MIN=310`
- `T_MAX=410`

The target length, restart behavior, L40S resource request, salt, logging cadence, MPI ranks, CUDA MPS setup, and `cranberry-dev` environment resolution match the C ACAG production helper unless overridden. The `310-410 K` temperature range matches the regular `ggcGCAAgcc` entry in `04-cranberry/melting_v2_cedric`.

## Independent MD Runs With CUDA MPS

Pack several independent `cranberry md` jobs onto the visible GPU set:

```bash
PDB_PATHS="/path/a_cg_vs_conect.pdb /path/b_cg_vs_conect.pdb" \
OUTPUT_ROOT=runs/md_pack \
MAX_PARALLEL=4 \
scripts/hpc/run_cranberry_md_mps_pack.sh
```

Useful controls:

- `PDB_PATH` runs a single input; `PDB_PATHS` runs a space-separated set.
- `NSTEPS`, `N_RECORD`, `TEMPERATURE`, `SALT`, `STEP_SIZE`, `MODEL`, and `OPENMM_PLATFORM` map to `cranberry md`.
- `RESTART=1` restarts a packed MD run from its existing `checkpoint.chk`.
- `USE_PBC=1` adds `--periodic --enforce-periodic-output`.
- `PLATFORM_PROPERTIES="Precision=mixed"` passes OpenMM platform properties to each MD run.
