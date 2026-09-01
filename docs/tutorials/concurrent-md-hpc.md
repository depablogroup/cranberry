# Prepare And Run Concurrent MD On HPC

This tutorial shows how to run several independent `cranberry md` jobs concurrently inside one Slurm GPU allocation. The concurrent-MD example is useful for small or moderate systems that do not fully occupy a GPU by themselves. The submitter starts CUDA MPS when available, then launches one plain `cranberry md` command per input PDB in the background.

## Prepare Inputs

`cranberry md` expects canonical CRANBERRY coarse-grained PDB files with virtual-site atoms and `CONECT` records. If you are starting from atomistic RNA PDB files, coarse-grain each input first:

```bash
cranberry cg atomistic_a.pdb --output system_a_cg_vs_conect.pdb
cranberry cg atomistic_b.pdb --output system_b_cg_vs_conect.pdb
cranberry cg atomistic_c.pdb --output system_c_cg_vs_conect.pdb
```

If you already have coarse-grained inputs, inspect them before launching an HPC run:

```bash
cranberry inspect input system_a_cg_vs_conect.pdb
cranberry inspect input system_b_cg_vs_conect.pdb
cranberry inspect input system_c_cg_vs_conect.pdb
```

Use `cranberry prepare --add-terminal-phosphate` only when you intentionally want Cranberry to add a missing 5'-terminal phosphate bead:

```bash
cranberry prepare system_a_cg_vs_conect.pdb \
  --add-terminal-phosphate \
  --output system_a_prepared_cg_vs_conect.pdb
```

## Submit Concurrent MD

Submit one Slurm job and pass one `--pdb` option per independent run:

```bash
scripts/hpc/submit_md_mps_concurrent_example_slurm.sh \
  --pdb system_a_cg_vs_conect.pdb \
  --pdb system_b_cg_vs_conect.pdb \
  --pdb system_c_cg_vs_conect.pdb \
  --run-name system_a \
  --run-name system_b \
  --run-name system_c \
  --steps 100000 \
  --n-record 1000 \
  --platform CUDA \
  --output-root runs/md-concurrent \
  --account my_account \
  --partition gpu
```

The submitter requests one GPU by default. Inside the allocation it runs commands equivalent to:

```bash
cranberry md system_a_cg_vs_conect.pdb \
  --steps 100000 \
  --n-record 1000 \
  --platform CUDA \
  --output-dir runs/md-concurrent/000_system_a &

cranberry md system_b_cg_vs_conect.pdb \
  --steps 100000 \
  --n-record 1000 \
  --platform CUDA \
  --output-dir runs/md-concurrent/001_system_b &

cranberry md system_c_cg_vs_conect.pdb \
  --steps 100000 \
  --n-record 1000 \
  --platform CUDA \
  --output-dir runs/md-concurrent/002_system_c &

wait
```

CUDA MPS is on by default. Disable it only for debugging or for systems where the CUDA MPS control daemon is unavailable:

```bash
scripts/hpc/submit_md_mps_concurrent_example_slurm.sh \
  --no-mps \
  --pdb system_a_cg_vs_conect.pdb \
  --pdb system_b_cg_vs_conect.pdb \
  --steps 100000 \
  --platform CUDA \
  --output-root runs/md-concurrent \
  --account my_account \
  --partition gpu
```

## Preview A Submission

Use `--dry-run` to inspect the generated `sbatch` command without submitting:

```bash
scripts/hpc/submit_md_mps_concurrent_example_slurm.sh \
  --dry-run \
  --pdb system_a_cg_vs_conect.pdb \
  --pdb system_b_cg_vs_conect.pdb \
  --steps 100000 \
  --platform CUDA \
  --output-root runs/md-concurrent \
  --account my_account \
  --partition gpu
```

## Outputs

Each PDB gets a separate output directory under `--output-root`:

```text
runs/md-concurrent/
  launcher_env.txt
  mps-log/
  000_system_a/
    launcher_cmd.txt
    md.log
    output.dcd
    log
    detailed.log
    args.json
    checkpoint.chk
    final.pdb
  001_system_b/
    ...
```

`launcher_cmd.txt` records the exact `cranberry md` command for that run. `md.log` captures stdout and stderr from that process. The other files are the standard `cranberry md` outputs.

## Periodic Runs

Pass `--periodic` when all concurrent runs should use explicit periodic boundary conditions:

```bash
scripts/hpc/submit_md_mps_concurrent_example_slurm.sh \
  --pdb duplex_a_cg_vs_conect.pdb \
  --pdb duplex_b_cg_vs_conect.pdb \
  --steps 100000 \
  --platform CUDA \
  --periodic \
  --box-padding 3.0 \
  --output-root runs/md-concurrent-periodic \
  --account my_account \
  --partition gpu
```

This adds `--periodic --box-padding 3.0 --enforce-periodic-output` to each `cranberry md` command.

## Practical Notes

- The intended example mode is one Slurm GPU with MPS enabled.
- By default, every input PDB launches concurrently: three PDBs means three `cranberry md` processes.
- Choose the number of PDBs per submission based on GPU memory, CPU cores, and observed throughput.
- The generic example does not implement production restart/resubmission policy. For long campaign runs that may hit walltime, use or adapt a site-specific production wrapper.
