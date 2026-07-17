# Benchmark Results

Raw benchmark summaries live here. Commit compact YAML summaries and small reports; leave raw run directories untracked unless a specific artifact is needed for diagnosis.

## Local RTX 2060 CUDA/MPS snapshot

Fixture: `ggcGCAAgcc_cg_vs_conect.pdb`, 20,000 MD steps, CUDA platform, NVIDIA GeForce RTX 2060.

For independent MD runners, `per-runner speed` is the mean speed reported by each runner log and `aggregate speed` is the sum across concurrent runners. The wall-time aggregate includes OpenMM context creation and CLI overhead, so it is lower than steady-state log speed.

| Mode | Processes/runners | Mean GPU util | Per-runner log speed | Aggregate log speed | Aggregate wall speed | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| single MD CUDA | 1 | 54.47% | 819.00 ns/day | 819.00 ns/day | 450.57 ns/day | passed |
| MPS MD CUDA | 2 | 64.59% | 752.50 ns/day | 1505.00 ns/day | 1025.52 ns/day | passed |
| MPS MD CUDA | 4 | 70.21% | 670.75 ns/day | 2683.00 ns/day | 1801.92 ns/day | passed |
| MPS MD CUDA | 8 | 66.67% | 551.25 ns/day | 4410.00 ns/day | 2827.31 ns/day | passed |

Source summary: `local_cuda_modes_ggcGCAAgcc.yaml`. Raw run directory: `local_cuda_modes_ggcGCAAgcc_runs/` (not intended for commit).

## REMD online-analysis comparison

OpenMMTools writes `timing_data.ns_per_day` in `output_real_time_analysis.yaml`; that value is aggregate REMD throughput across replicas, not per-rank throughput. The benchmark harness records it as `online_analysis_aggregate_speed_ns_per_day_*` and derives per-runner/per-replica values by division.

The serial REMD rows below use one Python process, so `mpiplus` does not distribute replicas across ranks. They are useful for measuring online-analysis overhead, but they are not the production-shaped multi-rank REMD benchmark.

| Mode | MPI ranks | MPS | JAX platform | n_analysis | Mean GPU util | Aggregate speed | Per-runner speed | Per-replica speed | Wall time | Status |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| serial REMD CUDA | 1 | no | unset | 0 | 64.62% | 469.42 ns/day | 469.42 ns/day | 58.68 ns/day | 171.92 s | passed |
| serial REMD CUDA | 1 | no | unset | 10 | 61.12% | 451.39 ns/day last, 425.98 mean | 451.39 ns/day last | 56.42 ns/day last | 176.89 s | passed |
| MPI REMD CUDA | 8 | yes | unset | 0 | 49.98% | n/a | n/a | n/a | 67.17 s | failed: JAX GPU cuSolver during analysis |
| MPI REMD CUDA | 8 | yes | unset | 10 | 21.06% | n/a | n/a | n/a | 37.61 s | failed: JAX GPU cuSolver during analysis |
| MPI REMD CUDA | 8 | yes | cpu | 0 | 46.46% | 1280.51 ns/day | 160.06 ns/day | 160.06 ns/day | 86.81 s | failed after timing: MPI post-run reporter read |
| MPI REMD CUDA | 8 | yes | cpu | 10 | 44.28% | 1533.26 ns/day last, 1215.65 mean | 191.66 ns/day last | 191.66 ns/day last | 66.31 s | failed after timing: MPI post-run reporter read |

Source summaries: `local_cuda_remd_analysis_compare_ggcGCAAgcc.yaml`, `local_cuda_remd_mpi_mps_analysis_compare_ggcGCAAgcc.yaml`, and `local_cuda_remd_mpi_mps_jaxcpu_analysis_compare_ggcGCAAgcc.yaml`.

For a real 8-temperature GPU REMD run, use MPI plus MPS, for example:

```bash
JAX_PLATFORM_NAME=cpu conda run -n cranberry-dev python benchmarks/benchmark_cuda_modes.py \
  --skip-single-md --skip-mps-md \
  --remd-steps 20000 \
  --remd-n-analysis 0 10 \
  --remd-mpi-ranks 8 \
  --start-mps \
  --output benchmarks/results/local_cuda_remd_mpi_mps_jaxcpu_analysis_compare_ggcGCAAgcc.yaml \
  --work-dir benchmarks/results/local_cuda_remd_mpi_mps_jaxcpu_analysis_compare_ggcGCAAgcc_runs
```

Without `mpirun` and `OPENMMTOOLS_ENABLE_MPI=1`, `mpiplus` sees a size-1 communicator and OpenMMTools advances the REMD workload serially. With CUDA online/offline analysis, prefer `JAX_PLATFORM_NAME=cpu` so PyMBAR/JAX does not compete with OpenMM for GPU solver resources.
