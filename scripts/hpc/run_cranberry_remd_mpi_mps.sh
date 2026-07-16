#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
source "$script_dir/mpi_runtime_env.sh"

: "${PDB_PATH:?Set PDB_PATH to the canonical CRANBERRY CG PDB.}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/remd_${RUN_TAG}}"
MODEL="${MODEL:-default}"
NSTEPS="${NSTEPS:-500000}"
SWAP_STEPS="${SWAP_STEPS:-${SWAP_STEP:-5000}}"
N_RECORD="${N_RECORD:-10000}"
N_ANALYSIS="${N_ANALYSIS:-10}"
N_REPLICAS="${N_REPLICAS:-8}"
N_MPI_RANKS="${N_MPI_RANKS:-$N_REPLICAS}"
T_MIN="${T_MIN:-298}"
T_MAX="${T_MAX:-600}"
SALT="${SALT:-150}"
STEP_SIZE="${STEP_SIZE:-5}"
OPENMM_PLATFORM="${OPENMM_PLATFORM:-CUDA}"
NUM_GPUS="${NUM_GPUS:-1}"
USE_MPS="${USE_MPS:-1}"
RESTART="${RESTART:-1}"
USE_PBC="${USE_PBC:-0}"
BOX_PADDING="${BOX_PADDING:-3.0}"
WRITE_DCD="${WRITE_DCD:-0}"
DCD_MODE="${DCD_MODE:-replica}"
MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun}"
MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS:-}"
STREAM_LOG="${STREAM_LOG:-0}"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$repo_root"
export OPENMMTOOLS_ENABLE_MPI="${OPENMMTOOLS_ENABLE_MPI:-1}"
export NUM_GPUS N_MPI_RANKS N_REPLICAS
export PARENT_CUDA_VISIBLE_DEVICES="${PARENT_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"

mkdir -p "$OUTPUT_DIR"

allocated_cpus="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-}}"
cpu_warning=""
if [[ "$allocated_cpus" =~ ^[0-9]+$ ]] && (( allocated_cpus > 0 && allocated_cpus < N_MPI_RANKS )); then
    cpu_warning="warning: N_MPI_RANKS=$N_MPI_RANKS exceeds allocated CPUs=$allocated_cpus; MPI ranks will share CPU cores and throughput may drop."
    echo "$cpu_warning" >&2
fi

mps_started=0
stop_mps() {
    if [ "$mps_started" -eq 1 ] && command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
        echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    fi
}
trap stop_mps EXIT

if [ "$USE_MPS" = "1" ]; then
    export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/cranberry-mps-${USER:-user}-${SLURM_JOB_ID:-$$}}"
    export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-$OUTPUT_DIR/mps-log}"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
        if nvidia-cuda-mps-control -d; then
            mps_started=1
        else
            echo "warning: failed to start CUDA MPS control daemon; continuing without a new daemon" >&2
        fi
    else
        echo "warning: nvidia-cuda-mps-control not found; continuing without CUDA MPS" >&2
    fi
fi

{
    "$PYTHON_BIN" -c "import openmmtools, mpi4py; print('preflight: openmmtools and mpi4py import OK')"
    "$PYTHON_BIN" -c "import os; print('preflight: JAX_PLATFORM_NAME=' + str(os.environ.get('JAX_PLATFORM_NAME'))); import jax; print('preflight: jax ' + jax.__version__); print('preflight: jax backend ' + jax.default_backend()); print('preflight: jax devices ' + ', '.join(map(str, jax.devices())))" || true
} > "$OUTPUT_DIR/preflight.log" 2>&1

cmd=(
    "$PYTHON_BIN" -m cranberry.cli remd "$PDB_PATH"
    --steps "$NSTEPS"
    --swap-steps "$SWAP_STEPS"
    --n-record "$N_RECORD"
    --n-analysis "$N_ANALYSIS"
    --output-dir "$OUTPUT_DIR"
    --model "$MODEL"
    --salt "$SALT"
    --timestep "$STEP_SIZE"
    --platform "$OPENMM_PLATFORM"
)

if [ -n "${TEMPERATURE_LADDER:-}" ]; then
    read -r -a ladder_values <<< "$TEMPERATURE_LADDER"
    cmd+=(--temperature-ladder "${ladder_values[@]}")
else
    cmd+=(--t-min "$T_MIN" --t-max "$T_MAX" --n-replicas "$N_REPLICAS")
fi

if [ -n "${ALT_PDB_PATH:-}" ] && [ ! -f "$OUTPUT_DIR/output.nc" ]; then
    cmd+=(--extra-start-pdb "$ALT_PDB_PATH")
fi

if [ "$RESTART" = "1" ] && [ -f "$OUTPUT_DIR/output.nc" ]; then
    if ! "$PYTHON_BIN" -c "import sys, netCDF4; dataset = netCDF4.Dataset(sys.argv[1], 'r'); dataset.close()" "$OUTPUT_DIR/output.nc"; then
        echo "error: $OUTPUT_DIR/output.nc exists but NetCDF4 could not open it." >&2
        echo "The previous run likely aborted while writing this file." >&2
        echo "Use a fresh OUTPUT_DIR, or move the incomplete run aside before retrying." >&2
        exit 3
    fi
    cmd+=(--restart-from "$OUTPUT_DIR/output.nc")
fi

if [ "${OVERWRITE:-0}" = "1" ]; then
    cmd+=(--overwrite)
fi

if [ "$USE_PBC" = "1" ]; then
    cmd+=(--periodic --box-padding "$BOX_PADDING")
fi

if [ "$WRITE_DCD" = "1" ]; then
    cmd+=(--write-dcd)
    if [ "$DCD_MODE" = "temperature" ]; then
        cmd+=(--by-temperature)
    else
        cmd+=(--by-replica)
    fi
fi

platform_properties="${PLATFORM_PROPERTIES:-}"
platform_properties="${platform_properties//,/ }"
for property in $platform_properties; do
    cmd+=(--platform-property "$property")
done

mpi_extra_args=()
if [ -n "$MPI_EXTRA_ARGS" ]; then
    read -r -a mpi_extra_args <<< "$MPI_EXTRA_ARGS"
fi

if [ "$MPI_LAUNCHER" = "srun" ]; then
    launcher=(srun -n "$N_MPI_RANKS" "$script_dir/mpi_rank_gpu_wrapper.sh")
else
    launcher=(mpirun --oversubscribe --bind-to none --mca psec ^munge "${mpi_extra_args[@]}" -np "$N_MPI_RANKS" "$script_dir/mpi_rank_gpu_wrapper.sh")
fi

{
    echo "date: $(date -Is)"
    echo "hostname: $(hostname)"
    echo "repo_root: $repo_root"
    echo "output_dir: $OUTPUT_DIR"
    echo "pdb_path: $PDB_PATH"
    echo "alt_pdb_path: ${ALT_PDB_PATH:-}"
    echo "nsteps: $NSTEPS"
    echo "swap_steps: $SWAP_STEPS"
    echo "n_record: $N_RECORD"
    echo "n_analysis: $N_ANALYSIS"
    echo "n_replicas: $N_REPLICAS"
    echo "n_mpi_ranks: $N_MPI_RANKS"
    echo "num_gpus: $NUM_GPUS"
    echo "use_mps: $USE_MPS"
    echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-}"
    echo "parent_cuda_visible_devices: ${PARENT_CUDA_VISIBLE_DEVICES:-}"
    echo "cuda_mps_pipe_directory: ${CUDA_MPS_PIPE_DIRECTORY:-}"
    echo "cuda_mps_log_directory: ${CUDA_MPS_LOG_DIRECTORY:-}"
    echo "jax_platform_name: ${JAX_PLATFORM_NAME:-}"
    echo "mpi_extra_args: ${MPI_EXTRA_ARGS:-}"
    echo "stream_log: $STREAM_LOG"
    echo "slurm_job_id: ${SLURM_JOB_ID:-}"
    echo "slurm_job_partition: ${SLURM_JOB_PARTITION:-}"
    echo "slurm_job_cpus_per_node: ${SLURM_JOB_CPUS_PER_NODE:-}"
    echo "slurm_cpus_on_node: ${SLURM_CPUS_ON_NODE:-}"
    echo "slurm_cpus_per_task: ${SLURM_CPUS_PER_TASK:-}"
    echo "slurm_ntasks: ${SLURM_NTASKS:-}"
    echo "omp_num_threads: ${OMP_NUM_THREADS:-}"
    echo "openmm_cpu_threads: ${OPENMM_CPU_THREADS:-}"
    echo "processors_online: $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo unknown)"
    echo "taskset_affinity: $(taskset -pc $$ 2>/dev/null || echo unknown)"
    echo "cpu_warning: $cpu_warning"
    echo "preflight_log: $OUTPUT_DIR/preflight.log"
    echo "stdout_log: $OUTPUT_DIR/stdout.log"
    echo "stderr_log: $OUTPUT_DIR/stderr.log"
    echo "combined_log: $OUTPUT_DIR/remd_mpi_mps.log"
    printf "command:"
    printf " %q" "${launcher[@]}" "${cmd[@]}"
    echo
} > "$OUTPUT_DIR/launcher_env.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "$OUTPUT_DIR/nvidia_smi_start.txt" || true
fi

start_epoch="$(date +%s)"
set +e
if [ "$STREAM_LOG" = "1" ]; then
    "${launcher[@]}" "${cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/remd_mpi_mps.log"
    status=${PIPESTATUS[0]}
    cp "$OUTPUT_DIR/remd_mpi_mps.log" "$OUTPUT_DIR/stdout.log" 2>/dev/null || true
    : > "$OUTPUT_DIR/stderr.log"
else
    "${launcher[@]}" "${cmd[@]}" > "$OUTPUT_DIR/stdout.log" 2> "$OUTPUT_DIR/stderr.log"
    status=$?
    {
        echo "### stdout.log"
        cat "$OUTPUT_DIR/stdout.log"
        echo
        echo "### stderr.log"
        cat "$OUTPUT_DIR/stderr.log"
    } > "$OUTPUT_DIR/remd_mpi_mps.log"
fi
set -e
end_epoch="$(date +%s)"
elapsed_seconds=$(( end_epoch - start_epoch ))

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "$OUTPUT_DIR/nvidia_smi_end.txt" || true
fi

"$PYTHON_BIN" -c "
import sys
nsteps = int(sys.argv[1])
timestep_fs = float(sys.argv[2])
n_replicas = int(sys.argv[3])
elapsed_s = max(float(sys.argv[4]), 1.0)
status = int(sys.argv[5])
per_replica_ns = nsteps * timestep_fs / 1_000_000.0
aggregate_replica_ns = per_replica_ns * n_replicas
aggregate_replica_ns_per_day = aggregate_replica_ns * 86400.0 / elapsed_s
per_replica_ns_per_day = per_replica_ns * 86400.0 / elapsed_s
print(f'status: {status}')
print(f'elapsed_seconds: {elapsed_s:.0f}')
print(f'nsteps_per_replica: {nsteps}')
print(f'n_replicas: {n_replicas}')
print(f'timestep_femtosecond: {timestep_fs:g}')
print(f'per_replica_ns: {per_replica_ns:.6f}')
print(f'aggregate_replica_ns: {aggregate_replica_ns:.6f}')
print(f'per_replica_ns_per_day: {per_replica_ns_per_day:.3f}')
print(f'aggregate_replica_ns_per_day: {aggregate_replica_ns_per_day:.3f}')
" "$NSTEPS" "$STEP_SIZE" "$N_REPLICAS" "$elapsed_seconds" "$status" | tee "$OUTPUT_DIR/timing_summary.txt"

exit "$status"
