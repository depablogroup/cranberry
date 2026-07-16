#!/usr/bin/env bash
# Submit or run a 5 us C ACAG PBC REMD production job on one L40S GPU.
#
# Run directly from the Cranberry repo to submit:
#   scripts/hpc/submit_cacag_5us_pbc_l40s.sh
#
# The batch payload stops after RUN_TIMEOUT, default 47h, inside a 48h Slurm
# allocation and resubmits itself until TARGET_ITERATIONS are present in output.nc.

#SBATCH --job-name=cacag_5us_pbc
#SBATCH --account=torch_pr_109_courant
#SBATCH --partition=l40s_courant
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=std.out
#SBATCH --open-mode=truncate

set -euo pipefail

if [ -n "${CRANBERRY_SUBMIT_SCRIPT:-}" ]; then
    script_path="$(realpath "$CRANBERRY_SUBMIT_SCRIPT")"
else
    script_path="$(realpath "${BASH_SOURCE[0]}")"
fi

if [ -n "${CRANBERRY_REPO_ROOT:-}" ]; then
    repo_root="$(realpath "$CRANBERRY_REPO_ROOT")"
else
    script_dir="$(cd "$(dirname "$script_path")" && pwd)"
    repo_root="$(cd "$script_dir/../.." && pwd)"
fi

helper_script="${CRANBERRY_REMD_HELPER:-$repo_root/scripts/hpc/run_cranberry_remd_mpi_mps.sh}"
if [[ "$helper_script" != /* ]]; then
    helper_script="$repo_root/$helper_script"
fi

ACCOUNT="${ACCOUNT:-torch_pr_109_courant}"
CONSTRAINT="${CONSTRAINT:-l40s}"
PARTITION="${PARTITION:-l40s_courant}"
JOB_NAME="${JOB_NAME:-cacag_5us_pbc}"
RUN_LABEL="${RUN_LABEL:-C ACAG 5us PBC production}"
SBATCH_TIME="${SBATCH_TIME:-48:00:00}"
RUN_TIMEOUT="${RUN_TIMEOUT:-47h}"
KILL_AFTER="${KILL_AFTER:-10m}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
SBATCH_MEM="${SBATCH_MEM:-32G}"
NUM_GPUS="${NUM_GPUS:-1}"
RESUBMIT_COUNT="${RESUBMIT_COUNT:-0}"
MAX_RESUBMITS="${MAX_RESUBMITS:-}"
CONDA_ENV="${CONDA_ENV:-cranberry-dev}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-${HOME:-}/.conda/envs/$CONDA_ENV}"

OUTPUT_DIR="${OUTPUT_DIR:-$repo_root/runs/cacag_pbc_5us_l40s}"
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$repo_root/$OUTPUT_DIR"
fi
if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "$CONDA_ENV_PATH/bin/python" ]; then
        PYTHON_BIN="$CONDA_ENV_PATH/bin/python"
    else
        PYTHON_BIN="python"
    fi
fi

PDB_PATH="${PDB_PATH:-/projects/rps/jjd8110/depablolab/cedric/04-cranberry/melting/seed/cacag_cg_vs_conect.pdb}"
ALT_PDB_PATH="${ALT_PDB_PATH:-/projects/rps/jjd8110/depablolab/cedric/04-cranberry/melting/seed/cacag_extended_cg_vs_conect.pdb}"

TARGET_STEPS="${TARGET_STEPS:-1000000000}"
STEP_SIZE="${STEP_SIZE:-5}"
SWAP_STEPS="${SWAP_STEPS:-5000}"
TARGET_ITERATIONS="${TARGET_ITERATIONS:-$(( TARGET_STEPS / SWAP_STEPS ))}"

FULL_RUN_N_RECORD="${N_RECORD:-10000}"
TARGET_CHECKPOINT_INTERVAL="${TARGET_CHECKPOINT_INTERVAL:-$(( TARGET_ITERATIONS / FULL_RUN_N_RECORD ))}"
if [ "$TARGET_CHECKPOINT_INTERVAL" -lt 1 ]; then
    TARGET_CHECKPOINT_INTERVAL=1
fi

if [ -n "${TARGET_ANALYSIS_INTERVAL:-}" ]; then
    if [ "$TARGET_ANALYSIS_INTERVAL" -eq 0 ]; then
        FULL_RUN_N_ANALYSIS=0
    else
        if [ "$TARGET_ANALYSIS_INTERVAL" -lt 1 ]; then
            TARGET_ANALYSIS_INTERVAL=1
        fi
        FULL_RUN_N_ANALYSIS=$(( TARGET_ITERATIONS / TARGET_ANALYSIS_INTERVAL ))
        if [ "$FULL_RUN_N_ANALYSIS" -lt 1 ]; then
            FULL_RUN_N_ANALYSIS=1
        fi
    fi
elif [ -n "${N_ANALYSIS+x}" ] && [ -n "$N_ANALYSIS" ]; then
    FULL_RUN_N_ANALYSIS="$N_ANALYSIS"
    if [ "$FULL_RUN_N_ANALYSIS" -gt 0 ]; then
        TARGET_ANALYSIS_INTERVAL=$(( TARGET_ITERATIONS / FULL_RUN_N_ANALYSIS ))
        if [ "$TARGET_ANALYSIS_INTERVAL" -lt 1 ]; then
            TARGET_ANALYSIS_INTERVAL=1
        fi
    else
        TARGET_ANALYSIS_INTERVAL=0
    fi
else
    TARGET_ANALYSIS_INTERVAL=$(( TARGET_ITERATIONS / 10 ))
    legacy_analysis_cap=$(( TARGET_CHECKPOINT_INTERVAL * 10 ))
    if [ "$legacy_analysis_cap" -lt "$TARGET_ANALYSIS_INTERVAL" ]; then
        TARGET_ANALYSIS_INTERVAL="$legacy_analysis_cap"
    fi
    if [ "$TARGET_ANALYSIS_INTERVAL" -lt 1 ]; then
        TARGET_ANALYSIS_INTERVAL=1
    fi
    FULL_RUN_N_ANALYSIS=$(( TARGET_ITERATIONS / TARGET_ANALYSIS_INTERVAL ))
    if [ "$FULL_RUN_N_ANALYSIS" -lt 1 ]; then
        FULL_RUN_N_ANALYSIS=1
    fi
fi

N_REPLICAS="${N_REPLICAS:-8}"
N_MPI_RANKS="${N_MPI_RANKS:-8}"
T_MIN="${T_MIN:-280}"
T_MAX="${T_MAX:-380}"
SALT="${SALT:-1000}"
USE_PBC="${USE_PBC:-1}"
BOX_PADDING="${BOX_PADDING:-2}"
USE_MPS="${USE_MPS:-1}"
RESTART="${RESTART:-1}"
STREAM_LOG="${STREAM_LOG:-0}"
OPENMM_PLATFORM="${OPENMM_PLATFORM:-CUDA}"
MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS:---mca pml ob1 --mca mtl ^ofi --mca shmem_mmap_enable_nfs_warning 0}"

submit_self() {
    mkdir -p "$OUTPUT_DIR"
    export CRANBERRY_REPO_ROOT="$repo_root"
    export CRANBERRY_SUBMIT_SCRIPT="$script_path"
    export CRANBERRY_REMD_HELPER="$helper_script"
    export ACCOUNT CONSTRAINT PARTITION JOB_NAME RUN_LABEL SBATCH_TIME RUN_TIMEOUT KILL_AFTER
    export CPUS_PER_TASK SBATCH_MEM NUM_GPUS RESUBMIT_COUNT MAX_RESUBMITS
    export CONDA_ENV CONDA_ENV_PATH OUTPUT_DIR PYTHON_BIN
    export PDB_PATH ALT_PDB_PATH TARGET_STEPS STEP_SIZE SWAP_STEPS TARGET_ITERATIONS
    export FULL_RUN_N_RECORD FULL_RUN_N_ANALYSIS TARGET_CHECKPOINT_INTERVAL TARGET_ANALYSIS_INTERVAL
    export N_REPLICAS N_MPI_RANKS T_MIN T_MAX SALT USE_PBC BOX_PADDING USE_MPS RESTART STREAM_LOG
    export OPENMM_PLATFORM MPI_EXTRA_ARGS

    sbatch_args=(
        --account="$ACCOUNT"
        --constraint="$CONSTRAINT"
        --job-name="$JOB_NAME"
        --nodes=1
        --ntasks=1
        --cpus-per-task="$CPUS_PER_TASK"
        --gres="gpu:$NUM_GPUS"
        --mem="$SBATCH_MEM"
        --time="$SBATCH_TIME"
        --chdir="$OUTPUT_DIR"
        --output=std.out
        --open-mode=truncate
        --export=ALL
    )
    if [ -n "$PARTITION" ]; then
        sbatch_args+=(--partition="$PARTITION")
    fi

    sbatch "${sbatch_args[@]}" "$script_path"
}

prepare_python_env() {
    if [ -x "$CONDA_ENV_PATH/bin/python" ]; then
        export PATH="$CONDA_ENV_PATH/bin:$PATH"
        export CONDA_PREFIX="$CONDA_ENV_PATH"
        export CONDA_DEFAULT_ENV="$CONDA_ENV"
        if [ "$PYTHON_BIN" = "python" ]; then
            PYTHON_BIN="$CONDA_ENV_PATH/bin/python"
            export PYTHON_BIN
        fi
    fi

    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        echo "ERROR: PYTHON_BIN is not executable or not on PATH: $PYTHON_BIN" >&2
        echo "Set PYTHON_BIN or CONDA_ENV_PATH to the cranberry-dev environment." >&2
        exit 2
    fi

    if ! "$PYTHON_BIN" -c "import openmmtools, mpi4py, netCDF4; print('submit preflight: openmmtools mpi4py netCDF4 OK')" ; then
        echo "ERROR: submit preflight failed under PYTHON_BIN=$PYTHON_BIN" >&2
        echo "Set CONDA_ENV=cranberry-dev, CONDA_ENV_PATH, or PYTHON_BIN to the correct environment." >&2
        exit 2
    fi

    "$PYTHON_BIN" -c "import jax; print('submit preflight: jax', jax.__version__, 'backend', jax.default_backend())" || true
}

stored_iteration() {
    if [ ! -f "$OUTPUT_DIR/output.nc" ]; then
        echo 0
        return
    fi
    "$PYTHON_BIN" - "$OUTPUT_DIR/output.nc" <<'PY' 2>/dev/null || echo 0
import sys
from netCDF4 import Dataset

with Dataset(sys.argv[1], "r") as dataset:
    if "last_iteration" in dataset.variables:
        print(int(dataset.variables["last_iteration"][0]))
    elif "iteration" in dataset.dimensions:
        print(max(0, len(dataset.dimensions["iteration"]) - 1))
    else:
        print(0)
PY
}

last_analysis_iteration() {
    local analysis_file="$OUTPUT_DIR/output_real_time_analysis.yaml"
    if [ ! -f "$analysis_file" ]; then
        echo 0
        return
    fi
    tail -n20 "$analysis_file" | grep -a -- "- iteration:" | tail -n1 | awk '{print $3}' || echo 0
}

iteration_progress() {
    local iteration="$1"
    awk -v iter="$iteration" -v swap="$SWAP_STEPS" -v dt="$STEP_SIZE" 'BEGIN {
        steps = iter * swap
        ns = steps * dt / 1000000.0
        us = ns / 1000.0
        printf "iterations=%d steps=%.0f ns=%.6f us=%.6f", iter, steps, ns, us
    }'
}

archive_slurm_output() {
    local current="$OUTPUT_DIR/std.out"
    local previous="$OUTPUT_DIR/std.out.prev"
    local tmp="$OUTPUT_DIR/std.out.prev.tmp.$$"
    if [ ! -f "$current" ]; then
        return
    fi
    if [ -f "$previous" ]; then
        cat "$previous" "$current" > "$tmp"
        mv "$tmp" "$previous"
    else
        cp "$current" "$previous"
    fi
}

append_production_log() {
    mkdir -p "$OUTPUT_DIR"
    {
        echo "date=$(date -Is)"
        echo "job_id=${SLURM_JOB_ID:-submit}"
        echo "node=${SLURM_NODELIST:-${HOSTNAME:-unknown}}"
        echo "event=$1"
        shift || true
        for item in "$@"; do
            echo "$item"
        done
        echo
    } >> "$OUTPUT_DIR/production_restarts.log"
}

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Submitting $JOB_NAME to Slurm"
    echo "  run_label=$RUN_LABEL"
    echo "  output_dir=$OUTPUT_DIR"
    echo "  account=$ACCOUNT constraint=$CONSTRAINT partition=${PARTITION:-default}"
    echo "  conda_env=$CONDA_ENV"
    echo "  conda_env_path=$CONDA_ENV_PATH"
    echo "  python_bin=$PYTHON_BIN"
    echo "  target_steps=$TARGET_STEPS target_iterations=$TARGET_ITERATIONS"
    echo "  target_checkpoint_interval=$TARGET_CHECKPOINT_INTERVAL"
    echo "  target_analysis_interval=$TARGET_ANALYSIS_INTERVAL full_run_n_analysis=$FULL_RUN_N_ANALYSIS"
    echo "  max_resubmits=${MAX_RESUBMITS:-unlimited}"
    submit_self
    exit 0
fi

mkdir -p "$OUTPUT_DIR"
cd "$repo_root"

prepare_python_env

if [ ! -x "$helper_script" ]; then
    echo "ERROR: Cranberry REMD helper not found or not executable: $helper_script" >&2
    echo "Set CRANBERRY_REPO_ROOT to the Cranberry checkout or CRANBERRY_REMD_HELPER to the helper script." >&2
    exit 2
fi

if [ $(( TARGET_STEPS % SWAP_STEPS )) -ne 0 ]; then
    echo "ERROR: TARGET_STEPS=$TARGET_STEPS must be divisible by SWAP_STEPS=$SWAP_STEPS" >&2
    exit 2
fi

current_iteration="$(stored_iteration)"
if [ "$current_iteration" -ge "$TARGET_ITERATIONS" ]; then
    echo "total $TARGET_ITERATIONS finished!"
    if [ "$current_iteration" -gt "$TARGET_ITERATIONS" ]; then
        echo "stored iteration is above target; not launching another attempt."
    fi
    append_production_log complete \
        "current_iteration=$current_iteration" \
        "run_label=$RUN_LABEL" \
        "current_progress=$(iteration_progress "$current_iteration")" \
        "target_iterations=$TARGET_ITERATIONS" \
        "target_progress=$(iteration_progress "$TARGET_ITERATIONS")" \
        "python_bin=$PYTHON_BIN" \
        "conda_env=$CONDA_ENV" \
        "conda_env_path=$CONDA_ENV_PATH"
    exit 0
fi

remaining_iterations=$(( TARGET_ITERATIONS - current_iteration ))
attempt_steps=$(( remaining_iterations * SWAP_STEPS ))
attempt_n_record=$(( remaining_iterations / TARGET_CHECKPOINT_INTERVAL ))
if [ "$attempt_n_record" -lt 1 ]; then
    attempt_n_record=1
fi
attempt_checkpoint_interval=$(( remaining_iterations / attempt_n_record ))
if [ "$FULL_RUN_N_ANALYSIS" -eq 0 ]; then
    attempt_n_analysis=0
    attempt_analysis_interval=0
else
    attempt_n_analysis=$(( remaining_iterations / TARGET_ANALYSIS_INTERVAL ))
    if [ "$attempt_n_analysis" -lt 1 ]; then
        attempt_n_analysis=1
    fi
    attempt_analysis_interval=$(( remaining_iterations / attempt_n_analysis ))
fi

append_production_log start \
    "current_iteration=$current_iteration" \
    "run_label=$RUN_LABEL" \
    "current_progress=$(iteration_progress "$current_iteration")" \
    "target_progress=$(iteration_progress "$TARGET_ITERATIONS")" \
    "remaining_iterations=$remaining_iterations" \
    "attempt_steps=$attempt_steps" \
    "attempt_n_record=$attempt_n_record" \
    "attempt_checkpoint_interval=$attempt_checkpoint_interval" \
    "attempt_n_analysis=$attempt_n_analysis" \
    "attempt_analysis_interval=$attempt_analysis_interval" \
    "full_run_n_analysis=$FULL_RUN_N_ANALYSIS" \
    "target_checkpoint_interval=$TARGET_CHECKPOINT_INTERVAL" \
    "target_analysis_interval=$TARGET_ANALYSIS_INTERVAL" \
    "python_bin=$PYTHON_BIN" \
    "conda_env=$CONDA_ENV" \
    "conda_env_path=$CONDA_ENV_PATH"

echo "$RUN_LABEL start"
echo "  job_id=${SLURM_JOB_ID:-unknown}"
echo "  node=${SLURM_NODELIST:-${HOSTNAME:-unknown}}"
echo "  output_dir=$OUTPUT_DIR"
echo "  conda_env=$CONDA_ENV"
echo "  conda_env_path=$CONDA_ENV_PATH"
echo "  python_bin=$PYTHON_BIN"
echo "  current_iteration=$current_iteration / $TARGET_ITERATIONS"
echo "  current_progress=$(iteration_progress "$current_iteration")"
echo "  target_progress=$(iteration_progress "$TARGET_ITERATIONS")"
echo "  remaining_iterations=$remaining_iterations"
echo "  attempt_steps=$attempt_steps"
echo "  attempt_checkpoint_interval=$attempt_checkpoint_interval"
echo "  attempt_n_analysis=$attempt_n_analysis"
echo "  attempt_analysis_interval=$attempt_analysis_interval"
echo "  full_run_n_analysis=$FULL_RUN_N_ANALYSIS"
echo "  timeout=$RUN_TIMEOUT kill_after=$KILL_AFTER"
echo "  resubmit_count=$RESUBMIT_COUNT max_resubmits=${MAX_RESUBMITS:-unlimited}"

set +e
TARGET_STEPS="$TARGET_STEPS" \
NSTEPS="$attempt_steps" \
SWAP_STEPS="$SWAP_STEPS" \
N_RECORD="$attempt_n_record" \
N_ANALYSIS="$attempt_n_analysis" \
N_REPLICAS="$N_REPLICAS" \
N_MPI_RANKS="$N_MPI_RANKS" \
NUM_GPUS="$NUM_GPUS" \
USE_MPS="$USE_MPS" \
RESTART="$RESTART" \
USE_PBC="$USE_PBC" \
BOX_PADDING="$BOX_PADDING" \
T_MIN="$T_MIN" \
T_MAX="$T_MAX" \
SALT="$SALT" \
STEP_SIZE="$STEP_SIZE" \
OPENMM_PLATFORM="$OPENMM_PLATFORM" \
PDB_PATH="$PDB_PATH" \
ALT_PDB_PATH="$ALT_PDB_PATH" \
OUTPUT_DIR="$OUTPUT_DIR" \
PYTHON_BIN="$PYTHON_BIN" \
STREAM_LOG="$STREAM_LOG" \
MPI_EXTRA_ARGS="$MPI_EXTRA_ARGS" \
timeout -k "$KILL_AFTER" "$RUN_TIMEOUT" "$helper_script"
exit_code=$?
set -e

current_iteration="$(stored_iteration)"
analysis_iteration="$(last_analysis_iteration)"

echo "pt exit with code: $exit_code"
echo "stored iteration: $current_iteration / $TARGET_ITERATIONS"
echo "stored progress: $(iteration_progress "$current_iteration")"
echo "last analysis iteration: ${analysis_iteration:-0}"

append_production_log exit \
    "exit_code=$exit_code" \
    "run_label=$RUN_LABEL" \
    "current_iteration=$current_iteration" \
    "current_progress=$(iteration_progress "$current_iteration")" \
    "target_progress=$(iteration_progress "$TARGET_ITERATIONS")" \
    "analysis_iteration=${analysis_iteration:-0}" \
    "target_iterations=$TARGET_ITERATIONS" \
    "resubmit_count=$RESUBMIT_COUNT" \
    "max_resubmits=${MAX_RESUBMITS:-unlimited}" \
    "python_bin=$PYTHON_BIN" \
    "conda_env=$CONDA_ENV" \
    "conda_env_path=$CONDA_ENV_PATH"

if [ "$current_iteration" -ge "$TARGET_ITERATIONS" ]; then
    echo "total $TARGET_ITERATIONS finished!"
    if [ "$current_iteration" -gt "$TARGET_ITERATIONS" ]; then
        echo "stored iteration is above target; not resubmitting."
    fi
    exit 0
fi

if [ "$exit_code" -eq 137 ] || [ "$exit_code" -eq 124 ]; then
    if [[ "$MAX_RESUBMITS" =~ ^[0-9]+$ ]] && [ "$RESUBMIT_COUNT" -ge "$MAX_RESUBMITS" ]; then
        echo "Not resubmitting because MAX_RESUBMITS=$MAX_RESUBMITS has been reached." >&2
        exit "$exit_code"
    fi
    echo "resubmitting job!"
    archive_slurm_output
    RESUBMIT_COUNT=$(( RESUBMIT_COUNT + 1 ))
    submit_self
    exit 0
fi

if [ "$exit_code" -eq 0 ]; then
    echo "ERROR: run exited successfully but stored iteration is below target; not resubmitting automatically." >&2
    exit 4
fi

echo "Not resubmitting because exit code $exit_code was not a timeout/kill restart code." >&2
exit "$exit_code"
