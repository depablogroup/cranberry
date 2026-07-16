#!/usr/bin/env bash
# Submit or run packed independent Cranberry MD jobs on one GPU with CUDA MPS.

#SBATCH --job-name=cranberry_md_pack
#SBATCH --account=torch_pr_109_courant
#SBATCH --partition=l40s_courant
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=std.out
#SBATCH --open-mode=truncate

set -euo pipefail

if [ -n "${CRANBERRY_MD_SUBMIT_SCRIPT:-}" ]; then
    script_path="$(realpath "$CRANBERRY_MD_SUBMIT_SCRIPT")"
else
    script_path="$(realpath "${BASH_SOURCE[0]}")"
fi

if [ -n "${CRANBERRY_REPO_ROOT:-}" ]; then
    repo_root="$(realpath "$CRANBERRY_REPO_ROOT")"
else
    script_dir="$(cd "$(dirname "$script_path")" && pwd)"
    repo_root="$(cd "$script_dir/../.." && pwd)"
fi

helper_script="${CRANBERRY_MD_HELPER:-$repo_root/scripts/hpc/run_cranberry_md_mps_pack.sh}"
if [[ "$helper_script" != /* ]]; then
    helper_script="$repo_root/$helper_script"
fi

: "${PDB_PATHS:?Set PDB_PATHS to one or more canonical CRANBERRY CG PDBs.}"

integer_value() {
    awk -v value="$1" 'BEGIN { printf "%.0f", value }'
}

last_logged_step() {
    local log_path="$1"
    if [ ! -f "$log_path" ]; then
        echo 0
        return
    fi
    awk -F',' '
        NR > 1 {
            gsub(/"/, "", $1)
            if ($1 ~ /^[0-9]+$/) {
                last = $1
            }
        }
        END { print last + 0 }
    ' "$log_path"
}

run_dir_for_index() {
    local index="$1"
    local pdb_path="$2"
    local run_name="$3"
    if [ "${#run_dirs[@]}" -gt "$index" ] && [ -n "${run_dirs[$index]}" ]; then
        printf "%s" "${run_dirs[$index]}"
        return
    fi
    if [ -z "$run_name" ]; then
        run_name="$(basename "$pdb_path")"
        run_name="${run_name%.*}"
    fi
    run_name="${run_name//[^A-Za-z0-9_.-]/_}"
    printf "%s/%03d_%s" "$OUTPUT_ROOT" "$index" "$run_name"
}

append_production_log() {
    mkdir -p "$META_DIR"
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
    } >> "$META_DIR/production_restarts.log"
}

archive_slurm_output() {
    local current="$META_DIR/std.out"
    local previous="$META_DIR/std.out.prev"
    local tmp="$META_DIR/std.out.prev.tmp.$$"
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

progress_summary() {
    local min_step=""
    local complete=0
    local total=0
    local incomplete=""
    local index=0
    local pdb_path run_name run_dir step

    for pdb_path in "${pdb_paths[@]}"; do
        run_name=""
        if [ "${#run_names[@]}" -gt "$index" ]; then
            run_name="${run_names[$index]}"
        fi
        run_dir="$(run_dir_for_index "$index" "$pdb_path" "$run_name")"
        step="$(last_logged_step "$run_dir/log")"
        if [ -z "$min_step" ] || [ "$step" -lt "$min_step" ]; then
            min_step="$step"
        fi
        if [ "$step" -ge "$TARGET_STEPS" ]; then
            complete=$(( complete + 1 ))
        else
            if [ -n "$run_name" ]; then
                incomplete+="$run_name:$step "
            else
                incomplete+="$(basename "$pdb_path"):$step "
            fi
        fi
        total=$(( total + 1 ))
        index=$(( index + 1 ))
    done

    PROGRESS_COMPLETE="$complete"
    PROGRESS_TOTAL="$total"
    PROGRESS_MIN_STEP="${min_step:-0}"
    PROGRESS_INCOMPLETE="$incomplete"
}

submit_self() {
    mkdir -p "$OUTPUT_ROOT" "$META_DIR"
    export CRANBERRY_REPO_ROOT="$repo_root"
    export CRANBERRY_MD_SUBMIT_SCRIPT="$script_path"
    export CRANBERRY_MD_HELPER="$helper_script"
    export ACCOUNT CONSTRAINT PARTITION JOB_NAME RUN_LABEL SBATCH_TIME RUN_TIMEOUT KILL_AFTER
    export CPUS_PER_TASK SBATCH_MEM NUM_GPUS RESUBMIT_COUNT MAX_RESUBMITS
    export CONDA_ENV CONDA_ENV_PATH OUTPUT_ROOT META_DIR PYTHON_BIN
    export PDB_PATHS RUN_NAMES RUN_DIRS SALTS TEMPERATURES TARGET_STEPS STEP_SIZE LOG_CADENCE_PS
    export FULL_RUN_N_RECORD REPORT_INTERVAL_STEPS MODEL SALT TEMPERATURE USE_MPS RESTART
    export MAX_PARALLEL USE_PBC BOX_PADDING OPENMM_PLATFORM PLATFORM_PROPERTIES STREAM_LOG
    export DRY_RUN

    sbatch_args=(
        --account="$ACCOUNT"
        --job-name="$JOB_NAME"
        --nodes=1
        --ntasks=1
        --cpus-per-task="$CPUS_PER_TASK"
        --gres="gpu:$NUM_GPUS"
        --mem="$SBATCH_MEM"
        --time="$SBATCH_TIME"
        --chdir="$META_DIR"
        --output=std.out
        --open-mode=truncate
        --export=ALL
    )
    if [ -n "$CONSTRAINT" ]; then
        sbatch_args+=(--constraint="$CONSTRAINT")
    fi
    if [ -n "$PARTITION" ]; then
        sbatch_args+=(--partition="$PARTITION")
    fi

    if [ "$DRY_RUN" = "1" ]; then
        printf "dry-run sbatch:"
        printf " %q" "${sbatch_args[@]}" "$script_path"
        echo
        return
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
        echo "Set PYTHON_BIN or CONDA_ENV_PATH to the Cranberry environment." >&2
        exit 2
    fi

    if ! "$PYTHON_BIN" -c "import openmm, cranberry; print('submit preflight: openmm cranberry OK')" ; then
        echo "ERROR: submit preflight failed under PYTHON_BIN=$PYTHON_BIN" >&2
        echo "Set CONDA_ENV=cranberry-dev, CONDA_ENV_PATH, or PYTHON_BIN to the correct environment." >&2
        exit 2
    fi
}

ACCOUNT="${ACCOUNT:-torch_pr_109_courant}"
CONSTRAINT="${CONSTRAINT:-l40s}"
PARTITION="${PARTITION:-l40s_courant}"
JOB_NAME="${JOB_NAME:-cranberry_md_pack}"
RUN_LABEL="${RUN_LABEL:-Cranberry packed MD production}"
SBATCH_TIME="${SBATCH_TIME:-48:00:00}"
RUN_TIMEOUT="${RUN_TIMEOUT:-47h}"
KILL_AFTER="${KILL_AFTER:-10m}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
SBATCH_MEM="${SBATCH_MEM:-64G}"
NUM_GPUS="${NUM_GPUS:-1}"
RESUBMIT_COUNT="${RESUBMIT_COUNT:-0}"
MAX_RESUBMITS="${MAX_RESUBMITS:-}"
CONDA_ENV="${CONDA_ENV:-cranberry-dev}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-${HOME:-}/.conda/envs/$CONDA_ENV}"
PYTHON_BIN="${PYTHON_BIN:-python}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$repo_root/runs/md_mps_production}"
if [[ "$OUTPUT_ROOT" != /* ]]; then
    OUTPUT_ROOT="$repo_root/$OUTPUT_ROOT"
fi
META_DIR="${META_DIR:-$OUTPUT_ROOT}"
if [[ "$META_DIR" != /* ]]; then
    META_DIR="$repo_root/$META_DIR"
fi

TARGET_STEPS="$(integer_value "${TARGET_STEPS:-1000000000}")"
STEP_SIZE="${STEP_SIZE:-5}"
LOG_CADENCE_PS="${LOG_CADENCE_PS:-100}"
REPORT_INTERVAL_STEPS="${REPORT_INTERVAL_STEPS:-$(awk -v ps="$LOG_CADENCE_PS" -v fs="$STEP_SIZE" 'BEGIN { printf "%.0f", ps * 1000.0 / fs }')}"
REPORT_INTERVAL_STEPS="$(integer_value "$REPORT_INTERVAL_STEPS")"
if [ "$REPORT_INTERVAL_STEPS" -lt 1 ]; then
    REPORT_INTERVAL_STEPS=1
fi
FULL_RUN_N_RECORD="${N_RECORD:-${FULL_RUN_N_RECORD:-$(( TARGET_STEPS / REPORT_INTERVAL_STEPS ))}}"
FULL_RUN_N_RECORD="$(integer_value "$FULL_RUN_N_RECORD")"
if [ "$FULL_RUN_N_RECORD" -lt 1 ]; then
    FULL_RUN_N_RECORD=1
fi

MODEL="${MODEL:-default}"
SALT="${SALT:-150}"
TEMPERATURE="${TEMPERATURE:-298}"
USE_MPS="${USE_MPS:-1}"
RESTART="${RESTART:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"
USE_PBC="${USE_PBC:-0}"
BOX_PADDING="${BOX_PADDING:-3.0}"
OPENMM_PLATFORM="${OPENMM_PLATFORM:-CUDA}"
PLATFORM_PROPERTIES="${PLATFORM_PROPERTIES:-}"
STREAM_LOG="${STREAM_LOG:-0}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a pdb_paths <<< "$PDB_PATHS"
read -r -a run_names <<< "${RUN_NAMES:-}"
read -r -a run_dirs <<< "${RUN_DIRS:-}"

if [ "$MAX_PARALLEL" -eq 0 ]; then
    MAX_PARALLEL="${#pdb_paths[@]}"
fi

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Submitting $JOB_NAME to Slurm"
    echo "  run_label=$RUN_LABEL"
    echo "  output_root=$OUTPUT_ROOT"
    echo "  meta_dir=$META_DIR"
    echo "  account=$ACCOUNT constraint=${CONSTRAINT:-none} partition=${PARTITION:-default}"
    echo "  conda_env=$CONDA_ENV"
    echo "  conda_env_path=$CONDA_ENV_PATH"
    echo "  python_bin=$PYTHON_BIN"
    echo "  target_steps=$TARGET_STEPS"
    echo "  report_interval_steps=$REPORT_INTERVAL_STEPS"
    echo "  full_run_n_record=$FULL_RUN_N_RECORD"
    echo "  max_parallel=$MAX_PARALLEL"
    echo "  max_resubmits=${MAX_RESUBMITS:-unlimited}"
    echo "  dry_run=$DRY_RUN"
    submit_self
    exit 0
fi

mkdir -p "$OUTPUT_ROOT" "$META_DIR"
cd "$repo_root"

prepare_python_env

if [ ! -x "$helper_script" ]; then
    echo "ERROR: Cranberry MD helper not found or not executable: $helper_script" >&2
    echo "Set CRANBERRY_REPO_ROOT to the Cranberry checkout or CRANBERRY_MD_HELPER to the helper script." >&2
    exit 2
fi

progress_summary
if [ "$PROGRESS_COMPLETE" -ge "$PROGRESS_TOTAL" ]; then
    echo "all $PROGRESS_TOTAL runs finished at target_steps=$TARGET_STEPS"
    append_production_log complete \
        "run_label=$RUN_LABEL" \
        "target_steps=$TARGET_STEPS" \
        "complete=$PROGRESS_COMPLETE/$PROGRESS_TOTAL" \
        "min_step=$PROGRESS_MIN_STEP" \
        "python_bin=$PYTHON_BIN" \
        "conda_env=$CONDA_ENV" \
        "conda_env_path=$CONDA_ENV_PATH"
    exit 0
fi

append_production_log start \
    "run_label=$RUN_LABEL" \
    "target_steps=$TARGET_STEPS" \
    "report_interval_steps=$REPORT_INTERVAL_STEPS" \
    "full_run_n_record=$FULL_RUN_N_RECORD" \
    "complete=$PROGRESS_COMPLETE/$PROGRESS_TOTAL" \
    "min_step=$PROGRESS_MIN_STEP" \
    "incomplete=$PROGRESS_INCOMPLETE" \
    "resubmit_count=$RESUBMIT_COUNT" \
    "max_resubmits=${MAX_RESUBMITS:-unlimited}" \
    "python_bin=$PYTHON_BIN" \
    "conda_env=$CONDA_ENV" \
    "conda_env_path=$CONDA_ENV_PATH"

echo "$RUN_LABEL start"
echo "  job_id=${SLURM_JOB_ID:-unknown}"
echo "  node=${SLURM_NODELIST:-${HOSTNAME:-unknown}}"
echo "  output_root=$OUTPUT_ROOT"
echo "  meta_dir=$META_DIR"
echo "  target_steps=$TARGET_STEPS"
echo "  report_interval_steps=$REPORT_INTERVAL_STEPS"
echo "  full_run_n_record=$FULL_RUN_N_RECORD"
echo "  complete=$PROGRESS_COMPLETE/$PROGRESS_TOTAL min_step=$PROGRESS_MIN_STEP"
echo "  incomplete=$PROGRESS_INCOMPLETE"
echo "  timeout=$RUN_TIMEOUT kill_after=$KILL_AFTER"
echo "  resubmit_count=$RESUBMIT_COUNT max_resubmits=${MAX_RESUBMITS:-unlimited}"

set +e
TARGET_STEPS="$TARGET_STEPS" \
N_RECORD="$FULL_RUN_N_RECORD" \
REPORT_INTERVAL_STEPS="$REPORT_INTERVAL_STEPS" \
MODEL="$MODEL" \
SALT="$SALT" \
SALTS="${SALTS:-}" \
TEMPERATURE="$TEMPERATURE" \
TEMPERATURES="${TEMPERATURES:-}" \
STEP_SIZE="$STEP_SIZE" \
OPENMM_PLATFORM="$OPENMM_PLATFORM" \
PDB_PATHS="$PDB_PATHS" \
RUN_NAMES="${RUN_NAMES:-}" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
META_DIR="$META_DIR" \
PYTHON_BIN="$PYTHON_BIN" \
RUN_DIRS="${RUN_DIRS:-}" \
USE_MPS="$USE_MPS" \
RESTART="$RESTART" \
MAX_PARALLEL="$MAX_PARALLEL" \
USE_PBC="$USE_PBC" \
BOX_PADDING="$BOX_PADDING" \
PLATFORM_PROPERTIES="$PLATFORM_PROPERTIES" \
timeout -k "$KILL_AFTER" "$RUN_TIMEOUT" "$helper_script"
exit_code=$?
set -e

progress_summary

echo "md pack exit with code: $exit_code"
echo "complete: $PROGRESS_COMPLETE/$PROGRESS_TOTAL"
echo "min_step: $PROGRESS_MIN_STEP / $TARGET_STEPS"
echo "incomplete: $PROGRESS_INCOMPLETE"

append_production_log exit \
    "exit_code=$exit_code" \
    "run_label=$RUN_LABEL" \
    "target_steps=$TARGET_STEPS" \
    "complete=$PROGRESS_COMPLETE/$PROGRESS_TOTAL" \
    "min_step=$PROGRESS_MIN_STEP" \
    "incomplete=$PROGRESS_INCOMPLETE" \
    "resubmit_count=$RESUBMIT_COUNT" \
    "max_resubmits=${MAX_RESUBMITS:-unlimited}" \
    "python_bin=$PYTHON_BIN" \
    "conda_env=$CONDA_ENV" \
    "conda_env_path=$CONDA_ENV_PATH"

if [ "$PROGRESS_COMPLETE" -ge "$PROGRESS_TOTAL" ]; then
    echo "all $PROGRESS_TOTAL runs finished at target_steps=$TARGET_STEPS"
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
    echo "ERROR: run exited successfully but some MD runs are below target; not resubmitting automatically." >&2
    exit 4
fi

echo "Not resubmitting because exit code $exit_code was not a timeout/kill restart code." >&2
exit "$exit_code"
