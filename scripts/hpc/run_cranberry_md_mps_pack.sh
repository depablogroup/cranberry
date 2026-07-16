#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

PDB_INPUTS="${PDB_PATHS:-${PDB_PATH:-}}"
if [ -z "$PDB_INPUTS" ]; then
    echo "Set PDB_PATHS to one or more canonical CRANBERRY CG PDBs, or set PDB_PATH for one run." >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/md_mps_${RUN_TAG}}"
META_DIR="${META_DIR:-$OUTPUT_ROOT}"
MODEL="${MODEL:-default}"
TARGET_STEPS="${TARGET_STEPS:-${NSTEPS:-500000}}"
N_RECORD="${N_RECORD:-1000}"
REPORT_INTERVAL_STEPS="${REPORT_INTERVAL_STEPS:-}"
TEMPERATURE="${TEMPERATURE:-298}"
SALT="${SALT:-150}"
STEP_SIZE="${STEP_SIZE:-5}"
OPENMM_PLATFORM="${OPENMM_PLATFORM:-CUDA}"
USE_MPS="${USE_MPS:-1}"
RESTART="${RESTART:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"
USE_PBC="${USE_PBC:-0}"
BOX_PADDING="${BOX_PADDING:-3.0}"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$repo_root"

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

value_at_index() {
    local fallback="$1"
    local index="$2"
    shift 2
    local values=("$@")
    if [ "${#values[@]}" -gt "$index" ]; then
        echo "${values[$index]}"
    else
        echo "$fallback"
    fi
}

TARGET_STEPS="$(integer_value "$TARGET_STEPS")"
N_RECORD="$(integer_value "$N_RECORD")"
if [ -n "$REPORT_INTERVAL_STEPS" ]; then
    REPORT_INTERVAL_STEPS="$(integer_value "$REPORT_INTERVAL_STEPS")"
else
    REPORT_INTERVAL_STEPS=$(( TARGET_STEPS / N_RECORD ))
fi
if [ "$REPORT_INTERVAL_STEPS" -lt 1 ]; then
    REPORT_INTERVAL_STEPS=1
fi

mkdir -p "$OUTPUT_ROOT" "$META_DIR"

mps_started=0
stop_mps() {
    if [ "$mps_started" -eq 1 ] && command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
        echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    fi
}
trap stop_mps EXIT

if [ "$USE_MPS" = "1" ]; then
    export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/cranberry-md-mps-${USER:-user}-${SLURM_JOB_ID:-$$}}"
    export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-$META_DIR/mps-log}"
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

read -r -a pdb_paths <<< "$PDB_INPUTS"
read -r -a run_names <<< "${RUN_NAMES:-}"
read -r -a run_dirs <<< "${RUN_DIRS:-}"
read -r -a salts <<< "${SALTS:-}"
read -r -a temperatures <<< "${TEMPERATURES:-}"
if [ "$MAX_PARALLEL" -eq 0 ]; then
    MAX_PARALLEL="${#pdb_paths[@]}"
fi
if [ "$MAX_PARALLEL" -lt 1 ]; then
    echo "MAX_PARALLEL must be at least 1" >&2
    exit 2
fi

{
    echo "date: $(date -Is)"
    echo "repo_root: $repo_root"
    echo "output_root: $OUTPUT_ROOT"
    echo "meta_dir: $META_DIR"
    echo "target_steps: $TARGET_STEPS"
    echo "n_record: $N_RECORD"
    echo "report_interval_steps: $REPORT_INTERVAL_STEPS"
    echo "max_parallel: $MAX_PARALLEL"
    echo "use_mps: $USE_MPS"
    echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-}"
    echo "cuda_mps_pipe_directory: ${CUDA_MPS_PIPE_DIRECTORY:-}"
    echo "cuda_mps_log_directory: ${CUDA_MPS_LOG_DIRECTORY:-}"
    echo "run_names: ${RUN_NAMES:-}"
    echo "run_dirs: ${RUN_DIRS:-}"
    echo "salts: ${SALTS:-}"
    echo "temperatures: ${TEMPERATURES:-}"
} > "$META_DIR/launcher_env.txt"

index=0
for pdb_path in "${pdb_paths[@]}"; do
    run_name="$(value_at_index "" "$index" "${run_names[@]}")"
    if [ -z "$run_name" ]; then
        run_name="$(basename "$pdb_path")"
        run_name="${run_name%.*}"
    fi
    run_name="${run_name//[^A-Za-z0-9_.-]/_}"
    run_dir="$(value_at_index "" "$index" "${run_dirs[@]}")"
    if [ -z "$run_dir" ]; then
        run_dir="$OUTPUT_ROOT/$(printf "%03d" "$index")_${run_name}"
    fi
    mkdir -p "$run_dir"

    last_step="$(last_logged_step "$run_dir/log")"
    if [ "$last_step" -ge "$TARGET_STEPS" ]; then
        echo "skipping ${run_name}: already complete at step ${last_step} / ${TARGET_STEPS}" | tee "$run_dir/md.log"
        index=$(( index + 1 ))
        continue
    fi

    attempt_steps="$TARGET_STEPS"
    attempt_n_record="$N_RECORD"
    restart_path=""
    if [ "$RESTART" = "1" ] && [ -f "$run_dir/checkpoint.chk" ] && [ "$last_step" -gt 0 ]; then
        attempt_steps=$(( TARGET_STEPS - last_step ))
        if [ "$attempt_steps" -lt 1 ]; then
            attempt_steps=1
        fi
        attempt_n_record=$(( attempt_steps / REPORT_INTERVAL_STEPS ))
        if [ "$attempt_n_record" -lt 1 ]; then
            attempt_n_record=1
        fi
        restart_path="$run_dir/checkpoint.chk"
    fi

    run_salt="$(value_at_index "$SALT" "$index" "${salts[@]}")"
    run_temperature="$(value_at_index "$TEMPERATURE" "$index" "${temperatures[@]}")"

    cmd=(
        "$PYTHON_BIN" -m cranberry.cli md "$pdb_path"
        --steps "$attempt_steps"
        --n-record "$attempt_n_record"
        --output-dir "$run_dir"
        --model "$MODEL"
        --temperature "$run_temperature"
        --salt "$run_salt"
        --timestep "$STEP_SIZE"
        --platform "$OPENMM_PLATFORM"
    )

    if [ -n "$restart_path" ]; then
        cmd+=(--restart-from "$restart_path")
    fi
    if [ "${NO_OVERWRITE:-0}" = "1" ]; then
        cmd+=(--no-overwrite)
    fi
    if [ "$USE_PBC" = "1" ]; then
        cmd+=(--periodic --box-padding "$BOX_PADDING" --enforce-periodic-output)
    fi

    platform_properties="${PLATFORM_PROPERTIES:-}"
    platform_properties="${platform_properties//,/ }"
    for property in $platform_properties; do
        cmd+=(--platform-property "$property")
    done

    {
        echo "run_name: $run_name"
        echo "pdb_path: $pdb_path"
        echo "target_steps: $TARGET_STEPS"
        echo "last_step: $last_step"
        echo "attempt_steps: $attempt_steps"
        echo "target_n_record: $N_RECORD"
        echo "attempt_n_record: $attempt_n_record"
        echo "report_interval_steps: $REPORT_INTERVAL_STEPS"
        echo "temperature: $run_temperature"
        echo "salt: $run_salt"
        echo "restart_path: $restart_path"
        printf "command:"
        printf " %q" "${cmd[@]}"
        echo
    } > "$run_dir/launcher_cmd.txt"

    "${cmd[@]}" > "$run_dir/md.log" 2>&1 &
    index=$(( index + 1 ))

    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
        wait -n
    done
done

wait
