#!/usr/bin/env bash
# Example Slurm submitter for packing independent Cranberry MD runs onto one GPU.
#
# Minimal usage:
#   scripts/hpc/submit_md_mps_pack_example_slurm.sh \
#       --pdb /path/a_cg_vs_conect.pdb \
#       --pdb /path/b_cg_vs_conect.pdb \
#       --steps 100000 \
#       --platform CUDA \
#       --output-root runs/md-pack \
#       --account my_account \
#       --partition gpu

set -euo pipefail

script_path="$(realpath "${BASH_SOURCE[0]}")"
script_dir="$(cd "$(dirname "$script_path")" && pwd)"
repo_root="${CRANBERRY_REPO_ROOT:-$(cd "$script_dir/../.." && pwd)}"

PDB_PATHS="${PDB_PATHS:-}"
RUN_NAMES="${RUN_NAMES:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$repo_root/runs/md-pack}"
STEPS="${STEPS:-${TARGET_STEPS:-}}"
N_RECORD="${N_RECORD:-}"
MODEL="${MODEL:-default}"
TEMPERATURE="${TEMPERATURE:-298}"
SALT="${SALT:-150}"
STEP_SIZE="${STEP_SIZE:-5}"
OPENMM_PLATFORM="${OPENMM_PLATFORM:-CUDA}"
USE_MPS="${USE_MPS:-1}"
PERIODIC="${PERIODIC:-0}"
BOX_PADDING="${BOX_PADDING:-3.0}"

JOB_NAME="${JOB_NAME:-cranberry_md_pack}"
ACCOUNT="${ACCOUNT:-}"
PARTITION="${PARTITION:-}"
CONSTRAINT="${CONSTRAINT:-}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
SBATCH_MEM="${SBATCH_MEM:-32G}"
SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
NUM_GPUS="${NUM_GPUS:-1}"
DRY_RUN="${DRY_RUN:-0}"

CONDA_ENV="${CONDA_ENV:-cranberry-dev}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-${HOME:-}/.conda/envs/$CONDA_ENV}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CRANBERRY_BIN="${CRANBERRY_BIN:-cranberry}"

usage() {
    cat <<'EOF'
usage: submit_md_mps_pack_example_slurm.sh [OPTIONS]

Submit one Slurm job that runs multiple independent `cranberry md` commands on
one GPU. CUDA MPS is started inside the allocation when available.

Run selection:
  --pdb PATH             Add one canonical CRANBERRY CG PDB. Repeat for multiple runs.
  --pdb-paths "PATHS"    Space-separated PDB list; same as PDB_PATHS.
  --run-name NAME        Optional per-run name. Repeat in --pdb order.

cranberry md options:
  --steps N              Integration steps per run. Required unless STEPS is set.
  --n-record N           Optional target number of trajectory/log records.
  --model NAME           Force-field model name.
  --temperature K        Temperature in kelvin.
  --salt MM              Salt concentration in millimolar.
  --timestep FS          Integration timestep in femtoseconds.
  --platform NAME        OpenMM platform name.
  --periodic             Add --periodic and --enforce-periodic-output.
  --box-padding NM       Periodic cubic box padding.
  --output-root PATH     Root directory for per-run output directories.

Slurm and environment options:
  --job-name NAME        Slurm job name.
  --account NAME         Slurm account.
  --partition NAME       Slurm partition.
  --constraint NAME      Slurm constraint.
  --cpus-per-task N      Slurm CPUs per task.
  --mem SIZE             Slurm memory request.
  --time HH:MM:SS        Slurm walltime.
  --num-gpus N           Number of GPUs requested from Slurm.
  --use-mps / --no-mps   Start or skip CUDA MPS control daemon.
  --conda-env NAME       Conda environment name.
  --conda-env-path PATH  Conda environment path.
  --python-bin PATH      Python executable for import preflight.
  --cranberry-bin PATH   Cranberry CLI executable.
  --dry-run              Print sbatch command without submitting.
  --help                 Show this help.
EOF
}

append_value() {
    local current="$1"
    local value="$2"
    if [ -n "$current" ]; then
        printf "%s %s" "$current" "$value"
    else
        printf "%s" "$value"
    fi
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --pdb)
            PDB_PATHS="$(append_value "$PDB_PATHS" "${2:?--pdb requires a path}")"
            shift 2
            ;;
        --pdb-paths)
            PDB_PATHS="${2:?--pdb-paths requires a value}"
            shift 2
            ;;
        --run-name)
            RUN_NAMES="$(append_value "$RUN_NAMES" "${2:?--run-name requires a value}")"
            shift 2
            ;;
        --steps)
            STEPS="${2:?--steps requires a value}"
            shift 2
            ;;
        --n-record)
            N_RECORD="${2:?--n-record requires a value}"
            shift 2
            ;;
        --model)
            MODEL="${2:?--model requires a value}"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="${2:?--temperature requires a value}"
            shift 2
            ;;
        --salt)
            SALT="${2:?--salt requires a value}"
            shift 2
            ;;
        --timestep)
            STEP_SIZE="${2:?--timestep requires a value}"
            shift 2
            ;;
        --platform)
            OPENMM_PLATFORM="${2:?--platform requires a value}"
            shift 2
            ;;
        --periodic)
            PERIODIC=1
            shift
            ;;
        --box-padding)
            BOX_PADDING="${2:?--box-padding requires a value}"
            shift 2
            ;;
        --output-root|--output-dir)
            OUTPUT_ROOT="${2:?--output-root requires a path}"
            shift 2
            ;;
        --job-name)
            JOB_NAME="${2:?--job-name requires a value}"
            shift 2
            ;;
        --account)
            ACCOUNT="${2:?--account requires a value}"
            shift 2
            ;;
        --partition)
            PARTITION="${2:?--partition requires a value}"
            shift 2
            ;;
        --constraint)
            CONSTRAINT="${2:?--constraint requires a value}"
            shift 2
            ;;
        --cpus-per-task)
            CPUS_PER_TASK="${2:?--cpus-per-task requires a value}"
            shift 2
            ;;
        --mem)
            SBATCH_MEM="${2:?--mem requires a value}"
            shift 2
            ;;
        --time)
            SBATCH_TIME="${2:?--time requires a value}"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="${2:?--num-gpus requires a value}"
            shift 2
            ;;
        --use-mps)
            USE_MPS=1
            shift
            ;;
        --no-mps)
            USE_MPS=0
            shift
            ;;
        --conda-env)
            CONDA_ENV="${2:?--conda-env requires a value}"
            shift 2
            ;;
        --conda-env-path)
            CONDA_ENV_PATH="${2:?--conda-env-path requires a path}"
            shift 2
            ;;
        --python-bin)
            PYTHON_BIN="${2:?--python-bin requires a path}"
            shift 2
            ;;
        --cranberry-bin)
            CRANBERRY_BIN="${2:?--cranberry-bin requires a path}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ -z "$PDB_PATHS" ]; then
    echo "ERROR: pass one or more --pdb options, or set PDB_PATHS." >&2
    usage >&2
    exit 2
fi
if [ -z "$STEPS" ]; then
    echo "ERROR: pass --steps, or set STEPS." >&2
    usage >&2
    exit 2
fi
if [[ "$OUTPUT_ROOT" != /* ]]; then
    OUTPUT_ROOT="$repo_root/$OUTPUT_ROOT"
fi

submit_self() {
    mkdir -p "$OUTPUT_ROOT"

    export CRANBERRY_REPO_ROOT="$repo_root"
    export PDB_PATHS RUN_NAMES OUTPUT_ROOT STEPS N_RECORD MODEL TEMPERATURE SALT STEP_SIZE
    export OPENMM_PLATFORM USE_MPS PERIODIC BOX_PADDING
    export CONDA_ENV CONDA_ENV_PATH PYTHON_BIN CRANBERRY_BIN

    sbatch_args=(
        --job-name="$JOB_NAME"
        --nodes=1
        --ntasks=1
        --cpus-per-task="$CPUS_PER_TASK"
        --gres="gpu:$NUM_GPUS"
        --mem="$SBATCH_MEM"
        --time="$SBATCH_TIME"
        --chdir="$OUTPUT_ROOT"
        --output=std.out
        --open-mode=truncate
        --export=ALL
    )
    if [ -n "$ACCOUNT" ]; then
        sbatch_args+=(--account="$ACCOUNT")
    fi
    if [ -n "$PARTITION" ]; then
        sbatch_args+=(--partition="$PARTITION")
    fi
    if [ -n "$CONSTRAINT" ]; then
        sbatch_args+=(--constraint="$CONSTRAINT")
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
        exit 2
    fi
    if ! command -v "$CRANBERRY_BIN" >/dev/null 2>&1; then
        echo "ERROR: CRANBERRY_BIN is not executable or not on PATH: $CRANBERRY_BIN" >&2
        exit 2
    fi

    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$repo_root"
    "$PYTHON_BIN" -c "import openmm, cranberry; print('preflight: openmm cranberry OK')"
}

start_mps() {
    mps_started=0
    stop_mps() {
        if [ "$mps_started" -eq 1 ] && command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
            echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
        fi
    }
    trap stop_mps EXIT

    if [ "$USE_MPS" != "1" ]; then
        return
    fi

    export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/cranberry-md-mps-${USER:-user}-${SLURM_JOB_ID:-$$}}"
    export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-$OUTPUT_ROOT/mps-log}"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
        if nvidia-cuda-mps-control -d; then
            mps_started=1
        else
            echo "warning: failed to start CUDA MPS control daemon; continuing" >&2
        fi
    else
        echo "warning: nvidia-cuda-mps-control not found; continuing without CUDA MPS" >&2
    fi
}

run_name_for_index() {
    local index="$1"
    local pdb_path="$2"
    shift 2
    local names=("$@")
    local name=""
    if [ "${#names[@]}" -gt "$index" ]; then
        name="${names[$index]}"
    fi
    if [ -z "$name" ]; then
        name="$(basename "$pdb_path")"
        name="${name%.*}"
    fi
    echo "${name//[^A-Za-z0-9_.-]/_}"
}

run_packed_md() {
    mkdir -p "$OUTPUT_ROOT"
    prepare_python_env
    start_mps

    read -r -a pdb_paths <<< "$PDB_PATHS"
    read -r -a run_names <<< "$RUN_NAMES"

    {
        echo "date: $(date -Is)"
        echo "repo_root: $repo_root"
        echo "output_root: $OUTPUT_ROOT"
        echo "steps: $STEPS"
        echo "n_record: $N_RECORD"
        echo "platform: $OPENMM_PLATFORM"
        echo "use_mps: $USE_MPS"
        echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-}"
        echo "cuda_mps_pipe_directory: ${CUDA_MPS_PIPE_DIRECTORY:-}"
        echo "cuda_mps_log_directory: ${CUDA_MPS_LOG_DIRECTORY:-}"
        echo "pdb_paths: $PDB_PATHS"
        echo "run_names: $RUN_NAMES"
    } > "$OUTPUT_ROOT/launcher_env.txt"

    local index=0
    local pdb_path run_name run_dir
    for pdb_path in "${pdb_paths[@]}"; do
        run_name="$(run_name_for_index "$index" "$pdb_path" "${run_names[@]}")"
        run_dir="$OUTPUT_ROOT/$(printf "%03d" "$index")_${run_name}"
        mkdir -p "$run_dir"

        cmd=(
            "$CRANBERRY_BIN" md "$pdb_path"
            --steps "$STEPS"
            --platform "$OPENMM_PLATFORM"
            --output-dir "$run_dir"
            --model "$MODEL"
            --temperature "$TEMPERATURE"
            --salt "$SALT"
            --timestep "$STEP_SIZE"
        )
        if [ -n "$N_RECORD" ]; then
            cmd+=(--n-record "$N_RECORD")
        fi
        if [ "$PERIODIC" = "1" ]; then
            cmd+=(--periodic --box-padding "$BOX_PADDING" --enforce-periodic-output)
        fi

        {
            echo "run_name: $run_name"
            echo "pdb_path: $pdb_path"
            printf "command:"
            printf " %q" "${cmd[@]}"
            echo
        } > "$run_dir/launcher_cmd.txt"

        "${cmd[@]}" > "$run_dir/md.log" 2>&1 &
        index=$(( index + 1 ))
    done

    wait
}

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Submitting $JOB_NAME to Slurm"
    echo "  output_root=$OUTPUT_ROOT"
    echo "  pdb_count=$(wc -w <<< "$PDB_PATHS")"
    echo "  steps=$STEPS platform=$OPENMM_PLATFORM use_mps=$USE_MPS"
    echo "  account=${ACCOUNT:-unset} partition=${PARTITION:-unset} constraint=${CONSTRAINT:-unset}"
    echo "  dry_run=$DRY_RUN"
    submit_self
    exit 0
fi

cd "$repo_root"
run_packed_md
