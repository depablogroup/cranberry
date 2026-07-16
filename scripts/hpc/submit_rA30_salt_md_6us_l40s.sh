#!/usr/bin/env bash
# Submit Cranberry MD production for the 04-cranberry rA30 salt conditions.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
submitter="${script_dir}/submit_cranberry_md_mps_pack_l40s.sh"
repo_root="$(cd "$script_dir/../.." && pwd)"

DISORDERED_RUN_ROOT="${DISORDERED_RUN_ROOT:-${OUTPUT_ROOT:-$repo_root/runs/disordered/rA30}}"
if [[ "$DISORDERED_RUN_ROOT" != /* ]]; then
    DISORDERED_RUN_ROOT="$repo_root/$DISORDERED_RUN_ROOT"
fi
PDB_NAME="${PDB_NAME:-rA30_20mM_0_cg_vs_conect.pdb}"

TARGET_STEPS="${TARGET_STEPS:-1200000000}"
LOG_CADENCE_PS="${LOG_CADENCE_PS:-100}"
STEP_SIZE="${STEP_SIZE:-5}"
USE_MPS="${USE_MPS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
SBATCH_MEM="${SBATCH_MEM:-32G}"
TEMPERATURE="${TEMPERATURE:-300}"
MODEL="${MODEL:-default}"

declare -A BATCH_SALTS=(
    [1]="20 100 200"
    [2]="400 600"
)

if [[ -n "${BATCH:-}" ]]; then
    batches=("$BATCH")
else
    batches=(1 2)
fi

for batch in "${batches[@]}"; do
    salts="${BATCH_SALTS[$batch]:-}"
    if [ -z "$salts" ]; then
        echo "ERROR: invalid BATCH=$batch (use 1 or 2)" >&2
        exit 2
    fi

    pdb_paths=""
    run_names=""
    run_dirs=""
    temperatures=""
    count=0
    for salt in $salts; do
        pdb_path="${DISORDERED_RUN_ROOT}/${salt}mM/${PDB_NAME}"
        if [ ! -f "$pdb_path" ]; then
            echo "ERROR: missing $pdb_path" >&2
            echo "Copy the rA30 input into each salt directory before submitting." >&2
            exit 2
        fi
        pdb_paths+="${pdb_path} "
        run_names+="${salt}mM "
        run_dirs+="${DISORDERED_RUN_ROOT}/${salt}mM "
        temperatures+="${TEMPERATURE} "
        count=$(( count + 1 ))
    done

    batch_meta_dir="${DISORDERED_RUN_ROOT}/_batches/batch${batch}"
    batch_job_name="${JOB_NAME:-rA30_md_b${batch}}"
    batch_run_label="${RUN_LABEL:-rA30 salt 6us MD batch ${batch}}"
    batch_max_parallel="${MAX_PARALLEL:-$count}"

    PDB_PATHS="${pdb_paths% }" \
    RUN_NAMES="${run_names% }" \
    RUN_DIRS="${run_dirs% }" \
    SALTS="$salts" \
    TEMPERATURES="${temperatures% }" \
    OUTPUT_ROOT="$DISORDERED_RUN_ROOT" \
    META_DIR="$batch_meta_dir" \
    JOB_NAME="$batch_job_name" \
    RUN_LABEL="$batch_run_label" \
    TARGET_STEPS="$TARGET_STEPS" \
    LOG_CADENCE_PS="$LOG_CADENCE_PS" \
    STEP_SIZE="$STEP_SIZE" \
    USE_MPS="$USE_MPS" \
    MAX_PARALLEL="$batch_max_parallel" \
    CPUS_PER_TASK="$CPUS_PER_TASK" \
    SBATCH_MEM="$SBATCH_MEM" \
    MODEL="$MODEL" \
    DRY_RUN="${DRY_RUN:-0}" \
    "$submitter"
done
