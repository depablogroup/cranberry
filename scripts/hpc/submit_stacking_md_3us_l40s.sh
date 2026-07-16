#!/usr/bin/env bash
# Submit Cranberry MD production for the 16 stacking dimers, 3 us each.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
submitter="${script_dir}/submit_cranberry_md_mps_pack_l40s.sh"
repo_root="$(cd "$script_dir/../.." && pwd)"

STACKING_RUN_ROOT="${STACKING_RUN_ROOT:-${OUTPUT_ROOT:-$repo_root/runs/stacking}}"
if [[ "$STACKING_RUN_ROOT" != /* ]]; then
    STACKING_RUN_ROOT="$repo_root/$STACKING_RUN_ROOT"
fi

TARGET_STEPS="${TARGET_STEPS:-600000000}"
LOG_CADENCE_PS="${LOG_CADENCE_PS:-100}"
STEP_SIZE="${STEP_SIZE:-5}"
USE_MPS="${USE_MPS:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
SBATCH_MEM="${SBATCH_MEM:-64G}"
TEMPERATURE="${TEMPERATURE:-298}"
SALT="${SALT:-150}"
MODEL="${MODEL:-default}"

declare -A BATCH_SYSTEMS=(
    [1]="AA AU AC AG GA GC GU GG"
    [2]="CA CC CU CG UA UC UG UU"
)

if [[ -n "${BATCH:-}" ]]; then
    batches=("$BATCH")
else
    batches=(1 2)
fi

for batch in "${batches[@]}"; do
    systems="${BATCH_SYSTEMS[$batch]:-}"
    if [ -z "$systems" ]; then
        echo "ERROR: invalid BATCH=$batch (use 1 or 2)" >&2
        exit 2
    fi

    pdb_paths=""
    run_dirs=""
    for system in $systems; do
        run_dir="${STACKING_RUN_ROOT}/${system}"
        pdb="${run_dir}/${system}_cg_vs_conect.pdb"
        if [ ! -f "$pdb" ]; then
            echo "ERROR: missing $pdb" >&2
            echo "Create or copy a real stacking-pair input there before submitting." >&2
            exit 2
        fi
        pdb_paths+="${pdb} "
        run_dirs+="${run_dir} "
    done

    batch_meta_dir="${STACKING_RUN_ROOT}/_batches/batch${batch}"
    batch_job_name="${JOB_NAME:-stacking_md_b${batch}}"
    batch_run_label="${RUN_LABEL:-stacking 3us MD batch ${batch}}"

    PDB_PATHS="${pdb_paths% }" \
    RUN_NAMES="$systems" \
    RUN_DIRS="${run_dirs% }" \
    OUTPUT_ROOT="$STACKING_RUN_ROOT" \
    META_DIR="$batch_meta_dir" \
    JOB_NAME="$batch_job_name" \
    RUN_LABEL="$batch_run_label" \
    TARGET_STEPS="$TARGET_STEPS" \
    LOG_CADENCE_PS="$LOG_CADENCE_PS" \
    STEP_SIZE="$STEP_SIZE" \
    USE_MPS="$USE_MPS" \
    MAX_PARALLEL="$MAX_PARALLEL" \
    CPUS_PER_TASK="$CPUS_PER_TASK" \
    SBATCH_MEM="$SBATCH_MEM" \
    TEMPERATURE="$TEMPERATURE" \
    SALT="$SALT" \
    MODEL="$MODEL" \
    DRY_RUN="${DRY_RUN:-0}" \
    "$submitter"
done
