#!/usr/bin/env bash
#SBATCH --job-name=cranberry-remd-mps
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

# Example: run one periodic REMD simulation with MPI ranks packed onto one GPU
# through CUDA MPS. Submit with:
#
#   sbatch scripts/hpc/submit_remd_mpi_mps_one_gpu_example_slurm.sh

CRANBERRY_BIN="${CRANBERRY_BIN:-cranberry}"
PDB_PATH="${PDB_PATH:-cranberry/data/examples/ggcGCAAgcc_cg_vs_conect.pdb}"
EXTRA_START_PDB="${EXTRA_START_PDB:-cranberry/data/examples/ggcGCAAgcc_extended_cg_vs_conect.pdb}"
OUTPUT_DIR="${OUTPUT_DIR:-remd-run}"
STEPS="${STEPS:-100000}"
N_REPLICAS="${N_REPLICAS:-${SLURM_NTASKS:-8}}"

mkdir -p "$OUTPUT_DIR"

export OPENMMTOOLS_ENABLE_MPI="${OPENMMTOOLS_ENABLE_MPI:-1}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/cranberry-remd-mps-${USER:-user}-${SLURM_JOB_ID:-$$}}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-$OUTPUT_DIR/mps-log}"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

mps_started=0
stop_mps() {
    if [ "$mps_started" -eq 1 ]; then
        echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    fi
}
trap stop_mps EXIT

if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    nvidia-cuda-mps-control -d
    mps_started=1
else
    echo "warning: nvidia-cuda-mps-control not found; continuing without CUDA MPS" >&2
fi

srun -n "$N_REPLICAS" "$CRANBERRY_BIN" remd "$PDB_PATH" \
    --extra-start-pdb "$EXTRA_START_PDB" \
    --periodic \
    --steps "$STEPS" \
    --n-replicas "$N_REPLICAS" \
    --platform CUDA \
    --output-dir "$OUTPUT_DIR"
