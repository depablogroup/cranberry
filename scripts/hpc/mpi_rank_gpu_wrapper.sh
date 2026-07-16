#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: mpi_rank_gpu_wrapper.sh COMMAND [ARGS...]" >&2
    exit 2
fi

local_rank="${OMPI_COMM_WORLD_LOCAL_RANK:-${MPI_LOCALRANKID:-${SLURM_LOCALID:-${PMIX_LOCAL_RANK:-0}}}}"
local_size="${OMPI_COMM_WORLD_LOCAL_SIZE:-${SLURM_NTASKS_PER_NODE:-${N_MPI_RANKS:-1}}}"
num_gpus="${NUM_GPUS:-1}"

if [ "$num_gpus" -lt 1 ]; then
    echo "NUM_GPUS must be at least 1" >&2
    exit 2
fi

ranks_per_gpu="${RANKS_PER_GPU:-}"
if [ -z "$ranks_per_gpu" ]; then
    ranks_per_gpu=$(( (local_size + num_gpus - 1) / num_gpus ))
fi
if [ "$ranks_per_gpu" -lt 1 ]; then
    echo "RANKS_PER_GPU must be at least 1" >&2
    exit 2
fi

gpu_slot=$(( local_rank / ranks_per_gpu ))
if [ "$gpu_slot" -ge "$num_gpus" ]; then
    gpu_slot=$(( num_gpus - 1 ))
fi

parent_visible="${PARENT_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
if [ -n "$parent_visible" ]; then
    IFS=',' read -r -a visible_gpus <<< "$parent_visible"
    if [ "${#visible_gpus[@]}" -gt "$gpu_slot" ]; then
        export CUDA_VISIBLE_DEVICES="${visible_gpus[$gpu_slot]}"
    else
        last_index=$(( ${#visible_gpus[@]} - 1 ))
        export CUDA_VISIBLE_DEVICES="${visible_gpus[$last_index]}"
    fi
else
    export CUDA_VISIBLE_DEVICES="$gpu_slot"
fi

export RANKS_PER_GPU="$ranks_per_gpu"
export PARENT_CUDA_VISIBLE_DEVICES="$parent_visible"

exec "$@"
