#!/usr/bin/env bash
# Source this before launching OpenMPI/PMIx jobs on clusters where munge is not available.

# Interactive Slurm shells can leave PMI/PMIx/OMPI rank bootstrap variables in
# the environment. If a user then runs bare `python -c "from mpi4py import MPI"`,
# OpenMPI may try to attach to that stale parent PMIx server and hang or fail.
# Clear only rank/bootstrap variables; keep SLURM_* allocation metadata intact.
unset PMIX_RANK
unset PMIX_NAMESPACE
unset PMIX_SERVER_URI
unset PMIX_SERVER_URI2
unset PMIX_ID
unset PMIX_LOCAL_RANK
unset PMIX_LOCAL_SIZE
unset PMIX_LOCAL_PEERS
unset PMIX_HOSTNAME
unset PMIX_SECURITY_MODE
unset PMI_RANK
unset PMI_SIZE
unset PMI_FD
unset PMI_PORT
unset PMI_ID
unset PMI_KVSNAME
unset OMPI_COMM_WORLD_RANK
unset OMPI_COMM_WORLD_SIZE
unset OMPI_COMM_WORLD_LOCAL_RANK
unset OMPI_COMM_WORLD_LOCAL_SIZE
unset OMPI_UNIVERSE_SIZE

export PMIX_MCA_psec="${PMIX_MCA_psec:-^munge}"
export OMPI_MCA_psec="${OMPI_MCA_psec:-^munge}"
export PRTE_MCA_psec="${PRTE_MCA_psec:-^munge}"
