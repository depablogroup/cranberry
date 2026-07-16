#!/usr/bin/env bash
# Submit or run a 5 us ggcGCAAgcc no-PBC REMD production job on one L40S GPU.
#
# This wrapper reuses the C ACAG production submitter so restart handling,
# environment setup, logging cadence, and Slurm resource defaults stay aligned.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export JOB_NAME="${JOB_NAME:-ggcGCAAgcc_5us_nopbc}"
export RUN_LABEL="${RUN_LABEL:-ggcGCAAgcc 5us no-PBC production}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/ggcGCAAgcc_nopbc_5us_l40s}"

export PDB_PATH="${PDB_PATH:-/projects/rps/jjd8110/depablolab/cedric/04-cranberry/melting/seed/ggcGCAAgcc_cg_vs_conect.pdb}"
export ALT_PDB_PATH="${ALT_PDB_PATH:-/projects/rps/jjd8110/depablolab/cedric/04-cranberry/melting/seed/ggcGCAAgcc_extended_cg_vs_conect.pdb}"

export USE_PBC="${USE_PBC:-0}"
export BOX_PADDING="${BOX_PADDING:-2}"
export T_MIN="${T_MIN:-310}"
export T_MAX="${T_MAX:-410}"

exec "$script_dir/submit_cacag_5us_pbc_l40s.sh"
