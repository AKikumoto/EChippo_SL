#!/bin/bash
# Submit a SLURM job array for the EChipp_SL community graph sweep on Oscar.
#
# Usage (from cluster/Oscar/):
#   bash submit_all.sh [--smoke]
#
#   --smoke    SMOKE_SWEEP (n_epochs=2, n_reps=3): run first to test the pipeline.
#
# After all jobs finish:
#   python aggregate.py [--smoke]

set -eu

SMOKE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE="--smoke"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p log

source /etc/profile.d/zz_activate_lmod_user.sh 2>/dev/null || true
module load anaconda3/2023.09-0-aqbc
source activate NN

N_JOBS=$(python -c "
import sys
sys.path.insert(0, '${REPO_ROOT}')
sys.path.insert(0, '${REPO_ROOT}/src')
from config.sweep_configs import DEFAULT_SWEEP, SMOKE_SWEEP, CONDITIONS
sw = SMOKE_SWEEP if '${SMOKE}' else DEFAULT_SWEEP
print(len(CONDITIONS) * sw['n_reps'])
")

SWEEP_NAME=$([ -n "$SMOKE" ] && echo "smoke" || echo "default")
JOB_NAME="echipp_${SWEEP_NAME}"
TIME=$([ -n "$SMOKE" ] && echo "00:05:00" || echo "00:15:00")

sbatch \
    --job-name="${JOB_NAME}" \
    --time="${TIME}" \
    --mem="4G" \
    --cpus-per-task=1 \
    --nodes=1 \
    --array=0-$((N_JOBS - 1)) \
    -o "log/${JOB_NAME}_%a.%j.out" \
    -e "log/${JOB_NAME}_%a.%j.err" \
    run_one_cell.sh ${SMOKE}

echo "Submitted ${N_JOBS} jobs  (sweep=${SWEEP_NAME})"
echo "Monitor:      squeue -u \$USER"
echo "After finish: python aggregate.py $([ -n "$SMOKE" ] && echo "--smoke")"
