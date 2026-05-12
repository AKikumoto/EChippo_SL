#!/bin/bash
# SLURM batch script for one (condition, rep) cell.
# Submitted as a job array by submit_all.sh.
# Each array slot: condition_idx = SLURM_ARRAY_TASK_ID // n_reps
#                  rep           = SLURM_ARRAY_TASK_ID  % n_reps

#SBATCH --account=<your-ccv-account>
#SBATCH --partition=batch
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1

set -eu

source /etc/profile.d/zz_activate_lmod_user.sh 2>/dev/null || true
module load anaconda3/2023.09-0-aqbc
source activate NN

python "${SLURM_SUBMIT_DIR}/run_one_cell.py" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    "$@"
