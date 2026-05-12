# EChipp_SL — Oscar (Brown CCV) Cluster Scripts

SLURM scripts for the Schapiro (2017) community graph sweep on Oscar.

## Prerequisites

1. **Edit `run_one_cell.sh`**: replace `<your-ccv-account>` with your CCV group account.
2. **Conda env on Oscar**: create the `NN` environment on Oscar if it does not exist:
   ```bash
   module load anaconda3/2023.09-0-aqbc
   conda create -n NN python=3.11
   conda activate NN
   pip install torch numpy
   ```
3. **Clone the repo** on Oscar and `cd` into `cluster/Oscar/`.

## Sweep structure

| File | Purpose |
|------|---------|
| `run_one_cell.py` | Worker: runs one (condition, rep) and saves a per-cell pickle |
| `run_one_cell.sh` | SLURM batch script: activates env, calls `run_one_cell.py` |
| `submit_all.sh` | Submits the full job array; queries job count from sweep_configs |
| `aggregate.py` | Collects per-cell pickles into one `results/<sweep>.pkl` |

Conditions (defined in `simulations/sweep_configs.py`):

| Label | Description |
|-------|-------------|
| `full` | Full model: MSP + TSP both learning |
| `msp_only` | MSP only: lr_TSP=0 (TSP pathway runs but does not learn) |
| `tsp_only` | TSP only: lr_MSP=0 (MSP pathway runs but does not learn) |

Default sweep: 500 reps × 3 conditions = 1500 SLURM array jobs (each ~5 min).

## Usage

```bash
# 1. Quick pipeline test (n_epochs=2, n_reps=3)
bash submit_all.sh --smoke
# Wait for squeue -u $USER to drain

python aggregate.py --smoke
# → results/smoke.pkl

# 2. Full sweep (n_epochs=10, n_reps=500)
bash submit_all.sh
# Wait for all 1500 jobs

python aggregate.py
# → results/default.pkl
```

If any cells are missing, `aggregate.py` prints the exact `sbatch --array=...` line to resubmit.

## Output schema

`results/default.pkl` is a dict keyed by condition label:

```python
sweep['full']['acts']['ca1']['m']   # float32 (500, 10, 60, 50)
#                                          reps  epochs trials n_CA1
```

All layers: `ecin`, `dg`, `ca3`, `ca1`, `ecout`.
All phases: `mid` (cycle 25), `m` (cycle 75), `p` (cycle 100).
