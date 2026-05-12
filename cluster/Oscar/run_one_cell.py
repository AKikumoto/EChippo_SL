"""Run one (condition, rep) and write a per-cell pickle.

Called by run_one_cell.sh as a SLURM array job.

Task ID → (condition, rep):
    condition_idx = task_id // n_reps
    rep           = task_id  % n_reps

Output
------
results/<sweep_name>_cells/cell_<label>_rep<rep:04d>.pkl
    dict: {condition, rep, acts, metadata}
    Existing pickles are skipped (idempotent — safe to resubmit).
"""
import argparse
import os
import pickle
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SRC_DIR    = os.path.join(REPO_ROOT, 'src')
for _p in (REPO_ROOT, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from model import M_Hip
from simulate import run_simulation
from tasks import T_CommunityGraphDataset
from config.sweep_configs import DEFAULT_SWEEP, SMOKE_SWEEP, CONDITIONS

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')


def cell_path(sweep_name: str, label: str, rep: int) -> str:
    cell_dir = os.path.join(RESULTS_DIR, f'{sweep_name}_cells')
    os.makedirs(cell_dir, exist_ok=True)
    return os.path.join(cell_dir, f'cell_{label}_rep{rep:04d}.pkl')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-id', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    sw         = SMOKE_SWEEP if args.smoke else DEFAULT_SWEEP
    sweep_name = 'smoke' if args.smoke else 'default'
    n_reps     = sw['n_reps']

    condition_idx = args.task_id // n_reps
    rep           = args.task_id  % n_reps
    cond          = CONDITIONS[condition_idx]
    label         = cond['label']

    out_path = cell_path(sweep_name, label, rep)
    if os.path.exists(out_path):
        print(f'skip {out_path} (exists)', flush=True)
        return

    torch.manual_seed(rep)
    np.random.seed(rep)

    n_items      = sw['n_communities'] * sw['items_per_community']
    model_kwargs = {'n_items': n_items, **cond['model_kwargs']}
    model        = M_Hip(**model_kwargs)

    # dataloader_fn called once per epoch → fresh random walk each epoch
    def dataloader_fn():
        return T_CommunityGraphDataset(
            n_steps=sw['trials_per_epoch'],
            n_communities=sw['n_communities'],
            items_per_community=sw['items_per_community'],
        )

    result = run_simulation(
        model, dataloader_fn,
        n_epochs=sw['n_epochs'],
        prev_scale=sw['prev_scale'],
        seed=rep,
        model_kwargs=model_kwargs,
    )
    result['condition'] = label
    result['rep']       = rep

    with open(out_path, 'wb') as f:
        pickle.dump(result, f, protocol=4)
    print(f'saved {out_path}', flush=True)


if __name__ == '__main__':
    main()
