"""Aggregate per-cell pickles into one result file.

Usage (from cluster/Oscar/):
    python aggregate.py [--smoke]

Output
------
results/default.pkl  (or results/smoke.pkl)
    dict keyed by condition label:
        sweep[label]['acts'][layer][phase] : float32 (n_reps, n_epochs, n_trials, n_units)
        sweep[label]['metadata']
"""
import argparse
import os
import pickle
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SRC_DIR    = os.path.join(REPO_ROOT, 'src')
for _p in (REPO_ROOT, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.config import DEFAULT_SWEEP, SMOKE_SWEEP, CONDITIONS
from simulate import LAYERS, PHASES

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')


def load_cells(sweep_name: str, label: str, n_reps: int) -> list:
    cells   = []
    missing = []
    for rep in range(n_reps):
        path = os.path.join(
            RESULTS_DIR, f'{sweep_name}_cells', f'cell_{label}_rep{rep:04d}.pkl'
        )
        if os.path.exists(path):
            with open(path, 'rb') as f:
                cells.append(pickle.load(f))
        else:
            missing.append(rep)

    if missing:
        n_jobs  = len(CONDITIONS)
        n_reps_ = n_reps
        cond_idx = next(i for i, c in enumerate(CONDITIONS) if c['label'] == label)
        indices  = [cond_idx * n_reps_ + r for r in missing]
        print(f'  WARNING: {label}: {len(missing)} missing reps')
        print(f'  Resubmit: sbatch --array={",".join(map(str, indices))} run_one_cell.sh')

    return cells


def aggregate(cells: list) -> dict:
    cells = sorted(cells, key=lambda c: c['rep'])
    return {
        'acts': {
            layer: {
                phase: np.stack([c['acts'][layer][phase] for c in cells])
                for phase in PHASES
            }
            for layer in LAYERS
        },
        'metadata': cells[0]['metadata'] | {'n_reps': len(cells)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    sw         = SMOKE_SWEEP if args.smoke else DEFAULT_SWEEP
    sweep_name = 'smoke' if args.smoke else 'default'
    n_reps     = sw['n_reps']

    sweep = {}
    for cond in CONDITIONS:
        label = cond['label']
        print(f'Aggregating {label} ({n_reps} reps)...', flush=True)
        cells = load_cells(sweep_name, label, n_reps)
        if not cells:
            print(f'  SKIP: no cells found for {label}')
            continue
        sweep[label] = aggregate(cells)
        shape = sweep[label]['acts']['ca1']['m'].shape
        print(f'  ca1/m shape: {shape}  (n_reps, n_epochs, n_trials, n_CA1)')

    out_path = os.path.join(RESULTS_DIR, f'{sweep_name}.pkl')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(sweep, f, protocol=4)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
