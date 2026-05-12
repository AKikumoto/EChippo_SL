"""simulate.py — model/task-agnostic training loop.

Works in two modes:
  Notebook : run_epoch / run_simulation  (interactive, returns in-memory)
  Script   : run_sequential / run_pool + run_sweep  (parameter sweep, saves to disk)

Reference design: cartpole_mpc_paper/simulations/simulation.py

Interface contracts
-------------------
model
    .run_trial(a_ecin_clamp, a_target) -> (act_mid, act_m, act_p)
        a_ecin_clamp : (n_items,) tensor — moving-window ECin pattern
        a_target     : (n_items,) tensor — next-item one-hot
        act_*        : dict with keys 'ecin', 'dg', 'ca3', 'ca1', 'ecout'
    .update_weights(act_m, act_p) -> None

dataloader
    Iterable yielding dicts per trial with at minimum:
        'item_onehot'   : (n_items,) tensor — current item one-hot
        'target_onehot' : (n_items,) tensor — next item one-hot
    Any T_*Dataset from tasks.py satisfies this contract.

dataloader_fn
    Callable() -> dataloader.
    Called once per epoch so each epoch gets a fresh random walk.

Moving-window construction (Schapiro 2017 §2.c)
-------------------------------------------------
ECin encodes the current item at full strength and the previous item at
decayed strength to give the network a temporal asymmetry (forward bias).
This is a property of the training procedure, not the task or model:

    a_ecin_clamp[t] = item_onehot[t] + prev_scale * item_onehot[t-1]

At t=0 (first trial of an epoch) there is no previous item; only the
current item is active (prev_scale contribution is zero).

Results schema
--------------
run_simulation returns a dict:
    'acts'     : dict[layer][phase] → np.ndarray float32 (n_epochs, n_trials, n_units)
                 layer : 'ecin' | 'dg' | 'ca3' | 'ca1' | 'ecout'
                 phase : 'mid'  | 'm'  | 'p'
    'metadata' : dict — n_epochs, n_trials, seed, model_kwargs, prev_scale

run_sweep aggregates over reps:
    sweep[condition_label]['acts'][layer][phase] : float32 (n_reps, n_epochs, n_trials, n_units)

Data I/O
--------
save_results / load_results  — pickle (handles nested dicts + numpy)
save_model   / load_model    — torch state_dict
"""
from __future__ import annotations

import multiprocessing
import os
import pickle
import time
from typing import Callable, Iterable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

LAYERS = ('ecin', 'dg', 'ca3', 'ca1', 'ecout')
PHASES = ('mid', 'm', 'p')


def _layer_sizes(model) -> dict:
    return {
        'ecin':  model.n_items,
        'dg':    model.n_DG,
        'ca3':   model.n_CA3,
        'ca1':   model.n_CA1,
        'ecout': model.n_items,
    }


def _empty_acts(n_epochs: int, n_trials: int, layer_sizes: dict) -> dict:
    """Pre-allocate acts[layer][phase] float32 arrays."""
    return {
        layer: {
            phase: np.empty((n_epochs, n_trials, layer_sizes[layer]), dtype=np.float32)
            for phase in PHASES
        }
        for layer in LAYERS
    }


# ---------------------------------------------------------------------------
# Core training loop — notebook-friendly
# ---------------------------------------------------------------------------

def run_epoch(
    model,
    dataloader: Iterable,
    train: bool = True,
    prev_scale: float = 0.9,
) -> list[dict]:
    """Run one epoch over all trials in dataloader.

    Returns list[dict] for interactive notebook use.
    Each dict: {'mid': act_mid, 'm': act_m, 'p': act_p}
    Pass the result to stack_records() for quick per-layer tensors,
    or use run_simulation() for structured numpy output.

    Parameters
    ----------
    model       : satisfies model interface contract (see module docstring)
    dataloader  : iterable of dicts with 'item_onehot' and 'target_onehot'
    train       : if True, call model.update_weights after each trial
    prev_scale  : activity level of previous item in moving window (0.9)
    """
    records: list[dict] = []
    prev_oh: torch.Tensor | None = None

    for sample in dataloader:
        cur_oh    = sample['item_onehot']
        target_oh = sample['target_onehot']
        a_ecin    = cur_oh if prev_oh is None else cur_oh + prev_scale * prev_oh

        act_mid, act_m, act_p = model.run_trial(a_ecin, target_oh)
        if train:
            model.update_weights(act_m, act_p)

        records.append({'mid': act_mid, 'm': act_m, 'p': act_p})
        prev_oh = cur_oh

    return records


def stack_records(
    records: list[dict],
    layer: str,
    phase: str,
) -> torch.Tensor:
    """Stack one layer's activations across trials into (n_trials, n_units).

    For notebook use with run_epoch output.

    Parameters
    ----------
    records : list[dict]  — output of run_epoch
    layer   : str         — 'ecin' | 'dg' | 'ca3' | 'ca1' | 'ecout'
    phase   : str         — 'mid'  | 'm'  | 'p'

    Example
    -------
    ca1_m = stack_records(epoch_records, layer='ca1', phase='m')
    # shape: (n_trials, n_CA1)
    """
    return torch.stack([r[phase][layer] for r in records])


def run_simulation(
    model,
    dataloader_fn: Callable[[], Iterable],
    n_epochs: int,
    train: bool = True,
    prev_scale: float = 0.9,
    seed: int | None = None,
    model_kwargs: dict | None = None,
) -> dict:
    """Run n_epochs epochs. Returns structured numpy results dict.

    dataloader_fn() is called once per epoch so that tasks with stochastic
    random walks produce a new trial sequence every epoch (Schapiro 2017 §3).

    Parameters
    ----------
    model          : satisfies model interface contract
    dataloader_fn  : callable → iterable of trial dicts
    n_epochs       : number of training epochs
    train          : if False, run in evaluation mode (no weight updates)
    prev_scale     : decayed activity of previous item (default 0.9)
    seed           : RNG seed for reproducibility
    model_kwargs   : stored in metadata for record-keeping

    Returns
    -------
    dict with:
        'acts'     : acts[layer][phase] float32 (n_epochs, n_trials, n_units)
        'metadata' : dict

    Example
    -------
    results = run_simulation(model, dataloader_fn, n_epochs=10)
    ca1_m = results['acts']['ca1']['m']   # shape: (10, 60, 50)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Run first epoch to determine n_trials and allocate arrays
    first_records = run_epoch(model, dataloader_fn(), train=train, prev_scale=prev_scale)
    n_trials = len(first_records)
    acts = _empty_acts(n_epochs, n_trials, _layer_sizes(model))

    def _write(epoch_idx: int, records: list[dict]) -> None:
        for t, rec in enumerate(records):
            for phase in PHASES:
                for layer in LAYERS:
                    acts[layer][phase][epoch_idx, t] = (
                        rec[phase][layer].detach().cpu().numpy()
                    )

    _write(0, first_records)
    for e in range(1, n_epochs):
        _write(e, run_epoch(model, dataloader_fn(), train=train, prev_scale=prev_scale))

    return {
        'acts': acts,
        'metadata': {
            'n_epochs':    n_epochs,
            'n_trials':    n_trials,
            'train':       train,
            'prev_scale':  prev_scale,
            'seed':        seed,
            'model_kwargs': model_kwargs or {},
        },
    }


# ---------------------------------------------------------------------------
# Parallelism 
# ---------------------------------------------------------------------------

def run_sequential(
    func,
    args_list,
    verbose: int = 1,
    on_result=None,
) -> list:
    """Run func over args_list in the calling process (no multiprocessing).

    Interface mirrors run_pool so callers can swap freely.
    Use this when a GPU process must own the device exclusively.
    """
    args_list = list(args_list)
    total     = len(args_list)
    t0        = time.perf_counter()
    results   = []

    if verbose > 0:
        print(f"  Sequential: {total} jobs", flush=True)

    for i, args in enumerate(args_list):
        result = func(args)
        if on_result is not None:
            on_result(result)
        else:
            results.append(result)

        done    = i + 1
        elapsed = time.perf_counter() - t0
        if verbose >= 1 and (done % 10 == 0 or done == total):
            expected = elapsed / done * total
            print(f"  {done}/{total}  elapsed={elapsed:.0f}s  ~{expected:.0f}s total",
                  flush=True)

    return results


def run_pool(
    func,
    args_list,
    n_processes: int | None = None,
    verbose: int = 1,
    on_result=None,
    maxtasksperchild: int = 20,
) -> list:
    """Run func over args_list using multiprocessing.Pool.

    maxtasksperchild=20: recycles workers after this many tasks to prevent
    memory bloat on long sweeps (same policy as cartpole_mpc_paper).

    Parameters
    ----------
    func             : callable taking a single argument (the job dict)
    args_list        : iterable of arguments to map over
    n_processes      : number of workers (default: cpu_count - 2)
    verbose          : 0 = silent, 1 = progress every 10 jobs
    on_result        : optional callback per result; skips accumulation
    maxtasksperchild : worker recycling interval
    """
    args_list = list(args_list)
    total     = len(args_list)

    if n_processes is None:
        n_processes = max(1, multiprocessing.cpu_count() - 2)

    if verbose > 0:
        print(f"  Pool: {total} jobs on {n_processes} workers "
              f"(maxtasksperchild={maxtasksperchild})", flush=True)

    t0      = time.perf_counter()
    results = []
    pool_kw = {'processes': n_processes, 'maxtasksperchild': maxtasksperchild}

    with multiprocessing.Pool(**pool_kw) as pool:
        for i, result in enumerate(pool.imap_unordered(func, args_list)):
            if on_result is not None:
                on_result(result)
            else:
                results.append(result)

            done    = i + 1
            elapsed = time.perf_counter() - t0
            if verbose >= 1 and (done % 10 == 0 or done == total):
                expected = elapsed / done * total
                print(f"  {done}/{total}  elapsed={elapsed:.0f}s  ~{expected:.0f}s total",
                      flush=True)

    return results


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def _run_condition_worker(cfg: dict) -> dict:
    """Worker function for run_sweep. One (condition, rep) per call.

    Must be a module-level function for pickle compatibility with
    multiprocessing.Pool.
    """
    import sys
    repo_src = cfg['repo_src']
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)

    from model import M_Hip

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    model  = M_Hip(**cfg['model_kwargs'])
    dl_fn  = cfg['dataloader_fn']

    result = run_simulation(
        model, dl_fn,
        n_epochs=cfg['n_epochs'],
        train=cfg['train'],
        prev_scale=cfg['prev_scale'],
        seed=cfg['seed'],
        model_kwargs=cfg['model_kwargs'],
    )
    result['condition'] = cfg['condition_label']
    result['rep']       = cfg['rep']
    return result


def run_sweep(
    conditions: list[dict],
    dataloader_fn: Callable,
    n_epochs: int,
    n_reps: int,
    base_seed: int = 0,
    train: bool = True,
    prev_scale: float = 0.9,
    n_processes: int | None = None,
    repo_src: str | None = None,
) -> dict:
    """Run all (condition × rep) combinations and return aggregated results.

    Parameters
    ----------
    conditions    : list of dicts, each with:
                    'label'        : str  — condition name
                    'model_kwargs' : dict — passed to M_Hip(...)
    dataloader_fn : picklable callable → dataloader
    n_epochs      : epochs per run
    n_reps        : independent replications per condition
    base_seed     : seed for rep i = base_seed + i
    n_processes   : None = cpu_count-2; 1 = run_sequential (no subprocess)

    Returns
    -------
    dict keyed by condition label:
        sweep[label]['acts'][layer][phase] : float32 (n_reps, n_epochs, n_trials, n_units)
        sweep[label]['metadata']

    Example
    -------
    CONDITIONS = [
        {'label': 'default',  'model_kwargs': {}},
        {'label': 'fast_MSP', 'model_kwargs': {'lr_MSP': 0.2}},
    ]
    sweep = run_sweep(CONDITIONS, dataloader_fn, n_epochs=10, n_reps=5)
    ca1_m = sweep['default']['acts']['ca1']['m']   # (5, 10, 60, 50)
    """
    if repo_src is None:
        repo_src = os.path.dirname(os.path.abspath(__file__))

    jobs = [
        {
            'repo_src':        repo_src,
            'condition_label': cond['label'],
            'model_kwargs':    cond.get('model_kwargs', {}),
            'dataloader_fn':   dataloader_fn,
            'n_epochs':        n_epochs,
            'train':           train,
            'prev_scale':      prev_scale,
            'seed':            base_seed + rep,
            'rep':             rep,
        }
        for cond in conditions
        for rep in range(n_reps)
    ]

    runner = run_sequential if n_processes == 1 else run_pool
    raw    = runner(
        _run_condition_worker, jobs,
        **({}  if n_processes == 1 else {'n_processes': n_processes}),
    )

    sweep = {}
    for cond in conditions:
        label    = cond['label']
        cond_raw = sorted([r for r in raw if r['condition'] == label],
                          key=lambda r: r['rep'])
        sweep[label] = {
            'acts': {
                layer: {
                    phase: np.stack([r['acts'][layer][phase] for r in cond_raw])
                    for phase in PHASES
                }
                for layer in LAYERS
            },
            'metadata': cond_raw[0]['metadata'] | {'n_reps': n_reps},
        }

    return sweep


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def save_results(results: dict, path: str) -> None:
    """Save results dict to pickle."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved: {path}")


def load_results(path: str) -> dict:
    """Load results dict from pickle."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_model(model, path: str) -> None:
    """Save model state dict to path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved model: {path}")


def load_model(model, path: str) -> None:
    """Load state dict into an existing model instance (in-place)."""
    model.load_state_dict(torch.load(path, weights_only=True))
    print(f"Loaded model: {path}")
