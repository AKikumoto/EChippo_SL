"""simulate.py — task-agnostic and model-agnostic training loop.

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
    Use: lambda: T_CommunityGraphDataset(...) or functools.partial.

Moving-window construction (Schapiro 2017 §2.c)
-------------------------------------------------
ECin encodes the current item at full strength and the previous item at
decayed strength to give the network a temporal asymmetry (forward bias).
This is a property of the training procedure, not the task or model, so
simulate.py constructs it here:

    a_ecin_clamp[t] = item_onehot[t] + prev_scale * item_onehot[t-1]

At t=0 (first trial of an epoch) there is no previous item; only the
current item is active (prev_scale contribution is zero).

Outputs
-------
run_epoch  returns list[dict], one entry per trial:
    {'mid': act_mid, 'm': act_m, 'p': act_p}

run_simulation  returns list[list[dict]], one inner list per epoch.

Use stack_records() to assemble a layer's activity across trials into a
(n_trials, n_units) tensor for RSA or other analyses.
"""
from __future__ import annotations

from typing import Callable, Iterable

import torch


# ---------------------------------------------------------------------------
# Analysis helper
# ---------------------------------------------------------------------------

def stack_records(
    records: list[dict],
    layer: str,
    phase: str,
) -> torch.Tensor:
    """Stack one layer's activations across trials into (n_trials, n_units).

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


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def run_epoch(
    model,
    dataloader: Iterable,
    train: bool = True,
    prev_scale: float = 0.9,
) -> list[dict]:
    """Run one epoch over all trials in dataloader.

    Moving-window ECin is constructed here from consecutive item_onehot
    entries. prev_scale=0.9 is the decayed activity of the previous item
    (Schapiro 2017 §2.c).

    Parameters
    ----------
    model       : satisfies model interface contract (see module docstring)
    dataloader  : iterable of dicts with 'item_onehot' and 'target_onehot'
    train       : if True, call model.update_weights after each trial
    prev_scale  : activity level of previous item in moving window (0.9)

    Returns
    -------
    list[dict], one per trial: {'mid': act_mid, 'm': act_m, 'p': act_p}
    """
    records: list[dict] = []
    prev_oh: torch.Tensor | None = None

    for sample in dataloader:
        cur_oh    = sample['item_onehot']
        target_oh = sample['target_onehot']

        # Moving window: first trial has no previous item
        a_ecin = cur_oh if prev_oh is None else cur_oh + prev_scale * prev_oh

        act_mid, act_m, act_p = model.run_trial(a_ecin, target_oh)
        if train:
            model.update_weights(act_m, act_p)

        records.append({'mid': act_mid, 'm': act_m, 'p': act_p})
        prev_oh = cur_oh

    return records


def run_simulation(
    model,
    dataloader_fn: Callable[[], Iterable],
    n_epochs: int,
    train: bool = True,
    prev_scale: float = 0.9,
) -> list[list[dict]]:
    """Run n_epochs epochs, generating a fresh dataloader each epoch.

    dataloader_fn() is called once per epoch so that tasks with stochastic
    random walks (T_CommunityGraphDataset, T_PairDataset) produce a new
    trial sequence every epoch, matching Schapiro (2017) §3.

    Parameters
    ----------
    model          : satisfies model interface contract
    dataloader_fn  : callable → iterable of trial dicts
    n_epochs       : number of training epochs
    train          : if False, run in evaluation mode (no weight updates)
    prev_scale     : decayed activity of previous item (default 0.9)

    Returns
    -------
    list[list[dict]] — outer index: epoch, inner index: trial

    Example
    -------
    from tasks import T_CommunityGraphDataset
    from simulate import run_simulation, stack_records

    results = run_simulation(
        model,
        dataloader_fn=lambda: T_CommunityGraphDataset(
            n_communities=3, items_per_community=5, n_steps=60
        ),
        n_epochs=10,
    )
    # CA1 activity at ActM for every trial in epoch 0:
    ca1_epoch0 = stack_records(results[0], layer='ca1', phase='m')
    """
    return [
        run_epoch(model, dataloader_fn(), train=train, prev_scale=prev_scale)
        for _ in range(n_epochs)
    ]
