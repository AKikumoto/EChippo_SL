"""
Utility functions for the EChipp_SL hippocampal circuit.

Model components (used inside layer.py / model.py)
---------------------------------------------------
F_nxx1        : NoisyXX1 activation (Leabra; O'Reilly & Munakata 2000 Ch. 2)
F_kWTA        : k-Winners-Take-All inhibition (Schapiro 2017 §2.2)
F_init_weights: uniform weight init per Schapiro (2017) SI Table 2
NET_SCALE     : forward-pass scale factors per projection

Analysis (used in notebooks and cluster/aggregate.py)
------------------------------------------------------
F_item_mean_vecs      : mean CA1 activity per item (normalized)
F_community_masks     : within- / between-community boolean masks
F_rsa_by_epoch        : RSA scores across all epochs of a run_simulation result
F_extract_rsa         : Pearson r similarity matrices across reps for a chosen snapshot
F_pattern_sim_by_mask : mean ± SE Pearson r per layer per boolean mask, across reps

Notes
-----
No DA-dependent gain modulation: gamma and theta are fixed across all layers.
Unlike BasalGangliaACC, there is no burst/dip modulation in this circuit.
"""

from collections.abc import Callable

import numpy as np
import rsatoolbox
import torch
import torch.nn as nn


# =============================================================================
# F_nxx1  —  NoisyXX1 activation function
# =============================================================================
# [Formula]:
#   XX1:   y = gamma * [Vm - theta]+ / (gamma * [Vm - theta]+ + 1)
#          = u / (u + 1),  where u = gamma * relu(Vm - theta)
#
# [NoisyXX1]:
#   Convolve XX1 with a Gaussian kernel N(0, sigma^2).
#   Produces a smooth, differentiable activation; models neural input noise.
#   O'Reilly & Munakata (2000) Ch. 2: "noisy XX1 with Gaussian convolution"
#
# [Parameters — Leabra defaults]:
#   gamma = 600   (O'Reilly & Munakata 2000)
#   theta = 0.25  (O'Reilly & Munakata 2000)
#   sigma = 0.005 (kernel width; same as BasalGangliaACC)
#
# [Usage]:
#   y = F_nxx1(vm)                          # scalar or any-shape tensor
#   y = F_nxx1(vm, gamma=600.0, theta=0.25) # explicit params

def F_nxx1(
    vm: torch.Tensor,
    *,
    gamma: float = 600.0,
    theta: float = 0.25,
    sigma: float = 0.005,
    n_kernel: int = 61,
) -> torch.Tensor:
    """NoisyXX1 activation — O'Reilly & Munakata (2000) Ch. 2 Eq. 2.12.

    Parameters
    ----------
    vm : Tensor
        Input (membrane potential), any shape.
    gamma : float
        Gain. Default 600 (O'Reilly & Munakata 2000).
    theta : float
        Threshold. Default 0.25 (O'Reilly & Munakata 2000).
    sigma : float
        Gaussian kernel width. Default 0.005.
    n_kernel : int
        Number of kernel points (must be odd). Default 61.

    Returns
    -------
    Tensor, same shape as vm, values in [0, 1].
    """
    half = 3.0 * sigma
    offsets = torch.linspace(-half, half, n_kernel, device=vm.device, dtype=vm.dtype)
    weights = torch.exp(-0.5 * (offsets / sigma) ** 2)
    weights = weights / weights.sum()

    # XX1 evaluated at vm - theta - offset for each kernel point
    v_shifted = vm.unsqueeze(-1) - theta - offsets   # (..., K)
    u = torch.relu(v_shifted) * gamma
    xx1 = u / (u + 1.0)

    return (xx1 * weights).sum(dim=-1)


# =============================================================================
# F_kWTA  —  k-Winners-Take-All inhibition
# =============================================================================
# [Role]:
#   Implements lateral inhibition: only the top-k units remain active.
#   Non-top-k units are suppressed to exactly zero.
#   Active units retain their original (pre-inhibition) values.
#
# [Sparsity targets — Schapiro (2017) SI Table 1]:
#   DG    : k_frac = 0.01  (~1%  active; strong pattern separation)
#   CA3   : k_frac = 0.06  (~6%  active)
#   CA1   : k_frac = 0.25  (~25% active; less sparse than DG/CA3)
#   ECout : absolute k=2  (matched to ECin)
#
# [Tie-breaking]:
#   torch.topk breaks ties arbitrarily. For equal-valued units exactly at
#   the threshold, units not in topk are suppressed; this is deterministic
#   given the same input tensor but not guaranteed to be consistent across
#   PyTorch versions.
#
# [Usage]:
#   out = F_kWTA(net_input, k_frac=0.10)

def F_kWTA(
    activity: torch.Tensor,
    *,
    k_frac: float,
) -> torch.Tensor:
    """k-Winners-Take-All inhibition — Schapiro (2017) §2.2.

    Parameters
    ----------
    activity : Tensor, shape (..., n_units)
        Pre-inhibition activity (net input or firing rate).
    k_frac : float
        Fraction of units to keep active. k = max(1, floor(k_frac * n_units)).

    Returns
    -------
    Tensor, same shape as activity.
        Top-k units retain original values; others are set to zero.
    """
    n_units = activity.shape[-1]
    n_active = max(1, int(k_frac * n_units))

    # Find the n_active-th largest value (threshold)
    # kthvalue(k) returns the k-th *smallest*, so we invert: position from bottom
    k_from_bottom = n_units - n_active + 1
    threshold = torch.kthvalue(activity, k_from_bottom, dim=-1).values

    mask = (activity >= threshold.unsqueeze(-1)).float()
    return activity * mask


# =============================================================================
# WEIGHT INITIALIZATION — Schapiro (2017) SI Table 2
# =============================================================================
# All weight initialization ranges and forward-time scale factors live here.
# Change a value once to update every projection that uses it.

# Weight initialization ranges per projection type.
# Schapiro (2017) SI Table 2, "Weight range" column.
_W_INIT: dict[str, tuple[float, float]] = {
    'default':     (0.25, 0.75),  # SI Table 2: all projections except the two below
    'mossy_fiber': (0.89, 0.91),  # SI Table 2: DG → CA3 detonator synapse (narrow high range)
    'big_loop':    (0.49, 0.51),  # SI Table 2: ECout → ECin back-projection
}

# Forward-time scale factors (multiplier on net input from a projection).
# Schapiro (2017) SI Table 2, "Scale (abs/rel)" column, abs value.
# Applied inside layer forward() so that pathway dominance matches the paper.
NET_SCALE: dict[str, float] = {
    'ecin_ca1':   3.0,  # SI Table 2 abs=3: ECin → CA1 (MSP direct path dominates Q1)
    'ecout_ecin': 2.0,  # SI Table 2 abs=2: ECout → ECin (big loop back-projection)
}


def F_init_weights(
    shape: tuple,
    projection: str = 'default',
    mask: torch.Tensor | None = None,
) -> nn.Parameter:
    """Uniform weight initialization per Schapiro (2017) SI Table 2.

    Parameters
    ----------
    shape : tuple
        Weight matrix shape (n_pre, n_post).
    projection : str
        Key into _W_INIT. 'default' for most projections.
    mask : Tensor or None
        If given, non-connected positions (mask == 0) are zeroed out.

    Returns
    -------
    nn.Parameter with requires_grad=False (CHL mode; switch with
    .requires_grad_(True) at the model level for backprop mode).
    """
    lo, hi = _W_INIT[projection]
    w = torch.empty(shape).uniform_(lo, hi)
    if mask is not None:
        w = w * mask
    return nn.Parameter(w, requires_grad=False)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
# Used in notebooks and cluster/aggregate.py to evaluate simulation results.
# These functions operate on run_simulation output (numpy float32 arrays).


def F_item_mean_vecs(
    ca1_acts: np.ndarray,
    ecin_acts: np.ndarray,
) -> torch.Tensor:
    """Mean CA1 activity vector per item, L2-normalized.

    Computes the average CA1 pattern across all trials in which each item
    was active in ECin, then normalizes each row to unit length.
    Items with no active trials fall back to the global mean.

    Parameters
    ----------
    ca1_acts  : float32 array (n_trials, n_CA1)   — one epoch of CA1 activity
    ecin_acts : float32 array (n_trials, n_items)  — one epoch of ECin activity

    Returns
    -------
    Tensor (n_items, n_CA1), L2-normalized row vectors.
    """
    ca1  = torch.as_tensor(ca1_acts)
    ecin = torch.as_tensor(ecin_acts)
    n_items = ecin.shape[1]

    vecs = []
    for i in range(n_items):
        mask = ecin[:, i] > 0.5
        vecs.append(ca1[mask].mean(0) if mask.sum() > 0 else ca1.mean(0))
    mat = torch.stack(vecs)                                # (n_items, n_CA1)
    return mat / (mat.norm(dim=1, keepdim=True) + 1e-8)   # L2-normalize



def F_community_masks(
    n_items: int,
    items_per_community: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean masks for within- and between-community item pairs.

    Assumes equal-sized communities: items 0..k-1 are community 0,
    items k..2k-1 are community 1, etc.

    Parameters
    ----------
    n_items             : total number of items
    items_per_community : items per community

    Returns
    -------
    within_mask  : BoolTensor (n_items, n_items) — within-community, off-diagonal
    between_mask : BoolTensor (n_items, n_items) — between-community pairs
    """
    community   = torch.tensor([i // items_per_community for i in range(n_items)])
    same_comm   = community.unsqueeze(1) == community.unsqueeze(0)
    within_mask  = same_comm & ~torch.eye(n_items, dtype=torch.bool)
    between_mask = ~same_comm
    return within_mask, between_mask


def F_rsa_by_epoch(
    result: dict,
    n_items: int,
    items_per_community: int,
) -> tuple[list[float], list[float]]:
    """Within- and between-community RSA scores for every epoch.

    Convenience wrapper around F_item_mean_vecs and F_community_masks.
    Operates on run_simulation output.

    Parameters
    ----------
    result              : run_simulation output dict
    n_items             : total number of items
    items_per_community : items per community

    Returns
    -------
    within_scores  : list[float], length n_epochs
    between_scores : list[float], length n_epochs

    Example
    -------
    within, between = F_rsa_by_epoch(result, n_items=15, items_per_community=5)
    plt.plot(within, label='within')
    plt.plot(between, label='between')
    """
    n_epochs = result['acts']['ca1']['m'].shape[0]
    within_mask, between_mask = F_community_masks(n_items, items_per_community)
    wm = within_mask.numpy()
    bm = between_mask.numpy()

    within_scores, between_scores = [], []
    for ep in range(n_epochs):
        vecs = F_item_mean_vecs(result['acts']['ca1']['m'][ep],
                                result['acts']['ecin']['m'][ep])
        rdm = rsatoolbox.rdm.calc_rdm(
            rsatoolbox.data.Dataset(vecs.numpy().astype(np.float64)),
            method='correlation',
        )
        sim = 1.0 - rdm.get_matrices()[0]
        within_scores.append(float(sim[wm].mean()))
        between_scores.append(float(sim[bm].mean()))

    return within_scores, between_scores


def F_extract_rsa(
    results: dict,
    snap_fn: Callable,
    layers: tuple = ('dg', 'ca3', 'ca1'),
) -> dict:
    """Pearson r RSA matrices per layer, stacked across reps.

    Parameters
    ----------
    results  : run_simulation output dict
    snap_fn  : callable — takes results['eval'][r] (list of epoch evals) and
               returns a layer-keyed dict of activity arrays (n_items, n_units).
               Examples:
                 lambda e: e[-1]       final epoch, settled representations
                 lambda e: e[0][0]     pre-training snap20
                 lambda e: e[-1][1]    post-training snap80
    layers   : tuple of layer names (default: ('dg', 'ca3', 'ca1'))

    Returns
    -------
    dict[layer] → ndarray (n_reps, n_items, n_items)
    """
    n_reps = results['metadata']['n_reps']
    # normalize: n_reps=1 backward-compat squeezes the rep dimension
    evals = [results['eval']] if n_reps == 1 else results['eval']
    return {
        l: np.stack([
            1.0 - rsatoolbox.rdm.calc_rdm(
                rsatoolbox.data.Dataset(snap_fn(evals[r])[l].astype(np.float64)),
                method='correlation',
            ).get_matrices()[0]
            for r in range(n_reps)
        ])
        for l in layers
    }


def F_pattern_sim_by_mask(
    rsa_dict: dict,
    masks: dict,
) -> dict:
    """Mean ± SE Pearson r per layer per boolean mask, across reps.

    Parameters
    ----------
    rsa_dict : dict[layer] → ndarray (n_reps, n_items, n_items)
    masks    : dict[name]  → bool ndarray (n_items, n_items)

    Returns
    -------
    dict[layer][name] → (mean: float, se: float)

    Example
    -------
    sim = F_pattern_sim_by_mask(rsa, {'same': same_mask, 'cross': cross_mask})
    print(sim['ca1']['same'])   # (mean, se) across reps
    """
    result: dict = {}
    for layer, rsas in rsa_dict.items():
        result[layer] = {}
        for name, mask in masks.items():
            vals = rsas[:, mask].mean(axis=1)   # (n_reps,)
            result[layer][name] = (
                float(vals.mean()),
                float(vals.std() / len(vals) ** 0.5),
            )
    return result
