"""
Activation functions for the EChipp_SL hippocampal circuit.

Functions
---------
F_nxx1  : NoisyXX1 activation (Leabra; O'Reilly & Munakata 2000 Ch. 2)
F_kWTA  : k-Winners-Take-All inhibition (Schapiro 2017 §2.2)

Notes
-----
No DA-dependent gain modulation: gamma and theta are fixed across all layers.
Unlike BasalGangliaACC, there is no burst/dip modulation in this circuit.
"""

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
