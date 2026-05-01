"""
layer.py: neural layers for the EC-hippocampus statistical learning circuit.

| Layer   | Description               | Role                                          |
|---------|---------------------------|-----------------------------------------------|
| L_ECin  | Entorhinal cortex input   | Input driver; one-hot item encoding           |
| L_DG    | Dentate gyrus             | Pattern separation; sparse kWTA (~1%)         |
| L_CA3   | CA3 field                 | Pattern completion; recurrent attractor       |
| L_CA1   | CA1 field                 | MSP + TSP convergence; output readout         |
| L_ECout | Entorhinal cortex output  | Reconstruction target; plus-phase teacher     |

Each layer is a circuit component from Schapiro et al. (2017).
See config/ARCHITECTURE_ENG.md for equations, parameters, and CHL settling policy.

Naming conventions:
  L_* : layer modules (nn.Module)
  M_* : full model classes (model.py)
  F_* : utility functions (util.py)
  T_* : task environments (tasks.py)
"""

import torch
import torch.nn as nn

# F_nxx1, F_kWTA defined in src/util.py (Step 1)
from util import F_nxx1, F_kWTA


# =========================================================================
# L_ECin: ENTORHINAL CORTEX INPUT
# =========================================================================
# [Role]:
#   Input driver. Encodes the current item as a localist (one-hot) or
#   distributed pattern of activity. ECin is always clamped to the
#   stimulus pattern — it does not settle and has no learnable weights.
#
# [Inputs]:
#   - item_idx : int or long tensor — index of the current item (0..n_items-1)
#
# [Outputs]:
#   - activity : (n_items,) one-hot float tensor
#
# [Learning]:
#   None. ECin is a fixed input driver.
#
# [Connections]:
#   ECin → CA1  (MSP: monosynaptic pathway; slow, W_ECin_CA1)
#   ECin → DG   (TSP first leg; fast, W_ECin_DG)
#
# [Notes]:
#   - Schapiro (2017) §2.1: "EC input layer uses localist representations"
#   - In minus phase: clamped to current item
#   - In plus phase: same clamping (ECin does not change between phases)
#   - ECin is stateless: no _activity buffer, no Euler integration needed

class L_ECin(nn.Module):
    """Entorhinal cortex input: item index → one-hot activity pattern.

    Stateless. Call forward(item_idx) to get the activity vector.
    """

    def __init__(self, n_items: int):
        """
        Parameters
        ----------
        n_items : int
            Total number of items in the task (e.g., 15 for Schapiro 2017).
            Schapiro (2017) Fig. 1: 15-item community graph.
        """
        super().__init__()
        self.n_items = n_items

    def forward(self, item_idx: torch.Tensor) -> torch.Tensor:
        """Return one-hot activity for item_idx.

        Parameters
        ----------
        item_idx : LongTensor, shape (,) or (batch,)
            Item index (0-based).

        Returns
        -------
        activity : FloatTensor, shape (n_items,) or (batch, n_items)
        """
        raise NotImplementedError("Step 2: implement L_ECin.forward()")


# =========================================================================
# L_DG: DENTATE GYRUS
# =========================================================================
# [Role]:
#   Pattern separation. DG maps ECin's distributed input to a highly sparse
#   (~1% active) orthogonal representation, so that even similar ECin patterns
#   produce distinct DG codes. This prevents interference in CA3 attractor memory.
#
# [Inputs]:
#   - a_ECin : (n_items,) — ECin activity
#
# [Outputs]:
#   - activity : (n_DG,) — sparse (~1%) float tensor via kWTA
#
# [Learning]:
#   CHL: W_ECin_DG updated with TSP learning rate (lr_TSP = 0.4).
#   ΔW = lr_TSP * (y_plus - y_minus) * x
#   Schapiro (2017) §2.3; O'Reilly & Munakata (2000) Ch. 4
#
# [Connections]:
#   ECin → DG  (W_ECin_DG, feedforward, TSP)
#   DG   → CA3 (W_DG_CA3, feedforward, TSP)
#
# [Stateful]:
#   Euler integration: _activity updated each settling cycle (tau = 0.1).
#   reset() before each minus phase.
#
# [Parameters]:
#   n_DG        : int   (default: 100 — tune as needed; Schapiro uses larger)
#   k_frac      : float (default: 0.01 → 1% sparsity; Schapiro 2017 §2.2)
#   tau         : float (default: 0.1; Leabra default)
#   use_euler   : bool  (default: True; False = stateless for unit tests)
#
# [Notes]:
#   - Schapiro (2017) §2.2: "DG: very sparse (~1%) via kWTA inhibition"
#   - No recurrent connections in DG (unlike CA3)
#   - Pattern separation is an emergent property of high sparsity + kWTA

class L_DG(nn.Module):
    """Dentate gyrus: sparse pattern-separated representation (~1% active).

    Stateful (Euler integration). Call reset() before each trial's minus phase.
    """

    def __init__(
        self,
        n_input: int,
        n_DG: int = 100,
        k_frac: float = 0.01,
        tau: float = 0.1,
        use_euler: bool = True,
    ):
        """
        Parameters
        ----------
        n_input : int
            ECin output dimension (= n_items).
        n_DG : int
            Number of DG units. Schapiro (2017): large relative to n_items.
        k_frac : float
            Fraction of active units after kWTA. Schapiro (2017) §2.2: ~0.01.
        tau : float
            Euler integration time constant. Leabra default: 0.1.
        use_euler : bool
            If False, stateless mode (for unit tests).
        """
        super().__init__()
        self.n_input = n_input
        self.n_DG = n_DG
        self.k_frac = k_frac
        self.tau = tau
        self.use_euler = use_euler

        # W_ECin_DG: feedforward weights, TSP pathway
        # Schapiro (2017) §2.3: initialized uniformly; learned via CHL
        self.W = nn.Parameter(torch.zeros(n_input, n_DG))

        # Euler activity buffer
        self.register_buffer('_activity', torch.zeros(n_DG))

    @property
    def activity(self) -> torch.Tensor:
        return self._activity

    def reset(self) -> None:
        """Reset activity to zero before each trial (before minus phase).

        # Schapiro (2017): DG starts inactive at trial onset.
        """
        self._activity.zero_()

    def forward(self, a_ECin: torch.Tensor) -> torch.Tensor:
        """One settling step: ECin → net input → nxx1 → kWTA → Euler update.

        Parameters
        ----------
        a_ECin : FloatTensor, shape (n_input,)

        Returns
        -------
        activity : FloatTensor, shape (n_DG,)
        """
        raise NotImplementedError("Step 3: implement L_DG.forward()")

    def update_weights(
        self, a_ECin_minus: torch.Tensor, a_ECin_plus: torch.Tensor,
        a_DG_minus: torch.Tensor, a_DG_plus: torch.Tensor,
        lr: float,
    ) -> None:
        """CHL weight update for W_ECin_DG.

        ΔW = lr * (y_plus * x_plus - y_minus * x_minus)
        O'Reilly & Munakata (2000) Ch. 4 Eq. 4.3
        """
        raise NotImplementedError("Step 3: implement L_DG.update_weights()")


# =========================================================================
# L_CA3: CA3 FIELD
# =========================================================================
# [Role]:
#   Pattern completion. CA3's recurrent connections form a Hopfield-like
#   attractor network. A partial cue activates a partial CA3 pattern, which
#   then reinstates the full stored pattern via recurrence.
#
# [Inputs]:
#   - a_DG   : (n_DG,)  — DG activity (feedforward from TSP)
#   - a_CA3  : (n_CA3,) — previous CA3 activity (recurrent)
#
# [Outputs]:
#   - activity : (n_CA3,) — ~10% active via kWTA
#
# [Learning]:
#   CHL: both W_DG_CA3 (feedforward) and W_CA3_CA3 (recurrent) updated.
#   TSP learning rate (lr_TSP = 0.4).
#   Schapiro (2017) §2.3; O'Reilly & Munakata (2000) Ch. 4
#
# [Connections]:
#   DG  → CA3  (W_DG_CA3, feedforward, TSP)
#   CA3 → CA3  (W_CA3_CA3, recurrent; enables pattern completion)
#   CA3 → CA1  (W_CA3_CA1, feedforward, TSP)
#
# [Stateful]:
#   Euler integration: _activity updated each settling cycle (tau = 0.1).
#   reset() before each trial's minus phase.
#
# [Parameters]:
#   n_CA3 : int   (default: 50)
#   k_frac : float (default: 0.10 → 10% sparsity; Schapiro 2017)
#   tau   : float (default: 0.1)
#
# [Notes]:
#   - Recurrent connections are the mechanism of pattern completion
#   - Schapiro (2017) §2.1: "CA3 contains recurrent (auto-associative) connections"
#   - On first encounter with a new item: CA3 output is near-random (no stored pattern)
#   - After learning: partial DG cue → CA3 completes the pattern

class L_CA3(nn.Module):
    """CA3 field: pattern completion via recurrent attractor dynamics (~10% active).

    Stateful (Euler integration). Call reset() before each trial's minus phase.
    """

    def __init__(
        self,
        n_DG: int,
        n_CA3: int = 50,
        k_frac: float = 0.10,
        tau: float = 0.1,
        use_euler: bool = True,
    ):
        """
        Parameters
        ----------
        n_DG : int
            DG output dimension.
        n_CA3 : int
            Number of CA3 units. Schapiro (2017): larger than n_items.
        k_frac : float
            kWTA sparsity. Schapiro (2017): ~0.10.
        tau : float
            Euler time constant. Leabra default: 0.1.
        use_euler : bool
            If False, stateless mode (for unit tests).
        """
        super().__init__()
        self.n_DG = n_DG
        self.n_CA3 = n_CA3
        self.k_frac = k_frac
        self.tau = tau
        self.use_euler = use_euler

        # W_DG_CA3: feedforward weights, TSP pathway
        self.W_ff = nn.Parameter(torch.zeros(n_DG, n_CA3))

        # W_CA3_CA3: recurrent weights (enable pattern completion)
        # Schapiro (2017) §2.1: "CA3 contains recurrent connections"
        self.W_rec = nn.Parameter(torch.zeros(n_CA3, n_CA3))

        self.register_buffer('_activity', torch.zeros(n_CA3))

    @property
    def activity(self) -> torch.Tensor:
        return self._activity

    def reset(self) -> None:
        """Reset activity to zero before each trial (before minus phase)."""
        self._activity.zero_()

    def forward(self, a_DG: torch.Tensor) -> torch.Tensor:
        """One settling step: DG + CA3_prev → net input → nxx1 → kWTA → Euler update.

        net = W_ff @ a_DG + W_rec @ a_CA3_prev
        a_CA3 = Euler(nxx1(net))
        """
        raise NotImplementedError("Step 4: implement L_CA3.forward()")

    def update_weights(
        self,
        a_DG_minus: torch.Tensor, a_DG_plus: torch.Tensor,
        a_CA3_minus: torch.Tensor, a_CA3_plus: torch.Tensor,
        lr: float,
    ) -> None:
        """CHL weight update for W_DG_CA3 and W_CA3_CA3.

        ΔW_ff  = lr * (y_CA3_plus * a_DG_plus  - y_CA3_minus * a_DG_minus)
        ΔW_rec = lr * (y_CA3_plus * y_CA3_plus - y_CA3_minus * y_CA3_minus)
        O'Reilly & Munakata (2000) Ch. 4 Eq. 4.3
        """
        raise NotImplementedError("Step 4: implement L_CA3.update_weights()")


# =========================================================================
# L_CA1: CA1 FIELD
# =========================================================================
# [Role]:
#   Convergence point of MSP and TSP. CA1 receives two streams:
#     (1) MSP: direct from ECin (W_ECin_CA1; slow, learns statistics)
#     (2) TSP: from CA3   (W_CA3_CA1; fast, learns episodes)
#   Plus-phase: ECout back-projection (W_ECout_CA1) provides teaching signal.
#
# [Inputs]:
#   - a_ECin  : (n_items,) — ECin activity (MSP)
#   - a_CA3   : (n_CA3,)  — CA3 activity (TSP)
#   - a_ECout : (n_items,) — ECout activity (plus phase only; None in minus phase)
#
# [Outputs]:
#   - activity : (n_CA1,) — ~10% active via kWTA
#
# [Learning]:
#   W_ECin_CA1: MSP learning rate (lr_MSP = 0.05). Slow → learns statistics.
#   W_CA3_CA1:  TSP learning rate (lr_TSP = 0.4). Fast → learns episodes.
#   W_ECout_CA1: same rate as W_ECin_CA1 (or lr_TSP; to be determined).
#   Schapiro (2017) §2.3, Go reimplementation params
#
# [Connections]:
#   ECin  → CA1  (W_ECin_CA1, MSP, slow)
#   CA3   → CA1  (W_CA3_CA1, TSP, fast)
#   ECout → CA1  (W_ECout_CA1, back-projection, plus-phase only)
#   CA1   → ECout (W_CA1_ECout, feedforward)
#
# [Stateful]:
#   Euler integration: _activity updated each settling cycle (tau = 0.1).
#   reset() before each trial's minus phase.
#
# [Notes]:
#   - MSP slow learning rate is what allows CA1 to develop community structure
#   - TSP fast learning rate enables single-trial episodic binding
#   - Schapiro (2017) §2.3: "MSP learning rate α_MSP = 0.05; TSP α_TSP = 0.4"
#     (Go reimplementation values; original emergent: 0.02 / 0.2)

class L_CA1(nn.Module):
    """CA1 field: MSP + TSP convergence (~10% active).

    Stateful (Euler integration). Call reset() before each trial's minus phase.
    """

    def __init__(
        self,
        n_items: int,
        n_CA3: int,
        n_CA1: int = 50,
        k_frac: float = 0.10,
        tau: float = 0.1,
        use_euler: bool = True,
    ):
        """
        Parameters
        ----------
        n_items : int
            ECin / ECout dimension.
        n_CA3 : int
            CA3 output dimension.
        n_CA1 : int
            Number of CA1 units.
        k_frac : float
            kWTA sparsity. Schapiro (2017): ~0.10.
        tau : float
            Euler time constant. Leabra default: 0.1.
        use_euler : bool
            If False, stateless mode (for unit tests).
        """
        super().__init__()
        self.n_items = n_items
        self.n_CA3 = n_CA3
        self.n_CA1 = n_CA1
        self.k_frac = k_frac
        self.tau = tau
        self.use_euler = use_euler

        # MSP: ECin → CA1 (slow learning rate; learns statistical regularities)
        # Schapiro (2017) §2.3: α_MSP = 0.05 (Go reimplementation)
        self.W_ECin = nn.Parameter(torch.zeros(n_items, n_CA1))

        # TSP: CA3 → CA1 (fast learning rate; learns individual episodes)
        # Schapiro (2017) §2.3: α_TSP = 0.4 (Go reimplementation)
        self.W_CA3 = nn.Parameter(torch.zeros(n_CA3, n_CA1))

        # Back-projection from ECout (plus-phase teaching signal)
        self.W_ECout = nn.Parameter(torch.zeros(n_items, n_CA1))

        self.register_buffer('_activity', torch.zeros(n_CA1))

    @property
    def activity(self) -> torch.Tensor:
        return self._activity

    def reset(self) -> None:
        """Reset activity to zero before each trial (before minus phase)."""
        self._activity.zero_()

    def forward(
        self,
        a_ECin: torch.Tensor,
        a_CA3: torch.Tensor,
        a_ECout: torch.Tensor = None,
    ) -> torch.Tensor:
        """One settling step.

        Minus phase: net = W_ECin @ a_ECin + W_CA3 @ a_CA3
        Plus phase:  net = W_ECin @ a_ECin + W_CA3 @ a_CA3 + W_ECout @ a_ECout

        Then: a_CA1 = Euler(kWTA(nxx1(net)))
        """
        raise NotImplementedError("Step 5: implement L_CA1.forward()")

    def update_weights(
        self,
        a_ECin_minus: torch.Tensor, a_ECin_plus: torch.Tensor,
        a_CA3_minus: torch.Tensor, a_CA3_plus: torch.Tensor,
        a_ECout_plus: torch.Tensor,
        a_CA1_minus: torch.Tensor, a_CA1_plus: torch.Tensor,
        lr_MSP: float, lr_TSP: float,
    ) -> None:
        """CHL weight updates for W_ECin (MSP), W_CA3 (TSP), W_ECout.

        ΔW_ECin  = lr_MSP * (y_CA1_plus * a_ECin_plus  - y_CA1_minus * a_ECin_minus)
        ΔW_CA3   = lr_TSP * (y_CA1_plus * a_CA3_plus   - y_CA1_minus * a_CA3_minus)
        ΔW_ECout = lr_MSP * (y_CA1_plus * a_ECout_plus - y_CA1_minus * 0)
        O'Reilly & Munakata (2000) Ch. 4 Eq. 4.3; Schapiro (2017) §2.3
        """
        raise NotImplementedError("Step 5: implement L_CA1.update_weights()")


# =========================================================================
# L_ECout: ENTORHINAL CORTEX OUTPUT
# =========================================================================
# [Role]:
#   Reconstruction layer. Receives from CA1 and attempts to reconstruct
#   the current item's ECin pattern. In the plus phase, ECout is clamped
#   to the target item — this clamped activity propagates back to CA1 via
#   W_ECout_CA1 (back-projection in L_CA1), providing the teaching signal.
#
# [Inputs]:
#   - a_CA1  : (n_CA1,) — CA1 activity (feedforward)
#
# [Outputs] (minus phase):
#   - activity : (n_items,) — reconstruction of current item
#
# [Plus phase]:
#   ECout is clamped to target item pattern (one-hot from L_ECin).
#   The clamped activity is passed to L_CA1.forward() as a_ECout.
#
# [Learning]:
#   CHL: W_CA1_ECout updated.
#   Same learning rate as MSP (lr_MSP = 0.05) or TSP depending on implementation.
#   Schapiro (2017) §2.3
#
# [Connections]:
#   CA1   → ECout (W_CA1_ECout, feedforward)
#   ECout → CA1   (back-projection handled in L_CA1.W_ECout)
#
# [Stateful]:
#   Euler integration: _activity updated each settling cycle (tau = 0.1).
#   In plus phase: _activity is overwritten by clamped target pattern.
#   reset() before each trial's minus phase.

class L_ECout(nn.Module):
    """Entorhinal cortex output: CA1 → reconstruction of input item.

    Plus phase: clamped to target item (ECin pattern).
    Stateful (Euler integration). Call reset() before each trial's minus phase.
    """

    def __init__(
        self,
        n_CA1: int,
        n_items: int,
        k_frac: float = 0.10,
        tau: float = 0.1,
        use_euler: bool = True,
    ):
        """
        Parameters
        ----------
        n_CA1 : int
            CA1 output dimension.
        n_items : int
            ECout output dimension (= n_items; matches ECin).
        k_frac : float
            kWTA sparsity. Schapiro (2017): ~0.10.
        tau : float
            Euler time constant. Leabra default: 0.1.
        use_euler : bool
            If False, stateless mode (for unit tests).
        """
        super().__init__()
        self.n_CA1 = n_CA1
        self.n_items = n_items
        self.k_frac = k_frac
        self.tau = tau
        self.use_euler = use_euler

        # W_CA1_ECout: CA1 → ECout feedforward
        self.W = nn.Parameter(torch.zeros(n_CA1, n_items))

        self.register_buffer('_activity', torch.zeros(n_items))

    @property
    def activity(self) -> torch.Tensor:
        return self._activity

    def reset(self) -> None:
        """Reset activity to zero before each trial (before minus phase)."""
        self._activity.zero_()

    def clamp(self, target_pattern: torch.Tensor) -> None:
        """Clamp ECout to target pattern for the plus phase.

        Overwrites _activity with the target item's ECin pattern.
        This clamped activity is passed back to CA1 as the teaching signal.
        Schapiro (2017) §2.3: ECout clamped to target in plus phase.
        """
        self._activity = target_pattern.clone().float()

    def forward(self, a_CA1: torch.Tensor) -> torch.Tensor:
        """One settling step (minus phase only): CA1 → net → nxx1 → kWTA → Euler.

        In plus phase, use clamp() instead of forward().
        """
        raise NotImplementedError("Step 2: implement L_ECout.forward()")

    def update_weights(
        self,
        a_CA1_minus: torch.Tensor, a_CA1_plus: torch.Tensor,
        a_ECout_minus: torch.Tensor, a_ECout_plus: torch.Tensor,
        lr: float,
    ) -> None:
        """CHL weight update for W_CA1_ECout.

        ΔW = lr * (y_ECout_plus * a_CA1_plus - y_ECout_minus * a_CA1_minus)
        O'Reilly & Munakata (2000) Ch. 4 Eq. 4.3
        """
        raise NotImplementedError("Step 2: implement L_ECout.update_weights()")
