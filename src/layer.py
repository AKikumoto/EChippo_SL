"""
layer.py: neural layers for the EC-hippocampus statistical learning circuit.

| Layer   | Description               | Role                                          |
|---------|---------------------------|-----------------------------------------------|
| L_ECin  | Entorhinal cortex input   | Input driver; one-hot item encoding           |
| L_DG    | Dentate gyrus             | Pattern separation; sparse kWTA (~1%)         |
| L_CA3   | CA3 field                 | Pattern completion; recurrent attractor       |
| L_CA1   | CA1 field                 | MSP + TSP convergence; output readout         |
| L_ECout | Entorhinal cortex output  | Reconstruction target; plus-phase teacher     |

Circuit from Schapiro et al. (2017) Phil. Trans. R. Soc. B, 372, 20160049.
See config/ARCHITECTURE_ENG.md for full equations, parameters, and CHL settling policy.

Two complementary pathways:
  MSP (monosynaptic):   ECin → CA1           slow lr=0.05; learns statistical regularities
  TSP (trisynaptic):    ECin → DG → CA3 → CA1  fast lr=0.4;  learns individual episodes

Schapiro (2017) §2: "the monosynaptic pathway—the pathway connecting entorhinal cortex
directly to region CA1—was able to support statistical learning, while the trisynaptic
pathway—connecting entorhinal cortex to CA1 through dentate gyrus and CA3—learned
individual episodes, with apparent representations of regularities resulting from
associative reactivation through recurrence."

Three-phase trial structure — 100 cycles (Schapiro 2017 §2.b–c):
  Q1       (cycles  1–25): ECin → CA1 strong; CA3 → CA1 inhibited. Theta trough (encoding).
  Q2–Q3    (cycles 26–75): CA3 → CA1 strong; ECin → CA1 reduced.  Theta peak (retrieval).
  Q4/plus  (cycles 76–100): ECout clamped to target. Weight update uses ActM (Q3) and ActP.

Naming conventions:
  L_* : layer modules in layer.py
  M_* : full model classes in model.py
  F_* : utility functions in util.py
  T_* : task environments in tasks.py
"""

import torch
import torch.nn as nn

# F_nxx1, F_kWTA defined in src/util.py (Step 1)
from util import F_nxx1, F_kWTA


# =========================================================================
# L_ECin: ENTORHINAL CORTEX INPUT
# =========================================================================
# [Role]:
#   Input driver. Encodes the current (and previous) item as a localist
#   pattern of activity. ECin is always clamped to the stimulus — it does
#   not settle freely and has no learnable weights.
#
#   Schapiro (2017) §2.a.ii: "each item in the paradigm was represented by
#   activation of one unit (with the number of units in ECin and ECout
#   varying across paradigms)."
#
# [Moving window — Schapiro (2017) §2.c]:
#   ECin encodes two items simultaneously:
#     current item  : activity = 1.0 (clamped)
#     previous item : activity = 0.9 (decayed)
#   All other units: 0.
#   This temporal asymmetry introduces a forward learning bias
#   (A → B is strengthened more than B → A).
#
#   §2.c: "presenting items to ECin using a moving window that encompassed
#   the current and previous items. The current item was presented with full
#   activity (clamped value = 1) and the previous stimulus were forced to
#   maintain high activity (0.9) while all other units were forced to have
#   no activity."
#
# [Separate Input layer — Schapiro (2017) §2.a.ii]:
#   A hidden Input layer (not shown in Fig. 1) sits upstream of ECin.
#   It has the same number of units as ECin and one-to-one connections.
#   Clamping is applied here so that ECin can also receive ECout
#   back-projections (the "big loop") without the clamp disrupting ECin.
#   §2.a.ii: "Input was clamped in this layer so as to allow ECin to also
#   receive input from ECout, completing the 'big loop' of the model."
#
# [Inhibition — Schapiro (2017) §2.a.ii]:
#   k = 2 (absolute count, not fraction).
#   §2.a.ii: "ECin and ECout each had inhibition set so that two units
#   could be active at a time (k = 2), unless otherwise noted."
#   For the community structure task (n_items=15): 2/15 ≈ 13% active.
#   [Note: k=2 matches moving window — exactly current + previous item.]
#
# [Inputs]:
#   - item_idx : int or LongTensor — index of the current item (0..n_items-1)
#
# [Outputs]:
#   - activity : (n_items,) one-hot float tensor (or moving-window two-hot)
#
# [Learning]:
#   None. ECin is a fixed input driver.
#
# [Connections]:
#   ECin → CA1  (MSP: monosynaptic pathway; W_ECin in L_CA1; lr_MSP = 0.05)
#   ECin → DG   (TSP first leg; W in L_DG; lr_TSP = 0.4)
#
# [Notes]:
#   - ECin is stateless: no _activity buffer, no Euler integration needed.
#   - In all phases (Q1, Q2-Q3, Q4): same ECin clamping.
#     The stimulus does not change across minus/plus phases within a trial.
#   - Community structure task: n_items = 15 (Schapiro 2017 Fig. 1; §3.b).
#   - For associative inference (Schapiro 2017 §3.c): n_items = 9.
#     §3.c: "we lowered inhibition to k=3 in ECin and ECout when testing."

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
            Schapiro (2017) Fig. 1: 15-item community graph (5 communities × 3).
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
#   (~1% active) orthogonal representation, so that even similar ECin
#   patterns produce distinct DG codes. This prevents interference in CA3.
#
#   Schapiro (2017) §2: "Connection sparsity and high inhibition result in
#   few units being active at any time in DG and CA3 (figure 1), and allow
#   the layers to avoid interference by forming separated, conjunctive
#   representations of incoming patterns, even when the patterns are highly
#   similar."
#
# [Inputs]:
#   - a_ECin : (n_items,) — ECin activity
#
# [Outputs]:
#   - activity : (n_DG,) — sparse (~1%) float tensor via kWTA
#
# [Sparse connectivity from ECin — Schapiro (2017) §2.a.iii]:
#   Each DG unit receives input from 25% of ECin units.
#   Implemented as a masked (sparse) weight matrix.
#   §2.a.iii: "ECin projects to DG and CA3 in the TSP. These projections
#   are sparse, reflecting known physiology. Each DG and CA3 unit receives
#   input from 25% of the ECin layer."
#   The sparse projections are randomized across 500 network initializations.
#   §2.a.v: "For each simulation, we ran 500 networks."
#
# [High inhibition — Schapiro (2017) §2.a.iii]:
#   DG uses kWTA with k_frac ≈ 0.01 (~1% active).
#   §2.a.iii: "DG and CA3 additionally have high levels of within-layer
#   inhibition."
#   §2.2: "High connection sparsity and high inhibition result in few units
#   being active at any time in DG and CA3."
#
# [Learning — CHL, TSP learning rate]:
#   W_ECin_DG updated via CHL with lr_TSP = 0.4.
#   ΔW = lr * (y_plus ⊗ x_plus − y_minus ⊗ x_minus)
#   Schapiro (2017) §2.b; O'Reilly & Munakata (2000) Ch. 4.
#   TSP learning rate is 10× MSP in the original model.
#   §2.b: "The learning rate in the TSP is set to be 10× higher than in
#   the MSP. See Ketz et al. [13] and the electronic supplementary
#   material, table S2 for more details."
#
# [Connections]:
#   ECin → DG  (W_ECin_DG; feedforward; sparse 25%; TSP)
#   DG   → CA3 (W_DG_CA3 in L_CA3; feedforward; 5% mossy fibre; TSP)
#
# [Stateful — Euler integration]:
#   _activity updated each settling cycle: tau = 0.1 (Leabra default).
#   reset() before each trial's minus phase (Q1 onset).
#   §2.b: "one trial (lasting 100 processing cycles) consisted of the
#   presentation of two such items for two minus phases and one plus phase."
#
# [Parameters]:
#   n_DG        : int   (default: 100 — scale to task; Schapiro uses larger)
#   k_frac      : float (default: 0.01 → 1% sparsity; Schapiro 2017 §2.2)
#   ecin_frac   : float (default: 0.25 → each DG unit gets 25% of ECin)
#   tau         : float (default: 0.1; Leabra default; O'Reilly & Munakata 2000)
#   use_euler   : bool  (default: True; False = stateless for unit tests)
#
# [Notes]:
#   - No recurrent connections in DG (unlike CA3).
#   - Pattern separation is emergent from high sparsity + kWTA + sparse ECin input.
#   - Why 1% active? → forces each episode to a near-unique DG code
#     → CA3 can form distinct, non-overlapping attractors per episode.
#   - "DG: pattern separation" vs "CA3: pattern completion" is the
#     foundational TSP dissociation (Schapiro 2017 §2; O'Reilly 2000 Ch.4).

class L_DG(nn.Module):
    """Dentate gyrus: sparse pattern-separated representation (~1% active).

    Stateful (Euler integration). Call reset() before each trial's minus phase.
    """

    def __init__(
        self,
        n_input: int,
        n_DG: int = 100,
        k_frac: float = 0.01,
        ecin_frac: float = 0.25,
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
        ecin_frac : float
            Fraction of ECin units each DG unit receives from.
            Schapiro (2017) §2.a.iii: 0.25 (25%).
        tau : float
            Euler integration time constant. Leabra default: 0.1.
        use_euler : bool
            If False, stateless mode (for unit tests).
        """
        super().__init__()
        self.n_input = n_input
        self.n_DG = n_DG
        self.k_frac = k_frac
        self.ecin_frac = ecin_frac
        self.tau = tau
        self.use_euler = use_euler

        # W_ECin_DG: sparse feedforward weights, TSP pathway
        # Schapiro (2017) §2.a.iii: 25% connectivity — each DG unit connects
        # to a random 25% of ECin units (re-randomized across 500 simulations).
        # requires_grad=False: updated manually via CHL, not autograd.
        self.W = nn.Parameter(torch.zeros(n_input, n_DG), requires_grad=False)

        # Sparse connectivity mask: 1 where connection exists, 0 otherwise.
        # Schapiro (2017) §2.a.v: sparse projections randomized per network.
        self.register_buffer(
            'mask',
            (torch.rand(n_input, n_DG) < ecin_frac).float()
        )

        # Euler activity buffer — zero at trial onset
        self.register_buffer('_activity', torch.zeros(n_DG))

    @property
    def activity(self) -> torch.Tensor:
        return self._activity

    def reset(self) -> None:
        """Reset activity to zero before each trial (before Q1 minus phase).

        Schapiro (2017) §2.b: layers re-initialize at trial onset.
        Plus phase (Q4) starts from Q2-Q3 final state — never reset mid-trial.
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

        ΔW = lr * (y_plus ⊗ x_plus − y_minus ⊗ x_minus)
        Masked: only update connected weights (mask == 1).
        O'Reilly & Munakata (2000) Ch. 4 Eq. 4.3; Schapiro (2017) §2.b.
        """
        raise NotImplementedError("Step 3: implement L_DG.update_weights()")


# =========================================================================
# L_CA3: CA3 FIELD
# =========================================================================
# [Role]:
#   Pattern completion. CA3's recurrent connections form a Hopfield-like
#   attractor. A partial cue from DG activates a partial CA3 pattern;
#   recurrent dynamics reinstate the full stored pattern.
#
#   Schapiro (2017) §2.a.iii: "CA3 also has a fully connected (every unit
#   to every other unit) projection to itself, which helps bind pieces of
#   a representation to one another and retrieve a full pattern from a
#   partial cue."
#
# [Inputs]:
#   - a_DG   : (n_DG,)  — DG activity (feedforward; mossy fibre)
#   - a_CA3  : (n_CA3,) — previous CA3 activity (recurrent self-connection)
#
# [Outputs]:
#   - activity : (n_CA3,) — ~10% active via kWTA
#
# [Sparse feedforward from DG — Schapiro (2017) §2.a.iii]:
#   DG → CA3: 5% connectivity (mossy fibre pathway).
#   §2.a.iii: "The 'mossy fibre' projection from DG to CA3 is even sparser
#   (5%)."
#   Sparse mossy fibre forces CA3 to store distinct attractors per episode
#   (each DG pattern activates a unique CA3 subpopulation).
#
# [Fully connected recurrent — Schapiro (2017) §2.a.iii]:
#   CA3 → CA3: every unit connects to every other unit.
#   This enables auto-associative (Hopfield) attractor dynamics:
#   partial cue → partial CA3 activation → recurrence fills in the rest.
#   §2.a.iii: "CA3 then has a fully connected projection to CA1,
#   completing the TSP."
#
# [Sparsity]:
#   k_frac ≈ 0.10 (~10% active). Less sparse than DG to allow the overlap
#   needed for pattern completion (Schapiro 2017; O'Reilly & Munakata 2000).
#
# [Learning — CHL, TSP learning rate]:
#   Both W_DG_CA3 (feedforward) and W_CA3_CA3 (recurrent) updated via CHL.
#   lr_TSP = 0.4 (Go reimplementation; original emergent: 0.2).
#   ΔW_ff  = lr * (y_CA3_plus ⊗ a_DG_plus  − y_CA3_minus ⊗ a_DG_minus)
#   ΔW_rec = lr * (y_CA3_plus ⊗ y_CA3_plus − y_CA3_minus ⊗ y_CA3_minus)
#   Schapiro (2017) §2.b; O'Reilly & Munakata (2000) Ch. 4 Eq. 4.3.
#
# [Connections]:
#   DG  → CA3  (W_ff;  feedforward; 5% mossy fibre; TSP)
#   CA3 → CA3  (W_rec; recurrent; fully connected; enables pattern completion)
#   CA3 → CA1  (W_CA3 in L_CA1; feedforward; fully connected; TSP)
#
# [Stateful — Euler integration]:
#   _activity carries the CA3 state across settling cycles.
#   Recurrent dynamics: net_CA3(t) = W_ff @ a_DG + W_rec @ a_CA3(t−1)
#                       a_CA3(t)   = (1−tau) * a_CA3(t−1) + tau * F_nxx1(net_CA3(t))
#   O'Reilly & Munakata (2000) Ch. 2.
#   reset() before each trial's Q1 minus phase.
#
# [Parameters]:
#   n_CA3    : int   (default: 50)
#   k_frac   : float (default: 0.10; Schapiro 2017)
#   dg_frac  : float (default: 0.05; mossy fibre; Schapiro 2017 §2.a.iii)
#   tau      : float (default: 0.1; Leabra default)
#   use_euler: bool  (default: True)
#
# [Notes]:
#   - On the first trial of a new item: no stored pattern → CA3 output near-random.
#     After learning: partial cue (from DG) → CA3 retrieves full stored pattern.
#   - Pattern completion (CA3) is complementary to pattern separation (DG):
#     DG ensures each episode has a unique code; CA3 stores and retrieves it.
#   - Schapiro (2017) Fig. 2a: CA3 shows pair-level similarity after training
#     in the episodic (non-statistical) task.
#   - Recurrent weights start near zero; they strengthen as CA3 learns
#     each (DG pattern, CA3 response) pair via CHL.

class L_CA3(nn.Module):
    """CA3 field: pattern completion via recurrent attractor dynamics (~10% active).

    Stateful (Euler integration). Call reset() before each trial's minus phase.
    """

    def __init__(
        self,
        n_DG: int,
        n_CA3: int = 50,
        k_frac: float = 0.10,
        dg_frac: float = 0.05,
        tau: float = 0.1,
        use_euler: bool = True,
    ):
        """
        Parameters
        ----------
        n_DG : int
            DG output dimension.
        n_CA3 : int
            Number of CA3 units.
        k_frac : float
            kWTA sparsity. Schapiro (2017): ~0.10.
        dg_frac : float
            Mossy fibre connectivity fraction.
            Schapiro (2017) §2.a.iii: 0.05 (5%).
        tau : float
            Euler time constant. Leabra default: 0.1.
        use_euler : bool
            If False, stateless mode (for unit tests).
        """
        super().__init__()
        self.n_DG = n_DG
        self.n_CA3 = n_CA3
        self.k_frac = k_frac
        self.dg_frac = dg_frac
        self.tau = tau
        self.use_euler = use_euler

        # W_DG_CA3: feedforward mossy fibre weights, TSP pathway
        # Schapiro (2017) §2.a.iii: 5% sparse (mossy fibre).
        # requires_grad=False: updated manually via CHL.
        self.W_ff = nn.Parameter(torch.zeros(n_DG, n_CA3), requires_grad=False)

        # Sparse mossy fibre mask: 5% of DG → CA3 connections exist
        self.register_buffer(
            'mask_ff',
            (torch.rand(n_DG, n_CA3) < dg_frac).float()
        )

        # W_CA3_CA3: recurrent weights (enable pattern completion)
        # Schapiro (2017) §2.a.iii: "fully connected (every unit to every other unit)"
        # requires_grad=False: updated manually via CHL.
        self.W_rec = nn.Parameter(torch.zeros(n_CA3, n_CA3), requires_grad=False)

        self.register_buffer('_activity', torch.zeros(n_CA3))

    @property
    def activity(self) -> torch.Tensor:
        return self._activity

    def reset(self) -> None:
        """Reset activity to zero before each trial (before Q1 minus phase)."""
        self._activity.zero_()

    def forward(self, a_DG: torch.Tensor) -> torch.Tensor:
        """One settling step: DG + CA3_prev → net input → nxx1 → kWTA → Euler update.

        net = W_ff * mask_ff @ a_DG + W_rec @ a_CA3_prev
        a_CA3(t) = (1 − tau) * a_CA3(t−1) + tau * kWTA(nxx1(net))
        O'Reilly & Munakata (2000) Ch. 2.
        """
        raise NotImplementedError("Step 4: implement L_CA3.forward()")

    def update_weights(
        self,
        a_DG_minus: torch.Tensor, a_DG_plus: torch.Tensor,
        a_CA3_minus: torch.Tensor, a_CA3_plus: torch.Tensor,
        lr: float,
    ) -> None:
        """CHL weight update for W_DG_CA3 and W_CA3_CA3.

        ΔW_ff  = lr * (y_CA3_plus ⊗ a_DG_plus  − y_CA3_minus ⊗ a_DG_minus)
        ΔW_rec = lr * (y_CA3_plus ⊗ y_CA3_plus − y_CA3_minus ⊗ y_CA3_minus)
        Masked: only update existing mossy fibre connections (mask_ff == 1).
        O'Reilly & Munakata (2000) Ch. 4 Eq. 4.3; Schapiro (2017) §2.b.
        """
        raise NotImplementedError("Step 4: implement L_CA3.update_weights()")


# =========================================================================
# L_CA1: CA1 FIELD
# =========================================================================
# [Role]:
#   Convergence point of MSP and TSP. Receives two input streams:
#     (1) MSP (W_ECin): direct from ECin. Slow lr=0.05. Learns statistics.
#     (2) TSP (W_CA3):  from CA3.         Fast lr=0.4.  Learns episodes.
#   Plus-phase back-projection from ECout corrects CA1 toward target.
#
#   Schapiro (2017) §2.a.iv: "There are fully connected projections in the
#   MSP from ECin to CA1, CA1 to ECout, and ECout to CA1. CA1 has much less
#   local inhibition than DG and CA3 and the projections within the MSP are
#   also not as sparse."
#
# [Inputs]:
#   - a_ECin  : (n_items,) — ECin activity (MSP)
#   - a_CA3   : (n_CA3,)  — CA3 activity (TSP)
#   - a_ECout : (n_items,) — ECout activity (plus phase only; None in minus phase)
#
# [Outputs]:
#   - activity : (n_CA1,) — ~10% active via kWTA
#
# [Net input per phase]:
#   Q1   (ECin-dominant minus):  net = W_ECin @ a_ECin  [CA3→CA1 inhibited]
#   Q2-Q3 (CA3-dominant minus):  net = W_CA3  @ a_CA3   [ECin→CA1 reduced]
#   Q4   (plus phase):           net = W_ECin @ a_ECin + W_CA3 @ a_CA3 + W_ECout @ a_ECout
#   NOTE: Schapiro implements Q1/Q2-Q3 distinction by toggling connection strengths.
#   See ARCHITECTURE_ENG.md §6 for Go reimplementation details.
#
# [Theta oscillation basis for two minus phases — Schapiro (2017) §2.b]:
#   §2.b: "At the trough of the theta cycle, as measured at the hippocampal
#   fissure, EC has a stronger influence on CA1, whereas at the peak, CA3
#   has a stronger influence on CA1. The model instantiates these two phases
#   of theta as two minus phases on each trial."
#   Refs [27] Hasselmo et al. (2002) Neural Comput.; [28] Brankack et al. (1993).
#
# [MSP learning rate — slow]:
#   lr_MSP = 0.05 (Go reimplementation; original emergent: 0.02).
#   Slow learning rate → CA1 accumulates statistics over many trials.
#   → Community structure emerges gradually (Schapiro 2017 Fig. 3c).
#   §2.b: "Modification of the model's internal representations to better
#   align with the observed environment is a general property of
#   error-driven learning algorithms."
#
# [TSP learning rate — fast]:
#   lr_TSP = 0.4 (Go reimplementation; original emergent: 0.2).
#   TSP is 10× MSP in original emergent: §2.b: "The learning rate in the
#   TSP is set to be 10× higher than in the MSP."
#   Fast learning → each episode bound in one or a few exposures.
#
# [Learning — CHL]:
#   ΔW_ECin  = lr_MSP * (y_CA1_plus ⊗ a_ECin_plus  − y_CA1_minus ⊗ a_ECin_minus)
#   ΔW_CA3   = lr_TSP * (y_CA1_plus ⊗ a_CA3_plus   − y_CA1_minus ⊗ a_CA3_minus)
#   ΔW_ECout = lr_MSP * (y_CA1_plus ⊗ a_ECout_plus − 0)
#   (ECout only active in plus phase; minus-phase ECout term is zero.)
#   O'Reilly & Munakata (2000) Ch. 4 Eq. 4.3; Schapiro (2017) §2.b.
#
# [Connections]:
#   ECin  → CA1  (W_ECin;  MSP; fully connected; slow)
#   CA3   → CA1  (W_CA3;   TSP; fully connected; fast)
#   ECout → CA1  (W_ECout; back-projection; plus phase only)
#   CA1   → ECout (W_CA1_ECout in L_ECout; forward)
#
# [Stateful — Euler integration]:
#   Euler: a_CA1(t) = (1−tau) * a_CA1(t−1) + tau * kWTA(nxx1(net(t)))
#   reset() before each trial's Q1.
#
# [Parameters]:
#   n_items  : int   (= n_ECin = n_ECout)
#   n_CA3    : int   (CA3 output dimension)
#   n_CA1    : int   (default: 50)
#   k_frac   : float (default: 0.10; Schapiro 2017 — less inhibition than DG)
#   tau      : float (default: 0.1; Leabra default)
#   use_euler: bool  (default: True)
#
# [Notes]:
#   - "CA1 is the most cortex-like area of the hippocampus, with more
#     overlapping representations and a slower learning rate... precisely
#     the properties that encourage neural networks to efficiently generalize
#     across experiences." (Schapiro 2017 §4.e, Discussion p.12)
#   - MSP community learning (Fig. 3a): after training, within-community
#     CA1 pairs show higher representational similarity than across-community.
#   - TSP episodic binding (Fig. 2a): CA3/DG show pair-level structure.
#   - CA1 bridges the two: MSP makes it sensitive to statistics;
#     TSP makes it sensitive to individual episodes.

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
            kWTA sparsity. Schapiro (2017): ~0.10 (less than DG; §2.a.iv).
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

        # MSP: ECin → CA1 (slow; learns statistical regularities)
        # Schapiro (2017) §2.a.iv: "fully connected projections in the MSP"
        # lr_MSP = 0.05 (Go reimplementation; Schapiro 2017 §2.b)
        self.W_ECin = nn.Parameter(torch.zeros(n_items, n_CA1), requires_grad=False)

        # TSP: CA3 → CA1 (fast; learns individual episodes)
        # Schapiro (2017) §2.a.iv: "CA3 then has a fully connected projection to CA1"
        # lr_TSP = 0.4 (Go reimplementation; Schapiro 2017 §2.b)
        self.W_CA3 = nn.Parameter(torch.zeros(n_CA3, n_CA1), requires_grad=False)

        # Back-projection from ECout (plus-phase teaching signal)
        # Schapiro (2017) §2.a.iv: ECout → CA1 "fully connected"
        # Completes the "big loop": ECin → CA1 → ECout → CA1
        self.W_ECout = nn.Parameter(torch.zeros(n_items, n_CA1), requires_grad=False)

        self.register_buffer('_activity', torch.zeros(n_CA1))

    @property
    def activity(self) -> torch.Tensor:
        return self._activity

    def reset(self) -> None:
        """Reset activity to zero before each trial (before Q1 minus phase)."""
        self._activity.zero_()

    def forward(
        self,
        a_ECin: torch.Tensor,
        a_CA3: torch.Tensor,
        a_ECout: torch.Tensor = None,
    ) -> torch.Tensor:
        """One settling step.

        Q1   (ECin-dominant minus): pass a_CA3=zeros (CA3→CA1 inhibited).
        Q2-Q3 (CA3-dominant minus): pass a_ECin=zeros (ECin→CA1 reduced).
        Q4   (plus phase):          pass all three inputs including a_ECout.

        net = W_ECin @ a_ECin + W_CA3 @ a_CA3 [+ W_ECout @ a_ECout if plus]
        a_CA1(t) = (1−tau) * a_CA1(t−1) + tau * kWTA(nxx1(net))
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

        ΔW_ECin  = lr_MSP * (y_CA1_plus ⊗ a_ECin_plus  − y_CA1_minus ⊗ a_ECin_minus)
        ΔW_CA3   = lr_TSP * (y_CA1_plus ⊗ a_CA3_plus   − y_CA1_minus ⊗ a_CA3_minus)
        ΔW_ECout = lr_MSP * (y_CA1_plus ⊗ a_ECout_plus − 0)
        lr_MSP = 0.05 (Go reimplementation); lr_TSP = 0.4.
        O'Reilly & Munakata (2000) Ch. 4 Eq. 4.3; Schapiro (2017) §2.b.
        """
        raise NotImplementedError("Step 5: implement L_CA1.update_weights()")


# =========================================================================
# L_ECout: ENTORHINAL CORTEX OUTPUT
# =========================================================================
# [Role]:
#   Reconstruction layer. Receives from CA1 and attempts to reproduce
#   the current item's ECin pattern (next-item prediction).
#   In the plus phase, ECout is clamped to the target — this propagates
#   back to CA1 via W_ECout (in L_CA1) as the CHL teaching signal.
#
#   Schapiro (2017) §2: "ECin serves as input and ECout as output for the
#   network. The network is trained to reproduce the pattern of activity in
#   ECin on ECout." (Fig. 1 caption)
#
# [Inputs]:
#   - a_CA1 : (n_CA1,) — CA1 activity (feedforward)
#
# [Outputs — minus phase]:
#   - activity : (n_items,) — CA1's predicted reconstruction of current item
#
# [Plus phase — clamp to target]:
#   ECout activity is overwritten by the target item's ECin pattern.
#   §2.b: "in the plus phase, the model is directly shown the correct output."
#   The clamped ECout activity propagates back via W_ECout in L_CA1,
#   providing the error-correction signal to CA1.
#   §2.b: "Weights are changed after each trial such that patterns of unit
#   coactivity during each minus phase are shifted more towards those of
#   the plus phase."
#
# [Inhibition — Schapiro (2017) §2.a.ii]:
#   k = 2 (absolute count). Matches ECin.
#   §2.a.ii: "ECin and ECout each had inhibition set so that two units could
#   be active at a time (k = 2), unless otherwise noted."
#   For n_items=15: 2/15 ≈ 13% active.
#   [Note: for associative inference (§3.c), k is raised to 3 during testing.]
#
# [Big-loop recurrence]:
#   ECout → CA1 (back-projection via W_ECout in L_CA1) completes the loop:
#   ECin → CA1 → ECout → CA1 → ECout → ...
#   This recurrence allows the network to generalize across training:
#   Schapiro (2017) §4.a: "big-loop recurrence" is key for community
#   structure learning and associative inference.
#
# [Learning — CHL]:
#   W_CA1_ECout updated via CHL, same rate as MSP.
#   ΔW = lr * (y_ECout_plus ⊗ a_CA1_plus − y_ECout_minus ⊗ a_CA1_minus)
#   O'Reilly & Munakata (2000) Ch. 4 Eq. 4.3; Schapiro (2017) §2.b.
#
# [Connections]:
#   CA1   → ECout (W_CA1_ECout; feedforward; fully connected)
#   ECout → CA1   (back-projection handled in L_CA1.W_ECout)
#
# [Stateful — Euler integration]:
#   In minus phases: Euler update from CA1 input.
#   In plus phase:   _activity overwritten by clamp().
#   reset() before each trial's Q1.
#
# [Parameters]:
#   n_CA1   : int   (CA1 output dimension)
#   n_items : int   (= n_ECin; reconstruction target)
#   k       : int   (default: 2; absolute count; Schapiro 2017 §2.a.ii)
#   tau     : float (default: 0.1; Leabra default)
#   use_euler: bool (default: True)
#
# [Notes]:
#   - k=2 is absolute, not fractional. Implementation: sort activity descending,
#     keep top 2, zero the rest.
#   - After training: ECout activation on test = probability of producing
#     a particular item as output (Schapiro 2017 §2.d.i: "probability of
#     activating a particular item in ECout above 0.5").
#   - Schapiro (2017) Fig. 2c/3c: output probability curves over training
#     are the key behavioral measure.

class L_ECout(nn.Module):
    """Entorhinal cortex output: CA1 → reconstruction of input item.

    Plus phase: clamped to target item (ECin pattern).
    Stateful (Euler integration). Call reset() before each trial's minus phase.
    """

    def __init__(
        self,
        n_CA1: int,
        n_items: int,
        k: int = 2,
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
        k : int
            Absolute number of active units after kWTA.
            Schapiro (2017) §2.a.ii: k=2. Default 2.
        tau : float
            Euler time constant. Leabra default: 0.1.
        use_euler : bool
            If False, stateless mode (for unit tests).
        """
        super().__init__()
        self.n_CA1 = n_CA1
        self.n_items = n_items
        self.k = k           # Schapiro (2017) §2.a.ii: k=2 (absolute)
        self.tau = tau
        self.use_euler = use_euler

        # W_CA1_ECout: CA1 → ECout feedforward; fully connected
        # Schapiro (2017) §2.a.iv: "fully connected projections in the MSP
        # from ECin to CA1, CA1 to ECout, and ECout to CA1."
        # requires_grad=False: updated manually via CHL.
        self.W = nn.Parameter(torch.zeros(n_CA1, n_items), requires_grad=False)

        self.register_buffer('_activity', torch.zeros(n_items))

    @property
    def activity(self) -> torch.Tensor:
        return self._activity

    def reset(self) -> None:
        """Reset activity to zero before each trial (before Q1 minus phase)."""
        self._activity.zero_()

    def clamp(self, target_pattern: torch.Tensor) -> None:
        """Clamp ECout to target pattern for the plus phase (Q4).

        Overwrites _activity with the target item's ECin pattern.
        This clamped activity is passed back to CA1 as the teaching signal.
        Schapiro (2017) §2.b: "in the plus phase, the model is directly
        shown the correct output."
        """
        self._activity = target_pattern.clone().float()

    def forward(self, a_CA1: torch.Tensor) -> torch.Tensor:
        """One settling step (minus phases Q1/Q2-Q3 only).

        net     = W @ a_CA1
        a_ECout = kWTA_k2(nxx1(net))    [k=2 absolute; Schapiro 2017 §2.a.ii]
        Euler:  _activity = (1−tau) * _activity + tau * a_ECout

        In plus phase, use clamp() instead — do not call forward().
        """
        raise NotImplementedError("Step 2: implement L_ECout.forward()")

    def update_weights(
        self,
        a_CA1_minus: torch.Tensor, a_CA1_plus: torch.Tensor,
        a_ECout_minus: torch.Tensor, a_ECout_plus: torch.Tensor,
        lr: float,
    ) -> None:
        """CHL weight update for W_CA1_ECout.

        ΔW = lr * (y_ECout_plus ⊗ a_CA1_plus − y_ECout_minus ⊗ a_CA1_minus)
        O'Reilly & Munakata (2000) Ch. 4 Eq. 4.3; Schapiro (2017) §2.b.
        """
        raise NotImplementedError("Step 2: implement L_ECout.update_weights()")
