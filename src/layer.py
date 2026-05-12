"""
layer.py: neural layers for the EC-hippocampus statistical learning circuit.

Connection diagram (Schapiro et al. 2017 §2.a):

  Input
    │
  L_ECin ──────────────────────────────────────────── W_ECin ──► L_CA1 ──► L_ECout
    │   (MSP: direct, lr=0.05; learns statistics)               ▲   ▲        │
    │                                                            │   │        │
    │   (TSP: lr=0.4; learns episodes)                          │   │ W_ECout│
    ├──25%──► L_DG ──5% mossy──► L_CA3 ──── W_CA3 ────────────►│   │◄───────┘
    │          (~1%)               (~10%)                        │   │  back-proj (Q4 only)
    └──25%──────────────────────────────────────────────────────►│   │
               (ECin→CA3 direct; Schapiro 2017 §2.a.iii)        │   │
                                    ↺ W_rec (fully connected)    │   │
                                                                 │   │
  W shapes:                                                      │   │
    W_ECin_DG  (n_ECin, n_DG)     mask: 25% sparse (ecin_frac)  │   │
    W_DG_CA3   (n_DG,  n_CA3)     mask:  5% sparse (dg_frac)    │   │
    W_CA3_CA3  (n_CA3, n_CA3)     full (recurrent attractor)     │   │
    W_ECin_CA1 (n_ECin, n_CA1)    full (MSP)                     │   │
    W_CA3_CA1  (n_CA3, n_CA1)     full (TSP)                     │   │
    W_ECout_CA1(n_ECin, n_CA1)    full (back-projection, Q4)     │   │
    W_CA1_ECout(n_CA1, n_ECin)    full (output)                  │   │

| Layer       | Description               | Role                                      |
|-------------|---------------------------|-------------------------------------------|
| L_ECin      | Entorhinal cortex input   | Float activity clamp; big-loop ready      |
| L_DG        | Dentate gyrus             | Pattern separation; ~1% kWTA              |
| L_CA3       | CA3 field                 | Pattern completion; recurrent W_rec       |
| L_CA1       | CA1 field                 | MSP + TSP convergence                     |
| L_ECout     | Entorhinal cortex output  | Reconstruction; plus-phase teacher        |
| L_PFC       | PFC recurrent (RNN/GRU/LSTM) | ECout → PFC hidden → net_pfc → ECin    |

Circuit from Schapiro et al. (2017) Phil. Trans. R. Soc. B, 372, 20160049.
See config/ARCHITECTURE_ENG.md for full equations, parameters, and CHL settling policy.

Two complementary pathways:
  MSP (monosynaptic):   ECin → CA1                       slow lr=0.05; statistical regularities
  TSP (trisynaptic):    ECin → DG → CA3 → CA1            fast lr=0.4;  episodic binding
                        ECin → CA3 also direct (25%)     Schapiro 2017 §2.a.iii

Schapiro (2017) §2: "the monosynaptic pathway—the pathway connecting entorhinal cortex
directly to region CA1—was able to support statistical learning, while the trisynaptic
pathway—connecting entorhinal cortex to CA1 through dentate gyrus and CA3—learned
individual episodes, with apparent representations of regularities resulting from
associative reactivation through recurrence."

Three-phase trial structure — 100 cycles (Schapiro 2017 §2.b–c):
  Q1       (cycles  1–25): ECin → CA1 strong; CA3 → CA1 inhibited. Theta trough (encoding).
  Q2–Q3    (cycles 26–75): CA3 → CA1 strong; ECin → CA1 reduced.  Theta peak (retrieval).
  Q4/plus  (cycles 76–100): ECout clamped to target. Weight update uses ActM (Q3) and ActP.

CHL — Contrastive Hebbian Learning (O'Reilly & Munakata 2000, Ch. 4; Schapiro 2017 §2.b):

  ΔW = lr × (ActP ⊗ ActP  −  ActM ⊗ ActM)

  ActM = activity at end of Q3 (cycle 75)   — minus phase (free prediction)
  ActP = activity at end of Q4 (cycle 100)  — plus phase  (ECout clamped to target)
  ⊗    = outer product of post × pre activity vectors

  Per-pathway learning rates (Go reimplementation; Schapiro 2017 §2.b):
    MSP  ECin → CA1                       lr = 0.05  slow; accumulates statistics
    TSP  ECin → DG, DG → CA3,            lr = 0.4   fast; binds individual episodes
         CA3 → CA3 (rec), CA3 → CA1
    out  CA1 → ECout                      lr = 0.05  matches MSP

  Why two rates? MSP needs many trials to learn community structure (slow smoothing).
  TSP needs one shot to bind an episode before the next trial overwrites it (fast Hebb).
  Schapiro (2017) §2.b: "The learning rate in the TSP is set to be 10× higher than in
  the MSP."

Naming conventions:
  L_* : layer modules in layer.py
  M_* : full model classes in model.py
  F_* : utility functions in util.py
  T_* : task environments in tasks.py
"""

import torch
import torch.nn as nn

# F_nxx1, F_kWTA, F_init_weights, NET_SCALE defined in src/util.py (Step 1)
from util import F_nxx1, F_kWTA, F_init_weights, NET_SCALE


# =========================================================================
# L_ECin: ENTORHINAL CORTEX INPUT
# =========================================================================
# [Role]:
#   Input driver. Settles to an externally supplied float activity pattern.
#   L_ECin does not construct the pattern — that is the model's job:
#
#   Schapiro replication (M_Hip):
#     clamp = one_hot(curr)*1.0 + one_hot(prev)*0.9  (moving window §2.c)
#   K&M task (M_Hip_KM):
#     clamp = T_TaskEmbedding(cond_idx)               (feature-coded)
#
#   Both cases call the same L_ECin.forward(clamp_pattern).
#
# [Why a separate Input layer is NOT needed — Schapiro (2017) §2.a.ii]:
#   The paper describes a hidden "Input layer" with one-to-one connections to
#   ECin. In emergent/Leabra, a layer can be either clamped OR driven by
#   learned weights, but not both simultaneously. The Input layer was the
#   workaround: Input is clamped to the stimulus; ECin then receives from
#   Input (weight=1, fixed) plus ECout (W_ECout, learned), so ECin itself
#   is never clamped and can settle freely.
#
#   In PyTorch there is no such constraint. clamp_pattern is simply a tensor
#   added directly to the net input computation:
#     net = clamp_pattern + a_ECout @ W_ECout
#   clamp_pattern acts as the Input layer's output (fixed stimulus signal).
#   a_ECout @ W_ECout provides the learned ECout back-projection.
#   No separate module is needed.
#
# [Big loop — Schapiro (2017) §2.a.ii]:
#   ECin receives ECout activity via W_ECout (n_ECout, n_units), learned via
#   CHL at MSP rate. Pass a_ECout to forward(); L_ECin projects it internally.
#   When a_ECout is supplied, use_euler should be True so ECin settles rather
#   than snapping. Omit n_ECout (or leave a_ECout=None) to disable the loop.
#
# [use_euler flag]:
#   False (default): _activity = kWTA(net) each cycle. ECin is effectively
#     clamped — equivalent to the Input layer driving ECin with weight=1.
#   True: Euler settling. Required when ECout back-projection modulates ECin.
#
# [Inhibition — Schapiro (2017) §2.a.ii]:
#   k = 2 absolute. For n_items=15: 2/15 ≈ 13% active.
#   §2.a.ii: "ECin and ECout each had inhibition set so that two units
#   could be active at a time (k = 2), unless otherwise noted."
#
# [Learning]:
#   W_ECout (ECout → ECin, big loop) updated via CHL at lr_MSP = 0.05.
#   All other ECin weights are fixed (clamp_pattern is not a learned weight).

class L_ECin(nn.Module):
    """Entorhinal cortex input: float activity pattern with optional Euler settling.

    Accepts any pre-built float vector (one-hot, feature-coded, or embedded).
    One-hot/moving-window construction belongs at the model level, not here.

    The separate 'Input layer' described in Schapiro (2017) §2.a.ii is not
    needed in PyTorch: clamp_pattern plays that role directly (see section
    comment above). W_ECout holds the learned ECout→ECin back-projection.
    """

    def __init__(
        self,
        n_units: int,
        n_ECout: int | None = None,
        k: int = 2,
        tau: float = 0.1,
        use_euler: bool = False,
    ):
        """
        Parameters
        ----------
        n_units : int
            Number of ECin units.
            Schapiro community task: 15 (= n_items; Schapiro 2017 Fig. 1).
            K&M feature-coded: 8 (4 rule units + 4 stim units).
        n_ECout : int or None
            Number of ECout units. If given, W_ECout (ECout→ECin back-
            projection) is created and the big loop is enabled.
            None (default): big loop disabled; a_ECout ignored in forward().
        k : int
            Absolute kWTA count. Schapiro (2017) §2.a.ii: k=2.
        tau : float
            Euler rate. Leabra default: 0.1 (O'Reilly & Munakata 2000).
            Only used when use_euler=True.
        use_euler : bool
            False: pure clamp — _activity snaps to kWTA(net) each cycle.
            True: Euler settling — needed when ECout back-projection is active.
        """
        super().__init__()
        self.n_units   = n_units
        self.n_ECout   = n_ECout
        self.k         = k
        self.tau       = tau
        self.use_euler = use_euler
        self._activity = torch.zeros(n_units)

        # ECout → ECin back-projection (big loop). None when n_ECout not given.
        # Schapiro (2017) §2.a.ii; learned at MSP rate (lr=0.05).
        # Init: uniform(0.49, 0.51) — SI Table 2 "big_loop" range; see util._W_INIT.
        if n_ECout is not None:
            self.W_ECout = F_init_weights((n_ECout, n_units), 'big_loop')
        else:
            self.W_ECout = None

    def reset(self):
        """Zero _activity. Call once per trial before Q1."""
        self._activity = torch.zeros(self.n_units, device=self._activity.device)

    def forward(
        self,
        clamp_pattern: torch.Tensor,
        a_ECout: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Settle ECin for one cycle.

        Parameters
        ----------
        clamp_pattern : FloatTensor (n_units,)
            External stimulus pattern (plays the role of the emergent Input
            layer output). Constructed by the caller:
              Schapiro → one_hot(curr)*1.0 + one_hot(prev)*0.9
              K&M      → T_TaskEmbedding(cond_idx)
        a_ECout : FloatTensor (n_ECout,), optional
            Raw ECout activity. L_ECin projects it via W_ECout internally.
            Only used when n_ECout was given at construction (big loop).

        Returns
        -------
        _activity : FloatTensor (n_units,)
        """
        if a_ECout is not None and self.W_ECout is not None:
            # SI Table 2 scale abs=2 for ECout→ECin; see util.NET_SCALE['ecout_ecin']
            net = clamp_pattern + NET_SCALE['ecout_ecin'] * (a_ECout @ self.W_ECout)
        else:
            net = clamp_pattern

        # nxx1 then absolute kWTA — Schapiro (2017) §2.a.ii; O'Reilly & Munakata (2000)
        # nxx1 bounds activity to [0,1]; without it the big-loop NET_SCALE would
        # pass amplified raw net values through kWTA, inflating CHL updates ~3×.
        activated = F_nxx1(net)
        k_from_bottom = self.n_units - self.k + 1
        threshold = torch.kthvalue(activated, k_from_bottom).values
        new_act = activated * (activated >= threshold).float()

        if self.use_euler:
            # Euler integration; O'Reilly & Munakata (2000) Ch. 2
            self._activity = (1.0 - self.tau) * self._activity + self.tau * new_act
        else:
            self._activity = new_act

        return self._activity

    def update_weights(
        self,
        a_ECout_minus: torch.Tensor,
        a_ECout_plus: torch.Tensor,
        a_ECin_minus: torch.Tensor,
        a_ECin_plus: torch.Tensor,
        lr: float = 0.05,
    ) -> None:
        """CHL update for W_ECout (ECout→ECin back-projection).

        ΔW = lr * (a_ECin_plus ⊗ a_ECout_plus − a_ECin_minus ⊗ a_ECout_minus)
        W shape (n_ECout, n_units): outer(a_ECout, a_ECin).
        O'Reilly & Munakata (2000) Ch. 4 Eq. 4.3; Schapiro (2017) §2.b.
        lr = 0.05 (MSP rate; Go reimplementation).

        No-op if big loop is disabled (W_ECout is None).
        """
        if self.W_ECout is None:
            return
        delta_plus  = torch.outer(a_ECout_plus,  a_ECin_plus)
        delta_minus = torch.outer(a_ECout_minus, a_ECin_minus)
        self.W_ECout.data += lr * (delta_plus - delta_minus)


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
        lr: float = 0.4,
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
        lr : float
            TSP learning rate. Go reimplementation: 0.4. (Schapiro 2017 §2.b)
        """
        super().__init__()
        self.n_input = n_input
        self.n_DG = n_DG
        self.k_frac = k_frac
        self.ecin_frac = ecin_frac
        self.tau = tau
        self.use_euler = use_euler
        self.lr = lr

        # Sparse connectivity mask: 1 where connection exists, 0 otherwise.
        # Schapiro (2017) §2.a.v: sparse projections randomized per network.
        mask = (torch.rand(n_input, n_DG) < ecin_frac).float()
        self.register_buffer('mask', mask)

        # W_ECin_DG: sparse feedforward weights, TSP pathway
        # Schapiro (2017) §2.a.iii: 25% connectivity — each DG unit connects
        # to a random 25% of ECin units (re-randomized across 500 simulations).
        # Init: uniform(0.25, 0.75) — SI Table 2 'default'; see util._W_INIT.
        # requires_grad=False: CHL updates via .data +=; use W.requires_grad_(True)
        # at the model level to switch to backprop mode when needed.
        self.W = F_init_weights((n_input, n_DG), 'default', mask=mask)

        # Separate Euler state (membrane potential) and firing rate (output)
        # In emergent/Leabra, Vm is integrated by Euler; y = F(Vm) is the output.
        # Feeding y (not Vm) into recurrent input is the correct Leabra formulation.
        self.register_buffer('_Vm', torch.zeros(n_DG))   # membrane potential (Euler state)
        self.register_buffer('_y',  torch.zeros(n_DG))   # firing rate (kWTA output)

    @property
    def activity(self) -> torch.Tensor:
        return self._y

    def reset(self) -> None:
        """Reset Vm and y to zero before each trial (before Q1 minus phase).

        Schapiro (2017) §2.b: layers re-initialize at trial onset.
        Plus phase (Q4) starts from Q2-Q3 final state — never reset mid-trial.
        """
        self._Vm.zero_()
        self._y.zero_()

    def forward(self, a_ECin: torch.Tensor) -> torch.Tensor:
        """One settling step: ECin → net input → Euler on Vm → nxx1 → kWTA → y.

        Parameters
        ----------
        a_ECin : FloatTensor, shape (n_input,)

        Returns
        -------
        y : FloatTensor, shape (n_DG,)
        """
        # Net input: masked sparse ECin → DG projection
        # W * mask zeros non-connected weights; Schapiro (2017) §2.a.iii: 25% sparse
        net = a_ECin @ (self.W * self.mask)   # (n_DG,)

        # Euler integration on membrane potential Vm; O'Reilly & Munakata (2000) Ch. 2
        # Vm(t) = (1−tau)*Vm(t−1) + tau*net(t); tau=0.1 Leabra default
        if self.use_euler:
            self._Vm = (1.0 - self.tau) * self._Vm + self.tau * net
        else:
            self._Vm = net

        # Firing rate: nxx1 then kWTA (~1% active)
        # Schapiro (2017) §2.2; O'Reilly & Munakata (2000) Ch. 2 Eq. 2.12
        self._y = F_kWTA(F_nxx1(self._Vm), k_frac=self.k_frac)
        return self._y

    def update_weights(
        self, a_ECin_minus: torch.Tensor, a_ECin_plus: torch.Tensor,
        a_DG_minus: torch.Tensor, a_DG_plus: torch.Tensor,
    ) -> None:
        """CHL weight update for W_ECin_DG.

        ΔW = lr * (y_plus ⊗ x_plus − y_minus ⊗ x_minus)
        Masked: only update connected weights (mask == 1).
        O'Reilly & Munakata (2000) Ch. 4 Eq. 4.3; Schapiro (2017) §2.b.
        """
        # W is (n_input, n_DG): outer(a_ECin, a_DG) has the same shape
        delta_plus  = torch.outer(a_ECin_plus,  a_DG_plus)
        delta_minus = torch.outer(a_ECin_minus, a_DG_minus)
        # Apply mask: non-connected weights stay at zero
        # Schapiro (2017) §2.a.v: connectivity pattern is fixed per network init
        self.W.data += self.lr * self.mask * (delta_plus - delta_minus)


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
#   - activity : (n_CA3,) — ~6% active via kWTA
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
#   k_frac ≈ 0.06 (~6% active). Less sparse than DG to allow the overlap
#   needed for pattern completion (Schapiro 2017 SI Table 1; O'Reilly & Munakata 2000).
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
# [Stateful — Euler integration, Vm/y separation]:
#   Two buffers: _Vm (membrane potential, Euler state) and _y (firing rate, output).
#   Recurrent input uses _y (firing rate), not _Vm (membrane potential).
#   net_CA3(t)  = W_ff @ a_DG + W_rec @ y_CA3(t−1)   ← y, not Vm
#   Vm_CA3(t)   = (1−tau) * Vm_CA3(t−1) + tau * net_CA3(t)
#   y_CA3(t)    = kWTA(F_nxx1(Vm_CA3(t)))
#   O'Reilly & Munakata (2000) Ch. 2.
#   reset() before each trial's Q1 minus phase.
#
# [Parameters]:
#   n_CA3    : int   (default: 50)
#   k_frac   : float (default: 0.06; Schapiro 2017 SI Table 1)
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
    """CA3 field: pattern completion via recurrent attractor dynamics (~6% active).

    Stateful (Euler integration). Call reset() before each trial's minus phase.
    """

    def __init__(
        self,
        n_DG: int,
        n_ECin: int,
        n_CA3: int = 50,
        k_frac: float = 0.06,
        dg_frac: float = 0.05,
        ecin_frac: float = 0.25,
        tau: float = 0.1,
        use_euler: bool = True,
        lr: float = 0.4,
    ):
        """
        Parameters
        ----------
        n_DG : int
            DG output dimension.
        n_ECin : int
            ECin dimension (perforant path direct input).
        n_CA3 : int
            Number of CA3 units.
        k_frac : float
            kWTA sparsity. Schapiro (2017) SI Table 1: 0.06.
        dg_frac : float
            Mossy fibre connectivity fraction.
            Schapiro (2017) §2.a.iii: 0.05 (5%).
        ecin_frac : float
            ECin → CA3 direct connectivity fraction.
            Schapiro (2017) §2.a.iii: 0.25 (25%).
        tau : float
            Euler time constant. Leabra default: 0.1.
        use_euler : bool
            If False, stateless mode (for unit tests).
        lr : float
            TSP learning rate. Go reimplementation: 0.4. (Schapiro 2017 §2.b)
        """
        super().__init__()
        self.n_DG = n_DG
        self.n_ECin = n_ECin
        self.n_CA3 = n_CA3
        self.k_frac = k_frac
        self.dg_frac = dg_frac
        self.ecin_frac = ecin_frac
        self.tau = tau
        self.use_euler = use_euler
        self.lr = lr

        # Sparse mossy fibre mask: 5% of DG → CA3 connections exist
        mask_ff = (torch.rand(n_DG, n_CA3) < dg_frac).float()
        self.register_buffer('mask_ff', mask_ff)

        # W_DG_CA3: feedforward mossy fibre weights, TSP pathway
        # Schapiro (2017) §2.a.iii: 5% sparse (mossy fibre).
        # Init: uniform(0.89, 0.91) — SI Table 2 'mossy_fiber'; detonator synapse.
        # High narrow range ensures DG drives a unique CA3 pattern even through 5% connectivity.
        # requires_grad=False: CHL updates via .data +=; use W_ff.requires_grad_(True)
        # at the model level to switch to backprop mode when needed.
        self.W_ff = F_init_weights((n_DG, n_CA3), 'mossy_fiber', mask=mask_ff)

        # Sparse perforant path mask: 25% of ECin → CA3 connections exist
        mask_ecin = (torch.rand(n_ECin, n_CA3) < ecin_frac).float()
        self.register_buffer('mask_ecin', mask_ecin)

        # W_ECin_CA3: perforant path direct ECin → CA3 (25% sparse), TSP pathway
        # Schapiro (2017) §2.a.iii: "direct EC input to CA3 (25% connectivity)"
        # Init: uniform(0.25, 0.75) — SI Table 2 'default'; see util._W_INIT.
        self.W_ecin = F_init_weights((n_ECin, n_CA3), 'default', mask=mask_ecin)

        # W_CA3_CA3: recurrent weights (enable pattern completion)
        # Schapiro (2017) §2.a.iii: "fully connected (every unit to every other unit)"
        # Init: uniform(0.25, 0.75) — SI Table 2 'default'; see util._W_INIT.
        # requires_grad=False: CHL updates via .data +=; use W_rec.requires_grad_(True)
        # at the model level to switch to backprop mode when needed.
        self.W_rec = F_init_weights((n_CA3, n_CA3), 'default')

        # Separate Euler state (membrane potential) and firing rate.
        # Recurrent input is y (firing rate), not Vm (membrane potential).
        self.register_buffer('_Vm', torch.zeros(n_CA3))   # membrane potential (Euler state)
        self.register_buffer('_y',  torch.zeros(n_CA3))   # firing rate (kWTA output)

    @property
    def activity(self) -> torch.Tensor:
        return self._y

    def reset(self) -> None:
        """Reset Vm and y to zero before each trial (before Q1 minus phase)."""
        self._Vm.zero_()
        self._y.zero_()

    def forward(self, a_DG: torch.Tensor, a_ECin: torch.Tensor) -> torch.Tensor:
        """One settling step: DG + ECin + y_CA3_prev → net → Euler on Vm → nxx1 → kWTA → y.

        net_CA3(t)  = a_DG @ W_ff + a_ECin @ W_ecin + y_CA3(t−1) @ W_rec   [y, not Vm]
        Vm_CA3(t)   = (1−tau) * Vm_CA3(t−1) + tau * net_CA3(t)
        y_CA3(t)    = kWTA(nxx1(Vm_CA3(t)))
        O'Reilly & Munakata (2000) Ch. 2.
        """
        # Mossy fibre: DG → CA3 (5% sparse); Schapiro (2017) §2.a.iii
        net = a_DG @ (self.W_ff * self.mask_ff)   # (n_CA3,)

        # Perforant path: ECin → CA3 direct (25% sparse); Schapiro (2017) §2.a.iii
        net = net + a_ECin @ (self.W_ecin * self.mask_ecin)   # (n_CA3,)

        # Recurrent: CA3 → CA3 (fully connected; pattern completion attractor)
        # Uses _y (firing rate), not _Vm (membrane potential).
        # Schapiro (2017) §2.a.iii: "fully connected projection to itself"
        # O'Reilly & Munakata (2000) Ch. 2: recurrent input = W_rec @ y(t-1)
        net = net + self._y @ self.W_rec   # (n_CA3,)

        # Euler integration on membrane potential; O'Reilly & Munakata (2000) Ch. 2
        if self.use_euler:
            self._Vm = (1.0 - self.tau) * self._Vm + self.tau * net
        else:
            self._Vm = net

        # Firing rate: nxx1 then kWTA (~6% active); Schapiro (2017) §2.a.iii SI Table 1
        self._y = F_kWTA(F_nxx1(self._Vm), k_frac=self.k_frac)
        return self._y

    def update_weights(
        self,
        a_DG_minus: torch.Tensor, a_DG_plus: torch.Tensor,
        a_ECin_minus: torch.Tensor, a_ECin_plus: torch.Tensor,
        a_CA3_minus: torch.Tensor, a_CA3_plus: torch.Tensor,
    ) -> None:
        """CHL weight update for W_DG_CA3, W_ECin_CA3, and W_CA3_CA3.

        ΔW_ff   = lr * (a_DG_plus ⊗ a_CA3_plus   − a_DG_minus ⊗ a_CA3_minus)   masked by mask_ff
        ΔW_ecin = lr * (a_ECin_plus ⊗ a_CA3_plus  − a_ECin_minus ⊗ a_CA3_minus) masked by mask_ecin
        ΔW_rec  = lr * (a_CA3_plus ⊗ a_CA3_plus   − a_CA3_minus ⊗ a_CA3_minus)
        O'Reilly & Munakata (2000) Ch. 4 Eq. 4.3; Schapiro (2017) §2.b.
        """
        # Feedforward (mossy fibre): W_ff is (n_DG, n_CA3)
        delta_ff = (torch.outer(a_DG_plus,  a_CA3_plus)
                  - torch.outer(a_DG_minus, a_CA3_minus))
        self.W_ff.data += self.lr * self.mask_ff * delta_ff

        # Perforant path direct (ECin → CA3): W_ecin is (n_ECin, n_CA3)
        delta_ecin = (torch.outer(a_ECin_plus,  a_CA3_plus)
                    - torch.outer(a_ECin_minus, a_CA3_minus))
        self.W_ecin.data += self.lr * self.mask_ecin * delta_ecin

        # Recurrent (Hopfield auto-association): W_rec is (n_CA3, n_CA3)
        # pre = post = CA3 activity → symmetric weight update
        # Schapiro (2017) §2.a.iii: "helps bind pieces of a representation"
        delta_rec = (torch.outer(a_CA3_plus,  a_CA3_plus)
                   - torch.outer(a_CA3_minus, a_CA3_minus))
        self.W_rec.data += self.lr * delta_rec


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
#   - activity : (n_CA1,) — ~25% active via kWTA
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
#   k_frac   : float (default: 0.25; Schapiro 2017 SI Table 1)
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
    """CA1 field: MSP + TSP convergence (~25% active).

    Stateful (Euler integration). Call reset() before each trial's minus phase.
    """

    def __init__(
        self,
        n_items: int,
        n_CA3: int,
        n_CA1: int = 50,
        k_frac: float = 0.25,
        tau: float = 0.1,
        use_euler: bool = True,
        lr_MSP: float = 0.05,
        lr_TSP: float = 0.4,
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
            kWTA sparsity. Schapiro (2017) SI Table 1: 0.25.
        tau : float
            Euler time constant. Leabra default: 0.1.
        use_euler : bool
            If False, stateless mode (for unit tests).
        lr_MSP : float
            MSP learning rate (ECin→CA1, ECout→CA1). Go reimplementation: 0.05.
        lr_TSP : float
            TSP learning rate (CA3→CA1). Go reimplementation: 0.4.
        """
        super().__init__()
        self.n_items = n_items
        self.n_CA3 = n_CA3
        self.n_CA1 = n_CA1
        self.k_frac = k_frac
        self.tau = tau
        self.use_euler = use_euler
        # Learning rates are per weight matrix, not per CA1 unit.
        # The same CA1 unit j receives synapses from both ECin (W_ECin[:, j]) and CA3
        # (W_CA3[:, j]). These are anatomically distinct projections (MSP = ECin Layer III;
        # TSP = CA3 Schaffer collaterals), so each carries independent plasticity.
        # MSP slow (0.05): accumulates statistics across many trials → community structure.
        # TSP fast (0.4):  binds a single episode before it is overwritten → episodic memory.
        # Schapiro (2017) §2.b; Go reimplementation values.
        self.lr_MSP = lr_MSP
        self.lr_TSP = lr_TSP

        # MSP: ECin → CA1 (slow; learns statistical regularities)
        # Schapiro (2017) §2.a.iv: "fully connected projections in the MSP"
        # lr_MSP = 0.05 (Go reimplementation; Schapiro 2017 §2.b)
        # Init: uniform(0.25, 0.75) — SI Table 2 'default'; see util._W_INIT.
        # Forward scale: NET_SCALE['ecin_ca1'] = 3.0 (SI Table 2 abs=3).
        self.W_ECin = F_init_weights((n_items, n_CA1), 'default')

        # TSP: CA3 → CA1 (fast; learns individual episodes)
        # Schapiro (2017) §2.a.iv: "CA3 then has a fully connected projection to CA1"
        # lr_TSP = 0.4 (Go reimplementation; Schapiro 2017 §2.b)
        # Init: uniform(0.25, 0.75) — SI Table 2 'default'; see util._W_INIT.
        self.W_CA3 = F_init_weights((n_CA3, n_CA1), 'default')

        # Back-projection from ECout (plus-phase teaching signal)
        # Schapiro (2017) §2.a.iv: ECout → CA1 "fully connected"
        # Completes the "big loop": ECin → CA1 → ECout → CA1
        # Init: uniform(0.25, 0.75) — SI Table 2 'default'; see util._W_INIT.
        self.W_ECout = F_init_weights((n_items, n_CA1), 'default')

        self.register_buffer('_Vm', torch.zeros(n_CA1))   # membrane potential (Euler state)
        self.register_buffer('_y',  torch.zeros(n_CA1))   # firing rate (kWTA output)

    @property
    def activity(self) -> torch.Tensor:
        return self._y

    def reset(self) -> None:
        """Reset Vm and y to zero before each trial (before Q1 minus phase)."""
        self._Vm.zero_()
        self._y.zero_()

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

        net      = W_ECin @ a_ECin + W_CA3 @ a_CA3 [+ W_ECout @ a_ECout if plus]
        Vm(t)    = (1−tau) * Vm(t−1) + tau * net(t)
        y_CA1(t) = kWTA(nxx1(Vm(t)))
        Schapiro (2017) §2.a.iv; O'Reilly & Munakata (2000) Ch. 2.
        """
        # MSP: ECin → CA1 with scale 3 (SI Table 2 abs=3; see util.NET_SCALE)
        # TSP: CA3 → CA1 with scale 1 (SI Table 2 default)
        # Schapiro (2017) §2.a.iv
        net = NET_SCALE['ecin_ca1'] * (a_ECin @ self.W_ECin) + a_CA3 @ self.W_CA3   # (n_CA1,)

        # Plus-phase back-projection: ECout → CA1 (teaching signal)
        # Schapiro (2017) §2.a.iv: "big loop" — ECout → CA1 fully connected
        if a_ECout is not None:
            net = net + a_ECout @ self.W_ECout             # (n_CA1,)

        # Euler on membrane potential; O'Reilly & Munakata (2000) Ch. 2
        if self.use_euler:
            self._Vm = (1.0 - self.tau) * self._Vm + self.tau * net
        else:
            self._Vm = net

        # Firing rate: nxx1 then kWTA (~25% active); Schapiro (2017) §2.a.iv SI Table 1
        self._y = F_kWTA(F_nxx1(self._Vm), k_frac=self.k_frac)
        return self._y

    def update_weights(
        self,
        a_ECin: torch.Tensor,
        a_CA3_minus: torch.Tensor, a_CA3_plus: torch.Tensor,
        a_ECout_minus: torch.Tensor, a_ECout_plus: torch.Tensor,
        a_CA1_minus: torch.Tensor, a_CA1_plus: torch.Tensor,
    ) -> None:
        """CHL weight updates for W_ECin (MSP), W_CA3 (TSP), W_ECout.

        Called once per trial after Q4. ECin is identical in minus and plus phases
        (same stimulus throughout; Schapiro 2017 §2.c), so one tensor suffices.

        ΔW_ECin  = lr_MSP * (a_ECin ⊗ (a_CA1_plus − a_CA1_minus))   [factored: ECin same both phases]
        ΔW_CA3   = lr_TSP * (a_CA3_plus ⊗ a_CA1_plus − a_CA3_minus ⊗ a_CA1_minus)
        ΔW_ECout = lr_MSP * (a_ECout_plus ⊗ a_CA1_plus − a_ECout_minus ⊗ a_CA1_minus)
        O'Reilly & Munakata (2000) Ch. 4 Eq. 4.3; Schapiro (2017) §2.b.
        """
        # MSP: ECin → CA1; a_ECin same in both phases → factored as a_ECin ⊗ (plus − minus)
        self.W_ECin.data += self.lr_MSP * torch.outer(
            a_ECin, a_CA1_plus - a_CA1_minus
        )
        # TSP: CA3 → CA1; CA3 differs between Q3 and Q4 end states
        self.W_CA3.data += self.lr_TSP * (
            torch.outer(a_CA3_plus,  a_CA1_plus)
          - torch.outer(a_CA3_minus, a_CA1_minus)
        )
        # ECout → CA1: full CHL — ECout active in both minus (free) and plus (clamped) phases
        # Schapiro (2017) §2.b; O'Reilly & Munakata (2000) Ch. 4 Eq. 4.3
        self.W_ECout.data += self.lr_MSP * (
            torch.outer(a_ECout_plus,  a_CA1_plus)
          - torch.outer(a_ECout_minus, a_CA1_minus)
        )


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
#   ΔW = lr * (a_CA1_plus ⊗ a_ECout_plus − a_CA1_minus ⊗ a_ECout_minus)
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
        lr: float = 0.05,
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
        lr : float
            Output learning rate. Go reimplementation: 0.05. (Schapiro 2017 §2.b)
        """
        super().__init__()
        self.n_CA1 = n_CA1
        self.n_items = n_items
        self.k = k           # Schapiro (2017) §2.a.ii: k=2 (absolute)
        self.tau = tau
        self.use_euler = use_euler
        self.lr = lr

        # W_CA1_ECout: CA1 → ECout feedforward; fully connected
        # Schapiro (2017) §2.a.iv: "fully connected projections in the MSP
        # from ECin to CA1, CA1 to ECout, and ECout to CA1."
        # Init: uniform(0.25, 0.75) — SI Table 2 'default'; see util._W_INIT.
        # requires_grad=False: CHL updates via .data +=; use W.requires_grad_(True)
        # at the model level to switch to backprop mode when needed.
        self.W = F_init_weights((n_CA1, n_items), 'default')

        self.register_buffer('_Vm', torch.zeros(n_items))   # membrane potential (Euler state)
        self.register_buffer('_y',  torch.zeros(n_items))   # firing rate (kWTA output)

    @property
    def activity(self) -> torch.Tensor:
        return self._y

    def reset(self) -> None:
        """Reset Vm and y to zero before each trial (before Q1 minus phase)."""
        self._Vm.zero_()
        self._y.zero_()

    def clamp(self, target_pattern: torch.Tensor) -> None:
        """Clamp ECout to target pattern for the plus phase (Q4).

        Sets both _y and _Vm to the target so that the next Euler step
        continues from the clamped state rather than the last predicted state.
        Schapiro (2017) §2.b: "in the plus phase, the model is directly
        shown the correct output."
        """
        self._y.copy_(target_pattern.float())
        self._Vm.copy_(target_pattern.float())

    def forward(self, a_CA1: torch.Tensor) -> torch.Tensor:
        """One settling step (minus phases Q1/Q2-Q3 only).

        net        = W @ a_CA1
        Vm(t)      = (1−tau) * Vm(t−1) + tau * net(t)
        y_ECout(t) = kWTA_k2(nxx1(Vm(t)))   [k=2 absolute; Schapiro 2017 §2.a.ii]

        In plus phase, use clamp() instead — do not call forward().
        """
        # net input from CA1: W is (n_CA1, n_items), a_CA1 is (n_CA1,)
        # Schapiro (2017) §2.a.iv; O'Reilly & Munakata (2000) Ch. 2.
        net = a_CA1 @ self.W                   # (n_items,)

        # Euler integration on membrane potential; O'Reilly & Munakata (2000) Ch. 2
        if self.use_euler:
            self._Vm = (1.0 - self.tau) * self._Vm + self.tau * net
        else:
            self._Vm = net

        # Firing rate: nxx1 then kWTA absolute k=2; Schapiro (2017) §2.a.ii
        vm = F_nxx1(self._Vm)
        k_from_bottom = self.n_items - self.k + 1
        threshold = torch.kthvalue(vm, k_from_bottom).values
        self._y = vm * (vm >= threshold).float()
        return self._y

    def update_weights(
        self,
        a_CA1_minus: torch.Tensor, a_CA1_plus: torch.Tensor,
        a_ECout_minus: torch.Tensor, a_ECout_plus: torch.Tensor,
    ) -> None:
        """CHL weight update for W_CA1_ECout.

        ΔW = lr * (a_CA1_plus ⊗ a_ECout_plus − a_CA1_minus ⊗ a_ECout_minus)
        O'Reilly & Munakata (2000) Ch. 4 Eq. 4.3; Schapiro (2017) §2.b.
        """
        # W is (n_CA1, n_items): outer(a_CA1, a_ECout) has correct shape.
        delta_plus  = torch.outer(a_CA1_plus,  a_ECout_plus)
        delta_minus = torch.outer(a_CA1_minus, a_ECout_minus)
        self.W.data += self.lr * (delta_plus - delta_minus)


# =========================================================================
# L_PFC: PREFRONTAL CORTEX RECURRENT LAYER
# =========================================================================
# [Role]:
#   Recurrent PFC module that receives ECout activity each cycle and projects
#   a net input back to ECin, closing the ECout → PFC → ECin loop.
#
#   Injection pattern at model level (M_Hip_KM):
#       net_pfc       = pfc(a_ECout)
#       a_ecin        = ecin(clamp, net_pfc=net_pfc)
#
#   rnn_type selects the recurrent cell:
#     'RNN'  — vanilla Elman RNN; interpretable; short sequences
#     'GRU'  — recommended default; fewer params than LSTM
#     'LSTM' — tuple hidden (h, c); long-range carry-over
#
#   Copied and adapted from EmbeddingRNN/src/layer.py:
#   - Explicit params (not cfg dict); consistent with EChipp_SL style.
#   - forward(a_ECout) → net_pfc (n_ECin,) for direct ECin injection.
#   - reset() zeroes hidden state at trial start.
#   - h property exposes hidden state (n_hidden,) for n_stable/n_dynamic analysis.
#   - No readout/logits layer; output IS the ECin projection.
#   - Trained with backprop (requires_grad=True by default); orthogonal to CHL layers.


class L_PFC(nn.Module):
    """Recurrent PFC: ECout → PFC hidden → net input to ECin.

    Parameters
    ----------
    n_ECout : int
        Input size (ECout activity; = n_items).
    n_hidden : int
        PFC recurrent hidden units.
    n_ECin : int
        Output projection size (= L_ECin.n_units).
    rnn_type : str
        One of 'RNN', 'GRU' (default), 'LSTM'.
    """

    _RNN_CLASSES = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}

    def __init__(self, n_ECout: int, n_hidden: int, n_ECin: int,
                 rnn_type: str = 'GRU'):
        super().__init__()
        if rnn_type not in self._RNN_CLASSES:
            raise ValueError(f"rnn_type must be one of {list(self._RNN_CLASSES)}; got {rnn_type!r}")
        self.rnn_type  = rnn_type
        self.n_hidden  = n_hidden
        self.rnn       = self._RNN_CLASSES[rnn_type](n_ECout, n_hidden, batch_first=True)
        self.proj_ecin = nn.Linear(n_hidden, n_ECin)
        self._hidden   = None  # Tensor (RNN/GRU) or tuple (LSTM)

    def reset(self) -> None:
        """Zero hidden state. Call once per trial before Q1."""
        self._hidden = None

    @property
    def h(self) -> torch.Tensor | None:
        """Current hidden state h_n (n_hidden,) for n_stable/n_dynamic analysis."""
        if self._hidden is None:
            return None
        # LSTM hidden is (h_n, c_n); RNN/GRU hidden is h_n directly.
        hn = self._hidden[0] if self.rnn_type == 'LSTM' else self._hidden
        return hn[0, 0]

    def forward(self, a_ECout: torch.Tensor) -> torch.Tensor:
        """One RNN step. Returns net_pfc (n_ECin,) to inject into L_ECin.forward()."""
        x = a_ECout.unsqueeze(0).unsqueeze(0)        # (1, 1, n_ECout)
        _, self._hidden = self.rnn(x, self._hidden)
        hn = self._hidden[0] if self.rnn_type == 'LSTM' else self._hidden
        return self.proj_ecin(hn[0, 0])              # (n_ECin,)
