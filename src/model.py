"""M_Hip — full hippocampal model assembly.

Schapiro et al. (2017) Complementary learning systems within the hippocampus.
Phil. Trans. R. Soc. B 372: 20160049.

M_Hip is task-agnostic: it receives (a_ecin_clamp, a_target) per trial and
returns (act_mid, act_m, act_p). Moving-window construction, random walk,
epoch iteration, and analysis all belong to the caller (tasks.py + notebook).
"""
import torch
import torch.nn as nn

from layer import L_ECin, L_ECout, L_DG, L_CA3, L_CA1


class M_Hip(nn.Module):
    """Hippocampal circuit: TSP (ECin→DG→CA3→CA1) + MSP (ECin→CA1) + ECout.

    Trial structure — theta_discrete convention (Schapiro 2017 §2.b):
      Q1      (cycles   1–25): ECin-dominant  (CA3→CA1 zeroed)
      Q2–Q3   (cycles  26–75): CA3-dominant   (ECin→CA1 zeroed)
      Q4      (cycles 76–100): plus phase     (ECout clamped to target)

    DG and CA3 run in ALL quarters; only their contribution to CA1 is gated.
    reset() is called once per trial before Q1; never between Q1→Q2-Q3→Q4.
    """

    def __init__(
        self,
        n_items: int,
        n_DG: int = 100,
        n_CA3: int = 50,
        n_CA1: int = 50,
        k_frac_DG: float = 0.01,
        k_frac_CA3: float = 0.06,
        k_frac_CA1: float = 0.25,
        ecin_frac: float = 0.25,
        dg_frac: float = 0.05,
        tau: float = 0.1,
        lr_MSP: float = 0.05,
        lr_TSP: float = 0.4,
        n_cycles_Q1: int = 25,
        n_cycles_Q23: int = 50,
        n_cycles_Q4: int = 25,
    ):
        """
        Parameters
        ----------
        n_items : int
            ECin / ECout dimension (15 for community-graph task; Schapiro 2017 Fig. 3).
        n_DG, n_CA3, n_CA1 : int
            Layer sizes. Schapiro (2017) SI Table 1 defaults.
        k_frac_DG, k_frac_CA3, k_frac_CA1 : float
            kWTA sparsity per layer. Schapiro (2017) SI Table 1.
        ecin_frac : float
            ECin→DG and ECin→CA3 connectivity fraction. Schapiro (2017) §2.a.iii: 0.25.
        dg_frac : float
            DG→CA3 mossy-fibre connectivity fraction. Schapiro (2017) §2.a.iii: 0.05.
        tau : float
            Euler time constant (all settling layers). Leabra default: 0.1.
        lr_MSP, lr_TSP : float
            CHL learning rates. Go reimplementation: 0.05 / 0.4. Schapiro (2017) §2.b.
        n_cycles_Q1, n_cycles_Q23, n_cycles_Q4 : int
            Settling cycles per quarter. Go reimplementation: 25 / 50 / 25.
        """
        super().__init__()
        self.n_items      = n_items
        self.n_DG         = n_DG
        self.n_CA3        = n_CA3
        self.n_CA1        = n_CA1
        self.k_frac_DG    = k_frac_DG
        self.k_frac_CA3   = k_frac_CA3
        self.k_frac_CA1   = k_frac_CA1
        self.ecin_frac    = ecin_frac
        self.dg_frac      = dg_frac
        self.tau          = tau
        self.lr_MSP       = lr_MSP
        self.lr_TSP       = lr_TSP
        self.n_cycles_Q1  = n_cycles_Q1
        self.n_cycles_Q23 = n_cycles_Q23
        self.n_cycles_Q4  = n_cycles_Q4

        # ECin: kWTA k=2 (Schapiro 2017 §2.a.ii); big loop enabled (n_ECout=n_items)
        self.ecin = L_ECin(n_units=n_items, n_ECout=n_items, tau=tau)

        # TSP layers
        self.dg  = L_DG(n_input=n_items, n_DG=n_DG, k_frac=k_frac_DG,
                        ecin_frac=ecin_frac, tau=tau, lr=lr_TSP)
        self.ca3 = L_CA3(n_DG=n_DG, n_ECin=n_items, n_CA3=n_CA3,
                         k_frac=k_frac_CA3, dg_frac=dg_frac,
                         ecin_frac=ecin_frac, tau=tau, lr=lr_TSP)

        # CA1: MSP (lr_MSP) + TSP (lr_TSP) convergence
        self.ca1 = L_CA1(n_items=n_items, n_CA3=n_CA3, n_CA1=n_CA1,
                         k_frac=k_frac_CA1, tau=tau,
                         lr_MSP=lr_MSP, lr_TSP=lr_TSP)

        # ECout: reconstruction + plus-phase teaching signal (lr_MSP)
        self.ecout = L_ECout(n_CA1=n_CA1, n_items=n_items, tau=tau, lr=lr_MSP)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset Euler state of all stateful layers. Call once per trial before Q1."""
        self.dg.reset()
        self.ca3.reset()
        self.ca1.reset()
        self.ecout.reset()

    @staticmethod
    def _snap(a_ecin, a_dg, a_ca3, a_ca1, a_ecout) -> dict:
        """Snapshot current layer activities into a plain dict of cloned tensors."""
        return {
            'ecin':  a_ecin.clone(),
            'dg':    a_dg.clone(),
            'ca3':   a_ca3.clone(),
            'ca1':   a_ca1.clone(),
            'ecout': a_ecout.clone(),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_trial(
        self,
        a_ecin_clamp: torch.Tensor,
        a_target: torch.Tensor,
    ) -> tuple:
        """Run one CHL trial (Q1 → Q2-Q3 → Q4) and return layer activations.

        Parameters
        ----------
        a_ecin_clamp : (n_items,)
            Moving-window ECin pattern built by the caller:
            current item = 1.0, previous item = 0.9. Schapiro (2017) §2.c.
        a_target : (n_items,)
            Next item one-hot; used to clamp ECout in Q4 (plus phase).

        Returns
        -------
        act_mid : dict  — end of Q1  (cycle 25;  ActMid in Go reimplementation)
        act_m   : dict  — end of Q2-Q3 (cycle 75;  ActM)
        act_p   : dict  — end of Q4  (cycle 100; ActP)
        Each dict has keys: 'ecin', 'dg', 'ca3', 'ca1', 'ecout'.
        """
        self.reset()

        dev = a_target.device
        zeros_ca3  = torch.zeros(self.n_CA3,  device=dev)
        zeros_ecin = torch.zeros(self.n_items, device=dev)
        a_ecout    = torch.zeros(self.n_items, device=dev)   # initial ECout for big loop

        # Q1 — ECin-dominant (theta trough; encoding; cycles 1–25)
        # CA3→CA1 zeroed; DG and CA3 still run to build their states.
        # Schapiro (2017) §2.b: "ECin projects strongly to CA1, CA3→CA1 inhibited."
        for _ in range(self.n_cycles_Q1):
            a_ecin  = self.ecin(a_ecin_clamp, a_ecout)
            a_dg    = self.dg(a_ecin)
            a_ca3   = self.ca3(a_dg, a_ecin)
            a_ca1   = self.ca1(a_ecin, zeros_ca3)
            a_ecout = self.ecout(a_ca1)
        act_mid = self._snap(a_ecin, a_dg, a_ca3, a_ca1, a_ecout)

        # Q2-Q3 — CA3-dominant (theta peak; retrieval; cycles 26–75)
        # ECin→CA1 zeroed; no reset (Euler state continues from Q1).
        # Schapiro (2017) §2.b: "CA3 projects strongly to CA1, ECin→CA1 inhibited."
        for _ in range(self.n_cycles_Q23):
            a_ecin  = self.ecin(a_ecin_clamp, a_ecout)
            a_dg    = self.dg(a_ecin)
            a_ca3   = self.ca3(a_dg, a_ecin)
            a_ca1   = self.ca1(zeros_ecin, a_ca3)
            a_ecout = self.ecout(a_ca1)
        act_m = self._snap(a_ecin, a_dg, a_ca3, a_ca1, a_ecout)

        # Q4 — plus phase (cycles 76–100); ECout clamped to target; no reset.
        # Big loop: a_target used as ECout for L_ECin (corrects ECin representation).
        # ECout.forward() not called — ECout is clamped, not settled.
        # Schapiro (2017) §2.b: "target pattern is directly clamped on ECout."
        for _ in range(self.n_cycles_Q4):
            a_ecin = self.ecin(a_ecin_clamp, a_target)
            a_dg   = self.dg(a_ecin)
            a_ca3  = self.ca3(a_dg, a_ecin)
            a_ca1  = self.ca1(a_ecin, a_ca3, a_ECout=a_target)
        act_p = self._snap(a_ecin, a_dg, a_ca3, a_ca1, a_target)

        return act_mid, act_m, act_p

    def update_weights(self, act_m: dict, act_p: dict) -> None:
        """CHL weight update for all projections. Call once after run_trial.

        Uses ActM (end of Q2-Q3) as minus phase and ActP (end of Q4) as plus phase.
        ΔW = lr × (ActP_post ⊗ ActP_pre − ActM_post ⊗ ActM_pre)
        O'Reilly & Munakata (2000) Ch. 4 Eq. 4.3; Schapiro (2017) §2.b.

        ECin is the same stimulus in both phases; L_CA1.update_weights accepts a
        single a_ECin tensor (factored form). act_m['ecin'] is used as the canonical
        ECin value (dominant in Q2-Q3; clamp_pattern dominates over big-loop noise).
        """
        # Big loop: ECout → ECin (lr_MSP via L_ECin.W_ECout_bigloop)
        self.ecin.update_weights(
            a_ECout_minus=act_m['ecout'], a_ECout_plus=act_p['ecout'],
            a_ECin_minus=act_m['ecin'],   a_ECin_plus=act_p['ecin'],
        )
        # TSP: ECin → DG
        self.dg.update_weights(
            a_ECin_minus=act_m['ecin'], a_ECin_plus=act_p['ecin'],
            a_DG_minus=act_m['dg'],     a_DG_plus=act_p['dg'],
        )
        # TSP: ECin → CA3, DG → CA3, CA3 → CA3 (recurrent)
        self.ca3.update_weights(
            a_DG_minus=act_m['dg'],     a_DG_plus=act_p['dg'],
            a_ECin_minus=act_m['ecin'], a_ECin_plus=act_p['ecin'],
            a_CA3_minus=act_m['ca3'],   a_CA3_plus=act_p['ca3'],
        )
        # MSP: ECin→CA1 (lr_MSP); TSP: CA3→CA1 (lr_TSP); ECout→CA1 back-proj (lr_MSP)
        self.ca1.update_weights(
            a_ECin=act_m['ecin'],
            a_CA3_minus=act_m['ca3'],     a_CA3_plus=act_p['ca3'],
            a_ECout_minus=act_m['ecout'], a_ECout_plus=act_p['ecout'],
            a_CA1_minus=act_m['ca1'],     a_CA1_plus=act_p['ca1'],
        )
        # CA1 → ECout (lr_MSP)
        self.ecout.update_weights(
            a_CA1_minus=act_m['ca1'],     a_CA1_plus=act_p['ca1'],
            a_ECout_minus=act_m['ecout'], a_ECout_plus=act_p['ecout'],
        )
