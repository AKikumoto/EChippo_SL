"""
Statistical learning tasks for the EC-hippocampus circuit.

Tasks
-----
Pair structure (§3.a):
    8 items in 4 fixed pairs (AB, CD, EF, GH); pair order randomized.
    T_PairEnv / T_PairDataset.

Chain / associative inference (§3.c):
    9 items in 3 triads (ABC/DEF/GHI); trains on A→B and B→C only.
    T_ChainDataset.

Community graph (§3.b):
    15 items in 5 communities × 3 items; random walk on graph.
    T_CommunityGraphEnv / T_CommunityGraphDataset.

Weather Prediction:
    4 probabilistic cues (A–D); 14 multi-cue patterns; rain/sun outcome.
    T_WeatherEnv / T_WeatherDataset.
    Frank, M. J. (2005). Dynamic dopamine modulation in the basal ganglia.
    J. Cogn. Neurosci., 17(1), 51–72.

Sequence tasks produce (item, next_item) pairs for CHL training:
    item       → ECin clamped pattern (current stimulus)
    next_item  → ECout clamped pattern (plus-phase teaching signal)

Weather task produces (cue_pattern, outcome) for CHL training:
    cue_pattern → ECin binary vector (n_cues,)
    outcome     → ECout target (0=sun, 1=rain)

Usage
-----
    from src.tasks import T_CommunityGraphEnv, T_CommunityGraphDataset
    from src.tasks import T_PairEnv, T_PairDataset
    from src.tasks import T_WeatherEnv, T_WeatherDataset

    env  = T_CommunityGraphEnv(n_communities=5, items_per_community=3)
    item = env.reset(seed=42)
    current, next_item = env.step()

    env = T_WeatherEnv(seed=42)
    env.reset()
    cue_pattern, outcome = env.step()

    dataset = T_WeatherDataset(n_trials=400, seed=42)
    step    = dataset[0]   # dict with cue_pattern, outcome, p_rain, pattern_idx

References
----------
Schapiro, A. C., Turk-Browne, N. B., Botvinick, M. M., & Norman, K. A. (2017).
    Complementary learning systems within the hippocampus: a neural network
    modelling approach to reconciling episodic memory with statistical learning.
    Phil. Trans. R. Soc. B, 372, 20160049.

Frank, M. J. (2005). Dynamic dopamine modulation in the basal ganglia:
    a neurocomputational account of cognitive deficits in medicated and
    nonmedicated Parkinsonism. J. Cogn. Neurosci., 17(1), 51–72.
"""

import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset


# =========================================================================
# T_PairEnv / T_PairDataset: PAIR STRUCTURE TASK
# =========================================================================
# [Role]:
#   Schapiro (2017) §3.a — 4 fixed pairs (AB/CD/EF/GH); each trial presents
#   the first item as input and the second item as the ECout target.
#   Pair order is random with no back-to-back repetitions.
#   80 inputs/epoch (Schapiro 2017 §3.a).
#
# [Pair structure]:
#   Pair 0: items 0, 1  (AB)
#   Pair 1: items 2, 3  (CD)
#   Pair 2: items 4, 5  (EF)
#   Pair 3: items 6, 7  (GH)
#
# [Why pairs?]:
#   In contrast to the community graph, the pair task has no statistical
#   community structure — only pairwise associations. MSP cannot exploit
#   higher-order transitional regularities; TSP encodes each pair directly.
#   Used as a control condition to dissociate MSP from TSP contributions.

class T_PairEnv:
    """Pair structure environment (Schapiro 2017 §3.a).

    Two modes matching Schapiro's with/without statistical learning conditions:

    interleaved=False (default) — sequential walk (with statistical learning):
        A→B (deterministic), B→A of other 3 pairs (uniform).
        Both A and B items appear as ECin. Pairs are detected via statistics:
        P(A→B)=1.0 >> P(B→specific_C)=1/3.
        Schapiro (2017) p.4: "After AB, BC, BE, or BG followed with equal probability."

    interleaved=True — isolated pair presentation (without statistical learning):
        Each step returns (A, B) for a randomly selected pair; no B→A links.
        Only A items appear as ECin. Pairs are directly memorized by TSP.
        Schapiro (2017) p.4: "AB, CD, EF, and GH all appeared but never BC or FG."
    """

    def __init__(
        self,
        n_pairs:     int            = 4,
        interleaved: bool           = False,
        seed:        Optional[int]  = None,
    ):
        self.n_pairs     = n_pairs
        self.n_items     = n_pairs * 2
        self.interleaved = interleaved
        # Pair i: items (2i, 2i+1); Schapiro (2017) §3.a: AB/CD/EF/GH
        self.pairs     = [(2 * i, 2 * i + 1) for i in range(n_pairs)]
        self.rng       = np.random.default_rng(seed)
        self._current  = 0   # sequential mode: current item in walk
        self._pair_idx = 0   # interleaved mode: current pair index

    def reset(self, seed: Optional[int] = None) -> int:
        """Start at a random position; return the first current item."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        pair_idx = int(self.rng.integers(self.n_pairs))
        if self.interleaved:
            self._pair_idx = pair_idx
            return self.pairs[pair_idx][0]
        else:
            self._current = self.pairs[pair_idx][0]
            return self._current

    def step(self) -> Tuple[int, int]:
        """Advance one step; return (current_item, next_item).

        Sequential: A→B (deterministic), B→A_other (uniform, own pair excluded).
        Interleaved: returns (A, B) for a randomly selected pair; no B→A links.

        Returns
        -------
        current_item : int  (input to ECin)
        next_item    : int  (ECout plus-phase target)
        """
        if self.interleaved:
            pair    = self.pairs[self._pair_idx]
            current = pair[0]
            target  = pair[1]
            # Next pair: exclude current (no back-to-back same pair)
            candidates     = [i for i in range(self.n_pairs) if i != self._pair_idx]
            self._pair_idx = int(self.rng.choice(candidates))
            return current, target

        current  = self._current
        pair_idx = current // 2
        if current % 2 == 0:
            # A item → B partner (deterministic)
            # Schapiro (2017) p.4: items within a pair always occurred in fixed order
            next_item = current + 1
        else:
            # B item → A item of one of the other 3 pairs (uniform)
            # Schapiro (2017) p.4: "After AB, BC, BE, or BG followed with equal probability"
            other_pairs = [i for i in range(self.n_pairs) if i != pair_idx]
            next_pair   = int(self.rng.choice(other_pairs))
            next_item   = self.pairs[next_pair][0]
        self._current = next_item
        return current, next_item

    @property
    def current_item(self) -> int:
        if self.interleaved:
            return self.pairs[self._pair_idx][0]
        return self._current


class T_PairDataset(Dataset):
    """Pre-generated pair task sequence for CHL training.

    Schapiro (2017) §3.a: 4 fixed pairs (AB/CD/EF/GH).

    interleaved=False (default): sequential walk (with statistical learning).
        Both A and B items appear as current item (ECin input).
    interleaved=True: isolated pair presentation (without statistical learning).
        Only A items appear as current item; pairs are never connected.
    """

    def __init__(
        self,
        n_steps:     int           = 800,
        n_pairs:     int           = 4,
        interleaved: bool          = False,
        device:      str           = "cpu",
        seed:        Optional[int] = None,
    ):
        """
        Parameters
        ----------
        n_steps : int
            Number of transitions to generate.
            Schapiro (2017) §3.a: 80 inputs/epoch; default 800 = 10 epochs.
        n_pairs : int
            Number of pairs. Schapiro (2017) §3.a: 4.
        interleaved : bool
            False (default): sequential walk (with SL).
            True: isolated pair presentation (without SL).
        device : str
            PyTorch device.
        seed : int, optional
            Random seed.
        """
        self.n_steps     = n_steps
        self.n_pairs     = n_pairs
        self.n_items     = n_pairs * 2
        self.pairs       = [(2 * k, 2 * k + 1) for k in range(n_pairs)]
        self.interleaved = interleaved
        self.device      = device

        self._items      : torch.Tensor = None
        self._next_items : torch.Tensor = None
        self._pair_labels: torch.Tensor = None

        self._generate(n_pairs, interleaved, seed)

    def _generate(self, n_pairs: int, interleaved: bool, seed: Optional[int]) -> None:
        """Run pair environment and store all (item, next_item) sequences."""
        env = T_PairEnv(n_pairs=n_pairs, interleaved=interleaved, seed=seed)
        env.reset()

        items, next_items, pair_labels = [], [], []
        for _ in range(self.n_steps):
            current, target = env.step()
            items.append(current)
            next_items.append(target)
            pair_labels.append(current // 2)

        self._items       = torch.tensor(items,       dtype=torch.long, device=self.device)
        self._next_items  = torch.tensor(next_items,  dtype=torch.long, device=self.device)
        self._pair_labels = torch.tensor(pair_labels, dtype=torch.long, device=self.device)

    def rsa_masks(self) -> Dict[str, np.ndarray]:
        """Boolean masks for same-pair vs cross-pair item comparisons.

        Returns
        -------
        dict with keys:
          'same'  : bool ndarray (n_items, n_items) — paired items (off-diagonal)
          'cross' : bool ndarray (n_items, n_items) — all other item pairs
        """
        pair_set = set(self.pairs) | {(b, a) for a, b in self.pairs}
        n = self.n_items
        same  = np.array([[(i, j) in pair_set and i != j for j in range(n)]
                          for i in range(n)])
        cross = np.array([[i != j and (i, j) not in pair_set for j in range(n)]
                          for i in range(n)])
        return {'same': same, 'cross': cross}

    def output_prob_by_epoch(self, ecout_arr: np.ndarray) -> np.ndarray:
        """Mean P(ECout_partner > 0.5) at each epoch, averaged over pairs and reps.

        Parameters
        ----------
        ecout_arr : ndarray (N_REPS, N_EPOCHS, n_items, n_items)

        Returns
        -------
        ndarray (N_EPOCHS,)
        """
        n_epochs = ecout_arr.shape[1]
        return np.array([
            np.mean([(ecout_arr[:, ep, a, b] > 0.5).mean() for a, b in self.pairs])
            for ep in range(n_epochs)
        ])

    def output_prob_detail_by_epoch(self, ecout_arr: np.ndarray) -> Dict[str, np.ndarray]:
        """Per-transition-type output probability curves (Schapiro 2017 Fig. 2c/f).

        For each pair (A, B) averaged over all pairs and reps:
          A→B : P(ECout_B > 0.5 | input A)  — partner activation
          B→A : P(ECout_A > 0.5 | input B)  — backward activation
          A→A : P(ECout_A > 0.5 | input A)  — self-activation
          B→B : P(ECout_B > 0.5 | input B)  — self-activation
          incorrect : mean P(ECout_k > 0.5 | input A) for k ∉ {A, B}

        Parameters
        ----------
        ecout_arr : ndarray (N_REPS, N_EPOCHS, n_items, n_items)
            ecout_arr[r, ep, i, j] = ECout unit j activity when presenting item i.

        Returns
        -------
        dict with keys 'A_B', 'B_A', 'A_A', 'B_B', 'incorrect',
        each ndarray (N_EPOCHS,)
        """
        n_epochs = ecout_arr.shape[1]
        keys = ['A_B', 'B_A', 'A_A', 'B_B', 'incorrect']
        accum = {k: [] for k in keys}
        for ep in range(n_epochs):
            vals = {k: [] for k in keys}
            for A, B in self.pairs:
                others = [k for k in range(self.n_items) if k != A and k != B]
                vals['A_B'].append((ecout_arr[:, ep, A, B] > 0.5).mean())
                vals['B_A'].append((ecout_arr[:, ep, B, A] > 0.5).mean())
                vals['A_A'].append((ecout_arr[:, ep, A, A] > 0.5).mean())
                vals['B_B'].append((ecout_arr[:, ep, B, B] > 0.5).mean())
                vals['incorrect'].append(
                    np.mean([(ecout_arr[:, ep, A, k] > 0.5).mean() for k in others])
                )
            for k in keys:
                accum[k].append(float(np.mean(vals[k])))
        return {k: np.array(v) for k, v in accum.items()}

    def __len__(self) -> int:
        return self.n_steps

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one (item, next_item) pair.

        Returns
        -------
        dict with keys:
          'item'          : LongTensor scalar
          'next_item'     : LongTensor scalar
          'pair'          : LongTensor scalar (pair index 0–n_pairs-1)
          'item_onehot'   : FloatTensor (n_items,)
          'target_onehot' : FloatTensor (n_items,)
        """
        item      = self._items[idx]
        next_item = self._next_items[idx]
        pair      = self._pair_labels[idx]

        item_onehot   = torch.zeros(self.n_items, device=self.device)
        target_onehot = torch.zeros(self.n_items, device=self.device)
        item_onehot[item]        = 1.0
        target_onehot[next_item] = 1.0

        return {
            'item':          item,
            'next_item':     next_item,
            'pair':          pair,
            'item_onehot':   item_onehot,
            'target_onehot': target_onehot,
        }


# =========================================================================
# T_ChainDataset: ASSOCIATIVE INFERENCE (CHAIN) TASK
# =========================================================================
# [Role]:
#   Schapiro (2017) §3.c — 9 items in 3 triads (ABC / DEF / GHI).
#   Trains on direct pairs only: A→B and B→C for each triad.
#   Test asks whether the network infers A→C (transitive, two-hop).
#   Requires CA3 recurrence for transitivity (Fig. 4).
#
# [Structure]:
#   Triad k: items (3k, 3k+1, 3k+2)
#   Direct pairs: (3k, 3k+1) and (3k+1, 3k+2) for each k
#   Between-triad connections: none
#
# [Relationship to T_PairDataset]:
#   T_PairDataset: A→B isolated (interleaved) or with statistical walk (sequential)
#   T_ChainDataset: A→B→C chains; no between-triad transitions; tests inference

class T_ChainDataset(Dataset):
    """Direct-pair training for associative inference task (Schapiro 2017 §3.c).

    Trains on A→B and B→C pairs within each triad only.
    Pairs are presented in random interleaved order; no between-triad transitions.

    Parameters
    ----------
    n_steps   : int   — number of training transitions (default 60)
    n_triads  : int   — number of triads (default 3; 9 items total)
    device    : str
    seed      : int, optional
    """

    def __init__(
        self,
        n_steps:  int           = 60,
        n_triads: int           = 3,
        device:   str           = 'cpu',
        seed:     Optional[int] = None,
    ):
        ipc                = 3                         # items per triad (fixed)
        self.n_triads      = n_triads
        self.ipc           = ipc
        self.n_items       = n_triads * ipc
        self.n_steps       = n_steps
        self.device        = device

        # Direct pairs: (3k, 3k+1) and (3k+1, 3k+2) for each triad k
        # Transitive pairs (never trained, test only): (3k, 3k+2)
        self.direct_pairs = [(t * ipc + i, t * ipc + i + 1)
                             for t in range(n_triads) for i in range(ipc - 1)]
        self.trans_pairs  = [(t * ipc, t * ipc + 2) for t in range(n_triads)]
        pairs = self.direct_pairs

        rng  = np.random.default_rng(seed)
        idxs = rng.integers(len(pairs), size=n_steps)

        items      = [pairs[i][0] for i in idxs]
        next_items = [pairs[i][1] for i in idxs]

        n = self.n_items
        self._item_oh   = torch.zeros(n_steps, n, device=device)
        self._target_oh = torch.zeros(n_steps, n, device=device)
        t_idx = torch.arange(n_steps)
        self._item_oh  [t_idx, torch.tensor(items)]      = 1.0
        self._target_oh[t_idx, torch.tensor(next_items)] = 1.0

    def rsa_masks(self) -> Dict[str, np.ndarray]:
        """Boolean masks for direct, transitive, and unrelated item pairs.

        Returns
        -------
        dict with keys:
          'direct'     : bool ndarray (n_items, n_items) — trained A↔B and B↔C pairs
          'transitive' : bool ndarray (n_items, n_items) — untrained A↔C (two-hop)
          'unrelated'  : bool ndarray (n_items, n_items) — cross-triad, off-diagonal
        """
        n          = self.n_items
        ipc        = self.ipc
        direct_set = set(self.direct_pairs) | {(b, a) for a, b in self.direct_pairs}
        trans_set  = set(self.trans_pairs)  | {(b, a) for a, b in self.trans_pairs}
        direct = np.array([[(i, j) in direct_set for j in range(n)] for i in range(n)])
        trans  = np.array([[(i, j) in trans_set  for j in range(n)] for i in range(n)])
        unrel  = np.array([[i != j
                            and (i, j) not in direct_set
                            and (i, j) not in trans_set
                            and i // ipc != j // ipc
                            for j in range(n)] for i in range(n)])
        return {'direct': direct, 'transitive': trans, 'unrelated': unrel}

    def output_prob(self, ecout_mat: np.ndarray) -> Dict[str, float]:
        """P(ECout_target > 0.5) for direct and transitive pairs.

        Parameters
        ----------
        ecout_mat : ndarray (N_REPS, n_items, n_items)
            Settled ECout activity for one epoch (typically the final epoch).
            ecout_mat[r, a, b] = ECout unit b activity when presenting item a, rep r.

        Returns
        -------
        dict with keys 'direct' and 'transitive' (float each)
        """
        return {
            'direct':     float(np.mean([(ecout_mat[:, a, b] > 0.5).mean()
                                         for a, b in self.direct_pairs])),
            'transitive': float(np.mean([(ecout_mat[:, a, b] > 0.5).mean()
                                         for a, b in self.trans_pairs])),
        }

    def __len__(self) -> int:
        return self.n_steps

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one (item, target) pair.

        Returns
        -------
        dict with keys 'item_onehot' (n_items,) and 'target_onehot' (n_items,)
        """
        return {
            'item_onehot':   self._item_oh[idx],
            'target_onehot': self._target_oh[idx],
        }


# =========================================================================
# T_CommunityGraphEnv: COMMUNITY GRAPH RANDOM WALK ENVIRONMENT
# =========================================================================
# [Role]:
#   Generates a random walk over the community graph.
#   Each step yields (current_item, next_item) pair for CHL training.
#
# [Graph structure]:
#   n_communities communities, each containing items_per_community items.
#   Within a community: all items are fully connected (triangle for ipc=3).
#   Between communities: ring topology — last node of community c connects
#   to first node of community (c+1) % n_communities (Schapiro 2017 Fig. 1).
#   Transition probabilities = uniform over neighbors (normalized adjacency).
#
# [Node degrees]:
#   Non-bottleneck nodes (middle of triangle): degree = items_per_community - 1
#   Bottleneck nodes (endpoints of ring edges): degree = items_per_community
#   This asymmetry produces natural within-community transition bias.
#
# [Training procedure]:
#   Schapiro (2017) §3.b: 60 inputs/epoch, 10 epochs
#   1. Start at a random item
#   2. Sample next_item from transition probabilities
#   3. Present current_item to ECin (input)
#   4. Present next_item to ECout (target, plus phase)
#   5. Run CHL; update weights
#   6. Move to next_item → repeat

class T_CommunityGraphEnv:
    """Random walk over the community graph (Schapiro 2017 Fig. 1).

    Call reset() to start, then step() repeatedly to get (current, next) pairs.
    """

    def __init__(
        self,
        n_communities: int = 5,
        items_per_community: int = 3,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        n_communities : int
            Number of communities. Schapiro (2017) Fig. 1: 5.
        items_per_community : int
            Items per community. Schapiro (2017) Fig. 1: 3.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.n_communities       = n_communities
        self.items_per_community = items_per_community
        self.n_items             = n_communities * items_per_community
        self.rng                 = np.random.default_rng(seed)

        # community[i] = community index for item i
        self.community = np.array([
            i // items_per_community for i in range(self.n_items)
        ])

        self._transition_matrix: np.ndarray = None
        self._build_graph()

        self._current_item: int = 0

    def _build_graph(self) -> None:
        """Build transition probability matrix from community structure.

        Within-community: fully connected (triangle for ipc=3).
        Schapiro (2017) Fig. 1.

        Between-community (ring): last node of community c ↔ first node of
        community (c+1) % n_communities. Produces natural bottleneck.

        Transition probabilities = row-normalized adjacency matrix.
        """
        ipc = self.items_per_community
        nc  = self.n_communities
        A   = np.zeros((self.n_items, self.n_items), dtype=np.float32)

        # Within-community: full triangles (Schapiro 2017 Fig. 1)
        for c in range(nc):
            for i in range(ipc):
                for j in range(i + 1, ipc):
                    a, b    = c * ipc + i, c * ipc + j
                    A[a, b] = A[b, a] = 1.0

        # Between-community ring: last of c ↔ first of (c+1)%nc
        # Schapiro (2017) Fig. 1: bottleneck nodes connect adjacent communities
        for c in range(nc):
            last  = c * ipc + (ipc - 1)
            first = ((c + 1) % nc) * ipc
            A[last, first] = A[first, last] = 1.0

        # Normalize rows → transition probabilities
        row_sums = A.sum(axis=1, keepdims=True)
        self._transition_matrix = A / row_sums

    def reset(self, seed: Optional[int] = None) -> int:
        """Reset to a random starting item.

        Parameters
        ----------
        seed : int, optional
            Reset the RNG seed.

        Returns
        -------
        item_idx : int
            Starting item index (0-based).
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._current_item = int(self.rng.integers(self.n_items))
        return self._current_item

    def step(self) -> Tuple[int, int]:
        """Sample one transition: current_item → next_item.

        Returns
        -------
        current_item : int
            Index of the current item (input to ECin).
        next_item : int
            Index of the next item (target for ECout plus phase).
        """
        current   = self._current_item
        probs     = self._transition_matrix[current]
        next_item = int(self.rng.choice(self.n_items, p=probs))
        self._current_item = next_item
        return current, next_item

    @property
    def current_item(self) -> int:
        return self._current_item


# =========================================================================
# T_CommunityGraphDataset: PYTORCH DATASET WRAPPER
# =========================================================================
# [Role]:
#   Pre-generates a sequence of (item, next_item) transitions by running
#   a random walk on the community graph. Wraps the sequence as a PyTorch
#   Dataset for use with DataLoader.
#
# [Data format]:
#   Each sample:
#     'item'          : LongTensor scalar — current item index
#     'next_item'     : LongTensor scalar — next item index
#     'community'     : LongTensor scalar — community label of current item
#     'item_onehot'   : FloatTensor (n_items,) — one-hot for current item
#     'target_onehot' : FloatTensor (n_items,) — one-hot for next item (ECout target)
#
# [Notes]:
#   Schapiro (2017) §3.b: 60 inputs/epoch × 10 epochs = 600 → default n_steps=600.

class T_CommunityGraphDataset(Dataset):
    """Pre-generated community graph random walk for CHL training.

    Schapiro (2017) §3.b: training sequence of (current_item, next_item) pairs.
    """

    def __init__(
        self,
        n_steps: int = 600,
        n_communities: int = 5,
        items_per_community: int = 3,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        n_steps : int
            Total number of transitions to generate.
            Schapiro (2017) §3.b: 60 inputs/epoch × 10 epochs = 600.
        n_communities : int
            Number of communities. Schapiro (2017) Fig. 1: 5.
        items_per_community : int
            Items per community. Schapiro (2017) Fig. 1: 3.
        device : str
            PyTorch device ('cpu' or 'cuda').
        seed : int, optional
            Random seed.
        """
        self.n_steps             = n_steps
        self.n_communities       = n_communities
        self.items_per_community = items_per_community
        self.n_items             = n_communities * items_per_community
        self.device              = device

        self._items      : torch.Tensor = None
        self._next_items : torch.Tensor = None
        self._communities: torch.Tensor = None

        self._generate(n_communities, items_per_community, seed)

    def _generate(
        self,
        n_communities: int,
        items_per_community: int,
        seed: Optional[int],
    ) -> None:
        """Run random walk and store all (item, next_item) transitions."""
        env = T_CommunityGraphEnv(
            n_communities=n_communities,
            items_per_community=items_per_community,
            seed=seed,
        )
        env.reset()

        items, next_items, communities = [], [], []
        for _ in range(self.n_steps):
            current, next_item = env.step()
            items.append(current)
            next_items.append(next_item)
            communities.append(int(env.community[current]))

        self._items       = torch.tensor(items,       dtype=torch.long, device=self.device)
        self._next_items  = torch.tensor(next_items,  dtype=torch.long, device=self.device)
        self._communities = torch.tensor(communities, dtype=torch.long, device=self.device)

    @property
    def internal_items(self) -> List[int]:
        """Items with only within-community edges (middle of each community).

        In the ring topology, only the first and last item of each community
        carry inter-community edges; all others are internal (degree = ipc-1).
        """
        ipc      = self.items_per_community
        boundary = {c * ipc for c in range(self.n_communities)} | \
                   {c * ipc + ipc - 1 for c in range(self.n_communities)}
        return [i for i in range(self.n_items) if i not in boundary]

    @property
    def boundary_items(self) -> List[int]:
        """Items carrying inter-community edges (first and last of each community)."""
        ipc = self.items_per_community
        return sorted({c * ipc for c in range(self.n_communities)} |
                      {c * ipc + ipc - 1 for c in range(self.n_communities)})

    def rsa_masks(self) -> Dict[str, np.ndarray]:
        """Boolean masks for within- vs between-community item pairs.

        Returns
        -------
        dict with keys:
          'within'  : bool ndarray (n_items, n_items) — same community, off-diagonal
          'between' : bool ndarray (n_items, n_items) — different communities
        """
        ipc          = self.items_per_community
        community_of = np.array([i // ipc for i in range(self.n_items)])
        within  = (community_of[:, None] == community_of[None, :]) & \
                  ~np.eye(self.n_items, dtype=bool)
        between = (community_of[:, None] != community_of[None, :])
        return {'within': within, 'between': between}

    def rsa_masks_detail(self) -> Dict[str, np.ndarray]:
        """Four-way boolean masks for detailed pattern similarity analysis.

        Schapiro (2017) Fig. 3e: six similarity categories across DG/CA3/CA1.
        Categories separate within-community pairs by boundary vs. internal node,
        and across-community pairs by direct ring connection vs. indirect.

        Returns
        -------
        dict with keys:
          'within_internal' : same community, BOTH nodes internal (not boundary)
          'within_boundary' : same community, AT LEAST ONE node is boundary
          'across_boundary' : different communities, directly connected by ring edge
          'across_other'    : different communities, not directly connected
        """
        ipc = self.items_per_community
        nc  = self.n_communities
        n   = self.n_items

        community_of = np.array([i // ipc for i in range(n)])
        same_comm = (community_of[:, None] == community_of[None, :])
        off_diag  = ~np.eye(n, dtype=bool)

        # Boundary items: first and last of each community (carry ring edges)
        boundary_set = set()
        for c in range(nc):
            boundary_set.add(c * ipc)
            boundary_set.add(c * ipc + ipc - 1)
        is_boundary = np.array([i in boundary_set for i in range(n)])

        # Ring connections: last of c ↔ first of (c+1)%nc
        ring = np.zeros((n, n), dtype=bool)
        for c in range(nc):
            last  = c * ipc + (ipc - 1)
            first = ((c + 1) % nc) * ipc
            ring[last, first] = ring[first, last] = True

        both_internal = (~is_boundary[:, None]) & (~is_boundary[None, :])

        return {
            'within_internal': same_comm & off_diag & both_internal,
            'within_boundary': same_comm & off_diag & ~both_internal,
            'across_boundary': (~same_comm) & ring,
            'across_other':    (~same_comm) & (~ring),
        }

    def output_prob_by_epoch(self, ecout_arr: np.ndarray) -> Dict[str, np.ndarray]:
        """P(argmax ECout is from same community) per epoch for internal vs boundary items.

        Schapiro (2017) Fig. 3c: probability of activating units from same community.
        Chance = items_per_community / n_items (= 0.33 for 3 communities × 5 items).

        Parameters
        ----------
        ecout_arr : ndarray (N_REPS, N_EPOCHS, n_items, n_items)
            ecout_arr[r, ep, i, :] = ECout activity when presenting item i.

        Returns
        -------
        dict with keys 'internal' and 'boundary', each ndarray (N_EPOCHS,)
        """
        n_epochs = ecout_arr.shape[1]
        ipc = self.items_per_community
        community_of = np.array([i // ipc for i in range(self.n_items)])

        def prob(items: List[int]) -> np.ndarray:
            return np.array([
                np.mean([
                    (community_of[np.argmax(ecout_arr[:, ep, i, :], axis=1)] == community_of[i]).mean()
                    for i in items
                ])
                for ep in range(n_epochs)
            ])

        return {'internal': prob(self.internal_items),
                'boundary': prob(self.boundary_items)}

    def __len__(self) -> int:
        return self.n_steps

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one (item, next_item) transition.

        Returns
        -------
        dict with keys:
          'item'          : LongTensor scalar
          'next_item'     : LongTensor scalar
          'community'     : LongTensor scalar
          'item_onehot'   : FloatTensor (n_items,)
          'target_onehot' : FloatTensor (n_items,)
        """
        item      = self._items[idx]
        next_item = self._next_items[idx]
        community = self._communities[idx]

        item_onehot   = torch.zeros(self.n_items, device=self.device)
        target_onehot = torch.zeros(self.n_items, device=self.device)
        item_onehot[item]        = 1.0
        target_onehot[next_item] = 1.0

        return {
            'item':          item,
            'next_item':     next_item,
            'community':     community,
            'item_onehot':   item_onehot,
            'target_onehot': target_onehot,
        }


# =========================================================================
# TASK DESIGN HELPERS — ECin initialisation utilities
# =========================================================================
# These functions operate on task design tables, NOT on neural responses.
# They produce *theoretical* RSA model matrices (what cosine similarity
# between conditions *would be* if the brain perfectly encoded a feature).
#
# For neural-response RSA (CA1 activity → correlation with model matrices),
# use F_rsa_fit in util.py (future implementation).
#
# design_to_units       : Task_Design.txt → per-feature one-hot matrices
# units_to_rsa          : one-hot matrices → theoretical RSA model matrices
# rsa_to_embedding_init : theoretical RSA matrix → MDS-style ECin init weights
# =========================================================================

def design_to_units(design: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Convert task design table to per-feature one-hot matrices.

    Parameters
    ----------
    design : pd.DataFrame
        Task_Design.txt; rows = conditions, columns = feature names
        (e.g. RULE, STIM, RESP, CONJ, COND).

    Returns
    -------
    dict {feature: np.ndarray (n_cond, n_values)}
        One-hot matrix per feature column (excluding 'COND').
    """
    units = {}
    skip = {'COND'}
    for col in design.columns:
        if col in skip:
            continue
        vals = design[col].values                      # (n_cond,) int
        n_values = int(vals.max())
        oh = np.zeros((len(design), n_values), dtype=np.float32)
        oh[np.arange(len(design)), vals - 1] = 1.0    # 1-indexed → 0-indexed
        units[col] = oh
    return units


def units_to_rsa(units: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Construct theoretical RSA model matrices from one-hot feature encodings.

    Each matrix M[i,j] is the cosine similarity between conditions i and j
    *assuming* the brain perfectly encodes that feature and nothing else.
    These are design-side model matrices, not measured neural similarities.

    For each feature, M[i,j] = dot(oh[i] / sqrt(k), oh[j] / sqrt(k))
    where k = number of units active per condition for that feature.

    Parameters
    ----------
    units : dict {feature: np.ndarray (n_cond, n_values)}
        Output of design_to_units().

    Returns
    -------
    dict {feature: np.ndarray (n_cond, n_cond)}
        Symmetric RSA similarity matrix per feature.
    """
    rsa = {}
    for col, oh in units.items():
        k = oh.sum(axis=1, keepdims=True)              # active units per row
        oh_norm = oh / np.sqrt(np.maximum(k, 1))
        rsa[col] = (oh_norm @ oh_norm.T).astype(np.float32)
    return rsa


def rsa_to_embedding_init(rsa_matrix: np.ndarray, emb_dim: int) -> np.ndarray:
    """Convert a theoretical RSA model matrix to (n, emb_dim) ECin init weights.

    Eigendecomposition (MDS-style): each row = projection of one condition
    onto the top-emb_dim principal axes of the RSA kernel.
    Used to initialise T_FeatureEmbedding so that ECin input geometry already
    reflects the target representational structure at trial onset.

    Returns
    -------
    np.ndarray, shape (n, emb_dim), dtype float32
        Top eigenvectors scaled by sqrt(eigenvalue). Zero-padded if emb_dim > n.
    """
    n = rsa_matrix.shape[0]
    vals, vecs = np.linalg.eigh(rsa_matrix)
    idx        = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    vals       = np.maximum(vals, 0.0)

    k = min(emb_dim, n)
    W = vecs[:, :k] * np.sqrt(vals[:k])

    if emb_dim > n:
        W = np.concatenate([W, np.zeros((n, emb_dim - n))], axis=1)
    return W.astype(np.float32)


# =========================================================================
# T_WeatherEnv / T_WeatherDataset: WEATHER PREDICTION TASK
# =========================================================================
# [Role]:
#   Frank (2005) Weather Prediction task. 4 probabilistic cues (A–D) predict
#   rain or sun. Each trial: one of 14 multi-cue patterns; outcome sampled
#   stochastically from P(rain | cues) = mean of active cue validities.
#
# [Cue validities — Frank (2005) p.62, Fig. 4]:
#   Cue A: P(rain) = 0.41  (weak)
#   Cue B: P(rain) = 0.59  (strong)
#   Cue C: P(rain) = 0.41  (weak)
#   Cue D: P(rain) = 0.59  (strong)
#
# [14 patterns]:
#   All non-empty subsets of {A, B, C, D} with 1–3 active cues:
#     size 1: 4 patterns  (A, B, C, D)
#     size 2: 6 patterns  (AB, AC, AD, BC, BD, CD)
#     size 3: 4 patterns  (ABC, ABD, ACD, BCD)
#   Total = C(4,1) + C(4,2) + C(4,3) = 4 + 6 + 4 = 14  ✓
#   Patterns are sampled uniformly at random each trial.
#
# [ECin/ECout mapping]:
#   cue_pattern → ECin binary vector (n_cues,)
#   outcome     → ECout target scalar (0=sun, 1=rain)
#
# [Why not gymnasium API]:
#   Gymnasium adds observation_space / action_space / (obs, info) wrapping
#   that is unused by the CHL training loop. T_* classes keep the same
#   simple step() convention as T_PairEnv and T_CommunityGraphEnv.

class T_WeatherEnv:
    """Weather Prediction task environment (Frank 2005 Fig. 4).

    14 multi-cue patterns (all non-empty subsets of n_cues with 1–max_cues
    active). Each step returns (cue_pattern, outcome); pattern sampled
    uniformly; outcome sampled from P(rain) = mean active cue validity.
    """

    def __init__(
        self,
        n_cues: int = 4,
        cue_validities: Optional[np.ndarray] = None,
        max_cues: int = 3,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        n_cues : int
            Number of cues. Frank (2005): 4 (A–D).
        cue_validities : ndarray (n_cues,), optional
            P(rain) for each cue. Default: [0.41, 0.59, 0.41, 0.59]
            (Frank 2005 p.62 Fig. 4).
        max_cues : int
            Maximum number of simultaneously active cues per trial. Default 3.
            Together with min=1 gives the 14 standard patterns.
        seed : int, optional
            Random seed.
        """
        self.n_cues = n_cues
        if cue_validities is None:
            # Frank (2005) p.62, Fig. 4: weak (0.41) and strong (0.59) cues
            cue_validities = np.array([0.41, 0.59, 0.41, 0.59])
        self.cue_validities = np.asarray(cue_validities, dtype=np.float32)
        self.max_cues = max_cues
        self.rng = np.random.default_rng(seed)

        # Pre-enumerate all 14 patterns (size 1..max_cues)
        self.patterns: List[np.ndarray] = self._build_patterns()
        self.n_patterns = len(self.patterns)
        self._pattern_idx: int = 0

    def _build_patterns(self) -> List[np.ndarray]:
        patterns = []
        for size in range(1, self.max_cues + 1):
            for combo in itertools.combinations(range(self.n_cues), size):
                v = np.zeros(self.n_cues, dtype=np.float32)
                v[list(combo)] = 1.0
                patterns.append(v)
        return patterns

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Sample first trial pattern; return cue_pattern."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._pattern_idx = int(self.rng.integers(self.n_patterns))
        return self.patterns[self._pattern_idx].copy()

    def step(self) -> Tuple[np.ndarray, int]:
        """Return (cue_pattern, outcome); advance to next (randomly sampled) trial.

        Returns
        -------
        cue_pattern : ndarray (n_cues,)
            Binary vector of active cues for this trial (ECin input).
        outcome : int
            0 = sun, 1 = rain.
            Sampled from P(rain) = mean(cue_validities[active_cues]).
        """
        pattern = self.patterns[self._pattern_idx].copy()
        active  = np.where(pattern)[0]
        p_rain  = float(np.mean(self.cue_validities[active]))
        outcome = int(self.rng.random() < p_rain)
        self._pattern_idx = int(self.rng.integers(self.n_patterns))
        return pattern, outcome

    @property
    def current_pattern(self) -> np.ndarray:
        return self.patterns[self._pattern_idx].copy()

    @property
    def pattern_p_rain(self) -> float:
        """P(rain) for the current pattern."""
        active = np.where(self.patterns[self._pattern_idx])[0]
        return float(np.mean(self.cue_validities[active]))


class T_WeatherDataset(Dataset):
    """Pre-generated weather prediction trial sequence.

    Frank (2005): 400 trials total. Each trial: binary cue pattern → outcome.
    """

    def __init__(
        self,
        n_trials: int = 400,
        n_cues: int = 4,
        cue_validities: Optional[np.ndarray] = None,
        max_cues: int = 3,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        n_trials : int
            Number of trials. Frank (2005) Fig. 7: 400 total. Default 400.
        n_cues : int
            Number of cues. Frank (2005): 4.
        cue_validities : ndarray (n_cues,), optional
            P(rain) per cue. Default: [0.41, 0.59, 0.41, 0.59]
            (Frank 2005 p.62 Fig. 4).
        max_cues : int
            Max active cues per trial. Default 3.
        device : str
            PyTorch device.
        seed : int, optional
            Random seed.
        """
        self.n_trials = n_trials
        self.n_cues   = n_cues
        self.device   = device
        if cue_validities is None:
            cue_validities = np.array([0.41, 0.59, 0.41, 0.59])
        self.cue_validities = np.asarray(cue_validities, dtype=np.float32)

        self._generate(cue_validities, max_cues, seed)

    def _generate(
        self,
        cue_validities: np.ndarray,
        max_cues: int,
        seed: Optional[int],
    ) -> None:
        env = T_WeatherEnv(
            n_cues=self.n_cues,
            cue_validities=cue_validities,
            max_cues=max_cues,
            seed=seed,
        )
        env.reset()

        patterns, outcomes, p_rains, pattern_idxs = [], [], [], []
        for _ in range(self.n_trials):
            idx = env._pattern_idx
            pattern, outcome = env.step()
            active = np.where(pattern)[0]
            p_rain = float(np.mean(cue_validities[active]))
            patterns.append(pattern)
            outcomes.append(outcome)
            p_rains.append(p_rain)
            pattern_idxs.append(idx)

        self._cue_patterns  = torch.tensor(
            np.array(patterns), dtype=torch.float32, device=self.device
        )
        self._outcomes      = torch.tensor(outcomes,      dtype=torch.long,  device=self.device)
        self._p_rains       = torch.tensor(p_rains,       dtype=torch.float32, device=self.device)
        self._pattern_idxs  = torch.tensor(pattern_idxs, dtype=torch.long,  device=self.device)

    def __len__(self) -> int:
        return self.n_trials

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one trial.

        Returns
        -------
        dict with keys:
          'cue_pattern'  : FloatTensor (n_cues,) — binary active-cue vector (ECin input)
          'outcome'      : LongTensor scalar      — 0=sun, 1=rain (ECout target)
          'p_rain'       : FloatTensor scalar     — theoretical P(rain) for this pattern
          'pattern_idx'  : LongTensor scalar      — pattern index (0–n_patterns-1)
        """
        return {
            'cue_pattern': self._cue_patterns[idx],
            'outcome':     self._outcomes[idx],
            'p_rain':      self._p_rains[idx],
            'pattern_idx': self._pattern_idxs[idx],
        }


# =========================================================================
# T_FeatureEmbedding
# =========================================================================
# Embedding table for one task-feature dimension (e.g. RULE, STIM, RESP).
# Optionally initialised from an RSA matrix so that conditions sharing a
# feature value start with similar representations.
# Ported from EmbeddingRNN/src/layer.py (FeatureEmbedding).

class T_FeatureEmbedding(nn.Module):
    """Embedding table for one task-feature dimension.

    Parameters
    ----------
    n_cond : int
        Number of conditions (vocabulary size of this embedding).
    emb_dim : int
        Embedding dimension.
    rsa_matrix : np.ndarray (n_cond, n_cond) or None
        RSA kernel for weight initialisation (from units_to_rsa).
        None → default nn.Embedding random init.
    freeze_on : bool
        If True, weights are not updated during training.
    """

    def __init__(
        self,
        n_cond: int,
        emb_dim: int,
        rsa_matrix: np.ndarray = None,
        freeze_on: bool = False,
    ):
        super().__init__()
        self.emb = nn.Embedding(n_cond, emb_dim)

        if rsa_matrix is not None:
            W = rsa_to_embedding_init(rsa_matrix, emb_dim)
            self.emb.weight = nn.Parameter(
                torch.tensor(W, dtype=torch.float32),
                requires_grad=not freeze_on,
            )
        elif freeze_on:
            self.emb.weight.requires_grad_(False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx : LongTensor (B,) → FloatTensor (B, emb_dim)"""
        return self.emb(idx)


# =========================================================================
# T_TaskEmbedding
# =========================================================================
# Combines per-feature T_FeatureEmbedding objects into one ECin input vector.
#
# Usage for K&M RuleAction_4rules task (ARCHITECTURE_ENG.md §10):
#   design = pd.read_csv('.../RuleAction_4rules/Task_Design.txt', sep='\t')
#   units  = design_to_units(design)
#   rsa    = units_to_rsa(units)
#   emb    = T_TaskEmbedding(design, units, emb_dim=8, emb_similarity=rsa)
#   ecin   = emb(cond_idx)    # (B, total_emb_dim) → feed to L_ECin
#
# Ported from EmbeddingRNN/src/layer.py (TaskEmbedding).

class T_TaskEmbedding(nn.Module):
    """Combine per-feature embeddings into one ECin input vector.

    Feature columns (all design columns except 'COND') each get a
    T_FeatureEmbedding. 'CONJ' is handled via compositional_on:
      compositional_on=True  → CONJ = sum of base feature embeddings
      compositional_on=False → CONJ has its own independent lookup table

    Output size:
      shared_space_on=False → total_emb_dim = sum of per-feature emb_dims
      shared_space_on=True  → total_emb_dim = emb_dim (projected + summed)

    Parameters
    ----------
    design : pd.DataFrame
        Task_Design.txt; rows=conditions, columns=feature names.
    units : dict {col: np.ndarray (n_cond, n_values)}
        Output of design_to_units().
    emb_dim : int or dict {feature: int}
        Embedding dimension — global int or per-feature dict.
    emb_similarity : dict {feature: np.ndarray (n_cond, n_cond)} or None
        RSA matrices for weight initialisation (from units_to_rsa()).
    shared_space_on : bool
        True → project each feature through shared linear, then sum.
    gate_on : bool
        True → multiply output by sigmoid(gate_net(t)); pass t in forward().
    freeze_on : bool or dict {feature: bool}
        Freeze embedding weights globally or per feature.
    compositional_on : bool
        True → CONJ = sum(base embeddings). False → CONJ independent table.
    """

    _SKIP_COLS = {'COND'}

    def __init__(
        self,
        design: pd.DataFrame,
        units: dict,
        emb_dim,
        emb_similarity=None,
        shared_space_on: bool = False,
        gate_on: bool = False,
        freeze_on=False,
        compositional_on: bool = False,
    ):
        super().__init__()
        self.compositional_on = compositional_on
        self.shared_space_on  = shared_space_on
        self.gate_on          = gate_on

        n_cond       = len(design)
        feature_cols = [c for c in design.columns if c not in self._SKIP_COLS]
        base_cols    = [c for c in feature_cols if c != 'CONJ']
        self.base_cols    = base_cols
        self.feature_cols = feature_cols

        dim_for    = (lambda col: emb_dim[col])   if isinstance(emb_dim, dict)   else (lambda col: emb_dim)
        freeze_for = (lambda col: freeze_on[col]) if isinstance(freeze_on, dict) else (lambda col: freeze_on)
        rsa_for    = lambda col: (emb_similarity or {}).get(col, None)

        self._ref_dim = (
            dim_for(base_cols[0]) if base_cols
            else (emb_dim if isinstance(emb_dim, int) else next(iter(emb_dim.values())))
        )

        self.embeddings = nn.ModuleDict({
            col: T_FeatureEmbedding(
                n_cond=n_cond,
                emb_dim=dim_for(col),
                rsa_matrix=rsa_for(col),
                freeze_on=freeze_for(col),
            )
            for col in base_cols
        })

        if not compositional_on and 'CONJ' in feature_cols:
            self.conj_emb = T_FeatureEmbedding(
                n_cond=n_cond,
                emb_dim=dim_for('CONJ'),
                rsa_matrix=rsa_for('CONJ'),
                freeze_on=freeze_for('CONJ'),
            )
        else:
            self.conj_emb = None

        self.shared_proj = nn.Linear(self._ref_dim, self._ref_dim, bias=False) if shared_space_on else None
        self.gate_net    = nn.Linear(1, self._ref_dim) if gate_on else None

        if shared_space_on:
            self.total_emb_dim = self._ref_dim
        else:
            dims = [dim_for(c) for c in base_cols]
            if compositional_on:
                dims.append(self._ref_dim)
            elif 'CONJ' in feature_cols:
                dims.append(dim_for('CONJ'))
            self.total_emb_dim = sum(dims)

    def forward(
        self,
        cond_idx: torch.Tensor,
        t: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        cond_idx : LongTensor (B,)      0-indexed condition
        t        : FloatTensor (B, 1) or None   timestep (for gate_on)
        Returns  : FloatTensor (B, total_emb_dim)
        """
        embs = [self.embeddings[col](cond_idx) for col in self.base_cols]

        if self.compositional_on:
            embs.append(sum(embs))
        elif self.conj_emb is not None:
            embs.append(self.conj_emb(cond_idx))

        if self.shared_space_on:
            out = sum(self.shared_proj(e) for e in embs)
        else:
            out = torch.cat(embs, dim=-1)

        if self.gate_on and t is not None:
            gate = torch.sigmoid(self.gate_net(t.float()))
            out  = out * gate

        return out
