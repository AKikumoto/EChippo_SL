"""
Schapiro et al. (2017) statistical learning tasks.

Tasks
-----
Pair structure (§3.a):
    8 items in 4 fixed pairs (AB, CD, EF, GH); pair order randomized.
    T_PairEnv / T_PairDataset.

Community graph (§3.b):
    15 items in 5 communities × 3 items; random walk on graph.
    T_CommunityGraphEnv / T_CommunityGraphDataset.

Both tasks produce (item, next_item) pairs for CHL training:
    item       → ECin clamped pattern (current stimulus)
    next_item  → ECout clamped pattern (plus-phase teaching signal)

Usage
-----
    from src.tasks import T_CommunityGraphEnv, T_CommunityGraphDataset
    from src.tasks import T_PairEnv, T_PairDataset

    env  = T_CommunityGraphEnv(n_communities=5, items_per_community=3)
    item = env.reset(seed=42)
    current, next_item = env.step()

    dataset = T_CommunityGraphDataset(n_steps=10000, seed=42)
    step    = dataset[0]   # dict with item, next_item, community, one-hots

References
----------
Schapiro, A. C., Turk-Browne, N. B., Botvinick, M. M., & Norman, K. A. (2017).
    Complementary learning systems within the hippocampus: a neural network
    modelling approach to reconciling episodic memory with statistical learning.
    Phil. Trans. R. Soc. B, 372, 20160049.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

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

    8 items in 4 fixed pairs. Each step returns (A, B) for one pair;
    pair order is randomized with no back-to-back repetitions.
    """

    def __init__(self, n_pairs: int = 4, seed: Optional[int] = None):
        """
        Parameters
        ----------
        n_pairs : int
            Number of pairs. Schapiro (2017) §3.a: 4 (AB/CD/EF/GH).
        seed : int, optional
            Random seed.
        """
        self.n_pairs = n_pairs
        self.n_items = n_pairs * 2
        # Pair i: items (2i, 2i+1); Schapiro (2017) §3.a: AB/CD/EF/GH
        self.pairs   = [(2 * i, 2 * i + 1) for i in range(n_pairs)]
        self.rng     = np.random.default_rng(seed)
        self._pair_idx = 0

    def reset(self, seed: Optional[int] = None) -> int:
        """Start at a random pair; return its first item index."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._pair_idx = int(self.rng.integers(self.n_pairs))
        return self.pairs[self._pair_idx][0]

    def step(self) -> Tuple[int, int]:
        """Return (A, B) for current pair; advance to a different pair.

        Returns
        -------
        current_item : int
            First item of the pair (input to ECin).
        next_item : int
            Second item of the pair (ECout plus-phase target).
        """
        pair    = self.pairs[self._pair_idx]
        current = pair[0]
        target  = pair[1]
        # Next pair: exclude current to prevent back-to-back repetition
        # Schapiro (2017) §3.a: no consecutive same-pair presentations
        candidates     = [i for i in range(self.n_pairs) if i != self._pair_idx]
        self._pair_idx = int(self.rng.choice(candidates))
        return current, target

    @property
    def current_item(self) -> int:
        return self.pairs[self._pair_idx][0]


class T_PairDataset(Dataset):
    """Pre-generated pair task sequence for CHL training.

    Schapiro (2017) §3.a: 4 fixed pairs (AB/CD/EF/GH); random pair order.
    """

    def __init__(
        self,
        n_steps: int = 800,
        n_pairs: int = 4,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        n_steps : int
            Number of transitions to generate.
            Schapiro (2017) §3.a: 80 inputs/epoch; default 800 = 10 epochs.
        n_pairs : int
            Number of pairs. Schapiro (2017) §3.a: 4.
        device : str
            PyTorch device.
        seed : int, optional
            Random seed.
        """
        self.n_steps = n_steps
        self.n_items = n_pairs * 2
        self.device  = device

        self._items      : torch.Tensor = None
        self._next_items : torch.Tensor = None
        self._pair_labels: torch.Tensor = None

        self._generate(n_pairs, seed)

    def _generate(self, n_pairs: int, seed: Optional[int]) -> None:
        """Run pair environment and store all (item, next_item) sequences."""
        env = T_PairEnv(n_pairs=n_pairs, seed=seed)
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
        p_within: float = None,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        n_communities : int
            Number of communities. Schapiro (2017) Fig. 1: 5.
        items_per_community : int
            Items per community. Schapiro (2017) Fig. 1: 3.
        p_within : float, optional
            Unused — transition probabilities are derived from graph topology
            (uniform over neighbors). Kept for API compatibility.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.n_communities      = n_communities
        self.items_per_community = items_per_community
        self.n_items            = n_communities * items_per_community
        self.rng                = np.random.default_rng(seed)

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
#   Schapiro (2017) §3.b: 60 inputs/epoch, 10 epochs → n_steps = 600 minimum.

class T_CommunityGraphDataset(Dataset):
    """Pre-generated community graph random walk for CHL training.

    Schapiro (2017) §3.b: training sequence of (current_item, next_item) pairs.
    """

    def __init__(
        self,
        n_steps: int = 10000,
        n_communities: int = 5,
        items_per_community: int = 3,
        p_within: float = None,
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
        p_within : float, optional
            Passed through to T_CommunityGraphEnv (unused; see env docstring).
        device : str
            PyTorch device ('cpu' or 'cuda').
        seed : int, optional
            Random seed.
        """
        self.n_steps = n_steps
        self.n_items = n_communities * items_per_community
        self.device  = device

        self._items      : torch.Tensor = None
        self._next_items : torch.Tensor = None
        self._communities: torch.Tensor = None

        self._generate(n_communities, items_per_community, p_within, seed)

    def _generate(
        self,
        n_communities: int,
        items_per_community: int,
        p_within: Optional[float],
        seed: Optional[int],
    ) -> None:
        """Run random walk and store all (item, next_item) transitions."""
        env = T_CommunityGraphEnv(
            n_communities=n_communities,
            items_per_community=items_per_community,
            p_within=p_within,
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
# TASK DESIGN HELPERS
# =========================================================================
# Functions for loading task design tables and constructing RSA matrices.
# Used by T_FeatureEmbedding / T_TaskEmbedding for ECin initialisation.
#
# design_to_units : Task_Design.txt → per-feature one-hot matrices
# units_to_rsa    : one-hot matrices → per-feature RSA similarity matrices
# rsa_to_embedding_init : RSA matrix → MDS-style initial embedding weights
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
    """Construct per-feature RSA similarity matrices from one-hot encodings.

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
    """Convert an (n, n) RSA similarity matrix to (n, emb_dim) initial weights.

    Eigendecomposition (MDS-style): each row = projection of one condition
    onto the top-emb_dim principal axes of the RSA kernel.

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
