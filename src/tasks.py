"""
Schapiro et al. (2017) community graph statistical learning task.

Task Design
-----------
Items       : 15 items organized into 5 communities × 3 items each
Structure   : Community graph — within-community transitions are more frequent
              than between-community (bottleneck) transitions
Input       : Current item index → L_ECin one-hot pattern
Target      : Next item index → L_ECout clamped pattern (plus phase)
Learning    : CHL (minus phase = prediction; plus phase = correction)

Community Graph
---------------
Within-community transitions: high probability (items A-B, B-C, A-C)
Between-community transitions: low probability (only at bottleneck nodes)
Random walk follows this transition structure during training.

Expected Results (Schapiro 2017)
---------------------------------
- CA1 representations cluster by community after learning
- MSP develops graded community structure (overlapping within community)
- TSP retains distinct episode-level representations
- Pattern completion: partial cue → CA3 completes → correct CA1 output

Usage
-----
    from src.tasks import CommunityGraphEnv, CommunityGraphDataset

    env = CommunityGraphEnv(n_communities=5, items_per_community=3)
    item_idx, next_idx = env.reset(seed=42), env.step()

    dataset = CommunityGraphDataset(n_steps=10000)
    step = dataset[0]  # {'item': tensor, 'next_item': tensor, 'community': int}

References
----------
Schapiro, A. C., Turk-Browne, N. B., Botvinick, M. M., & Norman, K. A. (2017).
    Complementary learning systems within the hippocampus: a neural network
    modelling approach to reconciling episodic memory with statistical learning.
    Phil. Trans. R. Soc. B, 372, 20160049.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset


# =========================================================================
# CommunityGraphEnv: COMMUNITY GRAPH RANDOM WALK ENVIRONMENT
# =========================================================================
# [Role]:
#   Generates a random walk over the community graph.
#   Each step yields (current_item, next_item) pair for CHL training.
#
# [Graph structure]:
#   n_communities communities, each containing items_per_community items.
#   Within-community transitions: uniform over other items in same community.
#   Between-community transitions: only at designated bottleneck nodes.
#   Schapiro (2017) Fig. 1: 15-node graph, 5 communities × 3 items.
#
# [Training procedure]:
#   1. Start at a random item
#   2. Sample next_item from transition probabilities
#   3. Present current_item to ECin (input)
#   4. Present next_item to ECout (target, plus phase)
#   5. Run CHL; update weights
#   6. Move to next_item → repeat
#
# [Key parameter]:
#   p_within : probability of within-community transition at each step.
#              Schapiro (2017): p_within is high (exact value from original code).
#              Between-community transition occurs with probability (1 - p_within).
#
# [Notes]:
#   - Schapiro (2017) §2.4: "random walk with higher within-community probability"
#   - The community structure is what MSP learns over many exposures
#   - TSP encodes individual transition episodes regardless of structure

class CommunityGraphEnv:
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
            Probability of within-community transition. Schapiro (2017): derived
            from graph structure (all within-community edges equally likely).
            If None, use the graph's natural transition probabilities.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.n_communities = n_communities
        self.items_per_community = items_per_community
        self.n_items = n_communities * items_per_community
        self.p_within = p_within
        self.rng = np.random.default_rng(seed)

        # Community membership: item i belongs to community i // items_per_community
        self.community = np.array([
            i // items_per_community for i in range(self.n_items)
        ])

        # Build transition matrix from community graph structure
        # Schapiro (2017) Fig. 1: specific graph topology with bottleneck nodes
        self._transition_matrix = None  # built in _build_graph()
        self._build_graph()

        self._current_item: int = 0

    def _build_graph(self) -> None:
        """Build transition probability matrix from community structure.

        Within-community edges are all equal weight.
        Between-community edges only at bottleneck nodes (one per community pair).
        Schapiro (2017) Fig. 1: 15-node graph with specific connectivity.
        """
        raise NotImplementedError("Step 6: implement CommunityGraphEnv._build_graph()")

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
        raise NotImplementedError("Step 6: implement CommunityGraphEnv.reset()")

    def step(self) -> Tuple[int, int]:
        """Sample one transition: current_item → next_item.

        Returns
        -------
        current_item : int
            Index of the current item (input to ECin).
        next_item : int
            Index of the next item (target for ECout plus phase).
        """
        raise NotImplementedError("Step 6: implement CommunityGraphEnv.step()")

    @property
    def current_item(self) -> int:
        return self._current_item


# =========================================================================
# CommunityGraphDataset: PYTORCH DATASET WRAPPER
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
# [Usage]:
#   dataset = CommunityGraphDataset(n_steps=10000, seed=42)
#   step = dataset[0]
#   # step['item'], step['next_item'], step['item_onehot'], step['target_onehot']
#
# [Notes]:
#   - Schapiro (2017) §2.4: model trained for many trials (exact count in paper)
#   - n_steps should be large enough to expose all transitions many times
#   - The same dataset can be used for both MSP and TSP learning

class CommunityGraphDataset(Dataset):
    """Pre-generated community graph random walk for CHL training.

    Schapiro (2017) §2.4: training sequence of (current_item, next_item) pairs.
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
            Schapiro (2017): large enough for convergence (~10k+).
        n_communities : int
            Number of communities. Schapiro (2017) Fig. 1: 5.
        items_per_community : int
            Items per community. Schapiro (2017) Fig. 1: 3.
        p_within : float, optional
            Within-community transition probability. See CommunityGraphEnv.
        device : str
            PyTorch device ('cpu' or 'cuda').
        seed : int, optional
            Random seed.
        """
        self.n_steps = n_steps
        self.n_items = n_communities * items_per_community
        self.device = device

        self._items: torch.Tensor = None
        self._next_items: torch.Tensor = None
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
        raise NotImplementedError("Step 6: implement CommunityGraphDataset._generate()")

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
        raise NotImplementedError("Step 6: implement CommunityGraphDataset.__getitem__()")


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
