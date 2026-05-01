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
from typing import Dict, List, Optional, Tuple

import torch
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
