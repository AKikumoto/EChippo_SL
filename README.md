# EChipp_SL — Hippocampal Statistical Learning Model

PyTorch reimplementation of Schapiro et al. (2017) hippocampal statistical learning (hip-sl) model, originally written in C++ emergent / Go. Extended with a temporally-structured successor representation SR(t) for the EfAb R21 grant project on causal mechanisms of efficiency and abstraction in context-dependent action selection.

## Project Goal

The ultimate aims are to:

1. **Understand every line of code** — primary goal; the model is built step by step with paper citations at every equation
2. **Reproduce Schapiro et al. (2017)** — MSP/TSP dissociation, community structure learning, pattern completion vs. pattern separation
3. **Implement SR(t) extension** — temporally-structured successor representation over conjunctive task space (EfAb computational framework)
4. **Generate EfAb predictions** — n_dynamic/n_stable convergence, power-law improvement, overnight abstraction as SR(t) compression

> **Foundational Principle:**
> Before writing code, always open and read the corresponding paper section.
> Every numerical value is marked with its source (paper name, page, equation number).
> Only after you can explain "why this value?" in your own words, move to coding.

---

## Target Model: Schapiro et al. (2017)

**Source:** Schapiro, A. C., Turk-Browne, N. B., Botvinick, M. M., & Norman, K. A. (2017). Complementary learning systems within the hippocampus: a neural network modelling approach to reconciling episodic memory with statistical learning. *Philosophical Transactions of the Royal Society B*, 372, 20160049.

**Original code:** https://github.com/schapirolab/hip-sl (Go/emergent reimplementation)

**Core idea:** Hippocampus contains two complementary pathways that serve different learning functions:

| Pathway | Route | Function |
|---------|-------|----------|
| **MSP** (Monosynaptic Pathway) | ECin → CA1 | Statistical learning of transition structure |
| **TSP** (Trisynaptic Pathway) | ECin → DG → CA3 → CA1 | Episodic binding of individual events |

MSP learns slowly via Hebbian plasticity and develops smooth, overlapping representations of items that co-occur — capturing statistical regularities. TSP uses DG pattern separation and CA3 pattern completion to bind unique episodes without interference.

---

## Circuit Architecture

```
ECin ──────────────────────────────► CA1 ──► ECout
 │                                    ▲         (MSP: direct, slow learning)
 │                                    │
 └──► DG (sparse, pattern separate) ──►  CA3 ──► CA1
       (TSP: episodic binding, fast)
```

| Layer | Description | Role |
|-------|-------------|------|
| **ECin** | Entorhinal cortex input | Pattern of activity for current item |
| **DG** | Dentate gyrus | Pattern separation (~1% sparsity); episodic uniqueness |
| **CA3** | CA3 field | Pattern completion; recurrent attractor dynamics |
| **CA1** | CA1 field | Convergence of MSP (ECin) and TSP (CA3) signals |
| **ECout** | Entorhinal cortex output | Reconstruction target; plus-phase teaching signal |

### CHL Learning (Contrastive Hebbian Learning)

Two phases per trial:
- **Minus phase:** ECin active, CA1 driven by prediction (no ECout teaching signal)
- **Plus phase:** ECin + ECout both active; CA1 driven toward correct output

Weight update: `ΔW ∝ y_plus - y_minus` (Schapiro 2017; O'Reilly & Munakata 2000)

### Key parameter differences from original emergent model (from Go reimplementation)
- KWTA inhibition replaced by FFFB inhibition in Go version
- MSP learning rate: 0.02 → 0.05; TSP learning rate: 0.2 → 0.4
- PyTorch implementation uses Euler integration for stateful settling (no ODE solver)

---

## Statistical Learning Task

Community graph structure (Schapiro 2017 Fig. 1):
- Items organized into communities (e.g., 3 communities × 3 items = 9 items)
- Within-community transitions are more frequent
- Between-community transitions happen only at bottleneck nodes
- After training: CA1 develops separate community-level representations
- MSP learns the full transition probability matrix
- TSP retains individual episode memories without blending

---

## EfAb Extension: SR(t) over Conjunctive Subspace

From EfAb_grant_v2.md (April 2026):

**SR(t) key idea:** Extend the standard successor representation matrix M(s,s') to M(s,s',t) where t indexes within-trial timestep. EC represents the eigenspace of M(s,s',t). Conjunctive subspace = high-eigenvalue dimensions of M(s,s',t).

**n_dynamic / n_stable framework:**
- **n_dynamic**: the dynamic portion of the within-trial trajectory (cue → S → R) — variable across trials early in learning, stabilizes with practice
- **n_stable**: the stable endpoint state (post-response conjunctive representation) — the target that n_dynamic converges to
- SR(t) = the probability that n_dynamic reaches n_stable by time t within a trial

**Behavioral signatures:**
- Power-law improvement = n_dynamic/n_stable convergence speeding up with practice
- Overnight abstraction = n_stable becomes cue-invariant (cue identity compressed to low eigenvectors)

---

## Implementation Status

| Step | Component | Status |
|------|-----------|--------|
| 1 | `F_nxx1`, `F_kWTA`, `F_fffb` (util.py) | not started |
| 2 | `L_ECin`, `L_ECout` (layer.py) | not started |
| 3 | `L_DG` — pattern separation (layer.py) | not started |
| 4 | `L_CA3` — attractor dynamics (layer.py) | not started |
| 5 | `L_CA1` — MSP + TSP convergence (layer.py) | not started |
| 6 | `CommunityGraphEnv`, `CommunityGraphDataset` (tasks.py) | not started |
| 7 | `M_HipSL` assembly + CHL training loop (model.py) | not started |
| 8 | Reproduce Schapiro 2017 results | not started |
| 9 | `M_HipSL_SR`: SR(t) extension (model.py) | not started |
| 10 | EfAb behavioral readouts | not started |

---

## Contents

```
EChipp_SL/
├── src/
│   ├── util.py           F_nxx1, F_kWTA, F_fffb
│   ├── layer.py          L_ECin, L_ECout, L_DG, L_CA3, L_CA1
│   ├── model.py          M_HipSL, M_HipSL_SR
│   └── tasks.py          CommunityGraphEnv, CommunityGraphDataset
├── notebook/
│   ├── test_nxx1.ipynb           Step 1
│   ├── test_layers.ipynb         Steps 2–5
│   ├── test_task.ipynb           Step 6
│   ├── test_full_model.ipynb     Steps 7–8
│   └── test_sr.ipynb             Steps 9–10
├── simulations/          (empty — not yet started)
├── scripts/              (empty — not yet started)
├── trained_models/
├── visualizations/
├── manuscript/
├── config/
│   ├── ARCHITECTURE.md   Master document (equations, parameters, roadmap)
│   └── requirements.txt
├── CLAUDE.md
└── README.md
```

---

## Quick Start

```bash
# Create virtual environment
uv venv .venv
source .venv/bin/activate
uv pip install torch numpy matplotlib seaborn pandas

# Open notebooks step by step
# notebook/test_nxx1.ipynb  ← start here (Step 1)
```

---

## Key References

- **Schapiro, A. C. et al. (2017).** Complementary learning systems within the hippocampus. *Phil. Trans. R. Soc. B*, 372, 20160049.
- **O'Reilly, R. C. & Munakata, Y. (2000).** *Computational Explorations in Cognitive Neuroscience* — Leabra framework (nxx1, kWTA, CHL).
- **Stachenfeld, K. L., Botvinick, M. M., & Gershman, S. J. (2017).** The hippocampus as a predictive map. *Nature Neuroscience*, 20, 1643–1653.
- **Momennejad, I. (2020).** Learning structures: predictive representations, replay, and generalization. *Current Opinion in Behavioral Sciences*, 32, 155–166.
- **Garvert, M. M., Dolan, R. J., & Behrens, T. E. J. (2017).** A map of abstract relational knowledge in the human hippocampal–entorhinal cortex. *eLife*, 6, e17086.
- **Kikumoto, A. et al. (2025).** Conjunctive representational trajectories predict power-law improvement and overnight abstraction. *Cerebral Cortex*.
- **Mylonas, D. et al. (2024).** Hippocampus is necessary for micro-offline gains. *J. Neurosci.*

---

## See Also

- `config/ARCHITECTURE.md` — Full implementation specification with equations and paper citations
- `EfAb_grant_v2.md` — Grant document motivating the SR(t) extension
