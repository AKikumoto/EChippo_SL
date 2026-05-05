# EChipp_SL — Hippocampal Statistical Learning Model

PyTorch reimplementation of Schapiro et al. (2017) hippocampal statistical learning model, originally written in C++ emergent / Go. Extended with neural state(t) analysis over conjunctive task subspaces.

---

Schapiro, A. C., Turk-Browne, N. B., Botvinick, M. M., & Norman, K. A. (2017). Complementary learning systems within the hippocampus: a neural network modelling approach to reconciling episodic memory with statistical learning. *Philosophical Transactions of the Royal Society B*, 372, 20160049. ([original Go code](https://github.com/schapirolab/hip-sl))

Hippocampus contains two complementary pathways that serve different learning functions:

| Pathway | Route | Function |
|---------|-------|----------|
| **MSP** (Monosynaptic) | ECin → CA1 | Statistical learning of transition structure |
| **TSP** (Trisynaptic) | ECin → DG → CA3 → CA1 | Episodic binding of individual events |

MSP learns slowly via Hebbian plasticity, developing smooth overlapping representations that capture statistical regularities. TSP uses DG pattern separation and CA3 pattern completion to bind unique episodes without interference.

---

## Circuit Architecture

```
ECin ─────────────────────────────────────────────── CA1 ──► ECout
 │                        (MSP: direct, lr=0.05)    ▲
 ├──► DG ──(5% mossy fiber)──┐                      │
 │    (pattern sep., ~1%)     ├──► CA3 (↺) ──────────┘
 └────────(25%, direct)───────┘   (pattern comp., ~10%)  (TSP, lr=0.4)
           Schapiro 2017 §2.a.iii
```

<!-- inline neutral card — explicit colors, no CSS variables -->
<div style="font-size:13px;color:#1a1a1a;padding:8px 0;font-family:sans-serif">
<svg width="100%" viewBox="0 0 680 205" role="img" style="margin-bottom:12px">
  <defs>
    <marker id="at2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#1D9E75" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker>
    <marker id="am2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#534AB7" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker>
    <marker id="ao2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#5F5E5A" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker>
  </defs>
  <rect x="14" y="8" width="14" height="3" rx="1" fill="#1D9E75"/>
  <text x="34" y="13" style="font-size:11px;fill:#085041;font-weight:500">Trisynaptic pathway (TSP) — EC Layer II → DG → CA3 → CA1</text>
  <line x1="14" y1="26" x2="28" y2="26" stroke="#534AB7" stroke-width="1.5" stroke-dasharray="4 2"/>
  <text x="34" y="30" style="font-size:11px;fill:#3C3489;font-weight:500">Monosynaptic pathway (MSP) — EC Layer III → CA1 direct</text>
  <g><rect x="14" y="80" width="80" height="90" rx="8" fill="#E1F5EE" stroke="#085041" stroke-width="0.5"/><text x="54" y="112" text-anchor="middle" dominant-baseline="central" style="font-size:12px;fill:#085041;font-weight:500">ECin</text><text x="54" y="130" text-anchor="middle" style="font-size:10px;fill:#085041">L.II → TSP</text><text x="54" y="145" text-anchor="middle" style="font-size:10px;fill:#085041">L.III → MSP</text></g>
  <g><rect x="152" y="50" width="84" height="60" rx="8" fill="#EAF3DE" stroke="#27500A" stroke-width="0.5"/><text x="194" y="72" text-anchor="middle" dominant-baseline="central" style="font-size:12px;fill:#27500A;font-weight:500">DG</text><text x="194" y="90" text-anchor="middle" style="font-size:10px;fill:#27500A">pattern sep.</text><text x="194" y="104" text-anchor="middle" style="font-size:10px;fill:#27500A">sparse codes</text></g>
  <g><rect x="298" y="50" width="84" height="60" rx="8" fill="#FAEEDA" stroke="#633806" stroke-width="0.5"/><text x="340" y="72" text-anchor="middle" dominant-baseline="central" style="font-size:12px;fill:#633806;font-weight:500">CA3</text><text x="340" y="90" text-anchor="middle" style="font-size:10px;fill:#633806">pattern comp.</text><text x="340" y="104" text-anchor="middle" style="font-size:10px;fill:#633806">recurrent net</text></g>
  <g><rect x="444" y="80" width="84" height="90" rx="8" fill="#EEEDFE" stroke="#26215C" stroke-width="0.5"/><text x="486" y="112" text-anchor="middle" dominant-baseline="central" style="font-size:12px;fill:#26215C;font-weight:500">CA1</text><text x="486" y="130" text-anchor="middle" style="font-size:10px;fill:#26215C">TSP + MSP</text><text x="486" y="145" text-anchor="middle" style="font-size:10px;fill:#26215C">converge</text></g>
  <g><rect x="592" y="80" width="74" height="90" rx="8" fill="#F1EFE8" stroke="#2C2C2A" stroke-width="0.5"/><text x="629" y="112" text-anchor="middle" dominant-baseline="central" style="font-size:12px;fill:#2C2C2A;font-weight:500">ECout</text><text x="629" y="130" text-anchor="middle" style="font-size:10px;fill:#2C2C2A">→ neocortex</text></g>
  <path d="M94 105 Q122 105 122 80 L152 80" fill="none" stroke="#1D9E75" stroke-width="1.5" marker-end="url(#at2)"/>
  <text x="123" y="72" text-anchor="middle" style="font-size:10px;fill:#085041">perforant path</text>
  <line x1="236" y1="80" x2="298" y2="80" stroke="#1D9E75" stroke-width="1.5" marker-end="url(#at2)"/>
  <text x="266" y="72" text-anchor="middle" style="font-size:10px;fill:#085041">mossy fiber</text>
  <path d="M382 80 Q412 80 412 110 L444 110" fill="none" stroke="#1D9E75" stroke-width="1.5" marker-end="url(#at2)"/>
  <text x="415" y="72" text-anchor="middle" style="font-size:10px;fill:#085041">Schaffer coll.</text>
  <path d="M94 118 Q196 148 298 98" fill="none" stroke="#1D9E75" stroke-width="1.2" marker-end="url(#at2)"/>
  <text x="196" y="163" text-anchor="middle" style="font-size:10px;fill:#085041">direct (25%)</text>
  <path d="M94 148 Q268 200 444 148" fill="none" stroke="#534AB7" stroke-width="1.5" stroke-dasharray="5 3" marker-end="url(#am2)"/>
  <text x="268" y="199" text-anchor="middle" style="font-size:10px;fill:#3C3489">temporoammonic path</text>
  <line x1="528" y1="125" x2="592" y2="125" stroke="#5F5E5A" stroke-width="1.5" marker-end="url(#ao2)"/>
  <text x="666" y="205" text-anchor="end" style="font-size:10px;fill:#888">Schapiro et al. 2017</text>
</svg>

| Region | Computational role | Schapiro model parameters |
|--------|-------------------|--------------------------|
| **ECin** | Grid cells (L.II → TSP; L.III → MSP); sole cortical gateway | 15 units; k=2 absolute; moving window curr=1.0 prev=0.9; no learned weights |
| **DG** | Pattern separation; sparse orthogonal codes | ~1% sparsity (k=0.01); ECin→DG 25%; DG→CA3 5% mossy fiber; TSP lr=0.4 |
| **CA3** | Pattern completion; recurrent attractor (Hopfield) | ~10% sparsity (k=0.10); CA3→CA3 recurrent; TSP lr=0.4 |
| **CA1** | TSP + MSP convergence; temporal integration | ~10% sparsity; MSP lr=0.05 (slow); TSP lr=0.4 (fast); Q1: ECin / Q2–Q3: CA3 / Q4: ECout clamp |
| **ECout** | Reconstruction target; plus-phase teaching signal | k=2 absolute; Q4 clamped → ECout→CA1 back-projection |

</div>

---

## Learning Rule

**CHL — Contrastive Hebbian Learning** (O'Reilly & Munakata 2000, Ch. 4; Schapiro 2017 §2.b)

Each trial has two phases. The weight update is the difference between them:

```
ΔW = lr × (ActP ⊗ ActP  −  ActM ⊗ ActM)

ActM  end of Q3 (cycle 75)   — minus phase: network's free prediction
ActP  end of Q4 (cycle 100)  — plus phase:  ECout clamped to correct next item
⊗     outer product (post × pre)
```

| Pathway | Weights updated | lr |
|---------|----------------|-----|
| MSP | ECin → CA1 | 0.05 |
| TSP | ECin → DG, DG → CA3, CA3 → CA3, CA3 → CA1 | 0.4 |
| Output | CA1 → ECout | 0.05 |

MSP is slow (many trials → statistical regularities). TSP is 10× faster (one-shot episode binding). Parameter values from the Go reimplementation; original emergent: MSP 0.02, TSP 0.2.

---

## Contents

```
EChipp_SL/
├── src/
│   ├── util.py                   F_nxx1, F_kWTA
│   ├── layer.py                  L_ECin, L_ECout, L_DG, L_CA3, L_CA1
│   ├── model.py                  M_HipSL  (not yet)
│   ├── tasks.py                  CommunityGraphEnv, RuleActionEnv
│   └── z_task_design_tables/     RSA model matrices (RuleAction, CITask, etc.)
├── notebook/
│   ├── test_nxx1.ipynb           Step 1: activation functions
│   ├── test_RuleAction_4rules.ipynb  K&M task exploration
│   └── test_WPtask.ipynb         WP task exploration
├── config/
│   ├── ARCHITECTURE_ENG.md       Master document (equations, parameters, roadmap)
│   └── requirements.txt
├── manuscript/
│   ├── hippocampal_circuit.html  TUS reference card
│   └── hippocampal_circuit_neutral.html  neutral reference card
├── visualizations/
├── trained_models/
└── README.md
```

---

## Tasks

**Community graph (Schapiro 2017):** 15 items in 5 communities; random walk; within-community transitions more frequent. CA1 develops community-level representations; MSP captures transition statistics; TSP retains episodic detail.

**Rule-action selection (K&M task):** 4 rules × 4 stimuli = 16 conjunctive action contexts (Kikumoto et al. 2025). ECin carries rate-coded rule + stimulus features; CA1 must form context-specific conjunctive representations (RSRCONJ) via Hebbian CHL.

---

## Extension to neural subspaces of task representations

Within-trial neural state(t) trajectory decomposition in CA1:

- **n_stable**: temporally stable representation (post-response conjunction); increases with practice
- **n_dynamic**: time-varying trajectory converging toward n_stable; decreases with practice

Practice signature: n_stable emerges earlier in the trial. Overnight consolidation compresses cue identity into low-variance dimensions, producing cue-invariant RSRCONJ representations (Kikumoto et al. 2025).

---

## Implementation Status

| Step | Component | Status |
|------|-----------|--------|
| 1 | `F_nxx1`, `F_kWTA` (util.py) | done |
| 2 | `L_ECin`, `L_ECout` (layer.py) | done |
| 3 | `L_DG` — pattern separation (layer.py) | not started |
| 4 | `L_CA3` — attractor dynamics (layer.py) | not started |
| 5 | `L_CA1` — MSP + TSP convergence (layer.py) | not started |
| 6 | `CommunityGraphEnv`, `CommunityGraphDataset` (tasks.py) | not started |
| 7 | `M_HipSL` assembly + CHL training loop (model.py) | not started |
| 8 | Reproduce Schapiro 2017 results | not started |
| 9 | `RuleActionEnv`, `RuleActionDataset` — K&M task (tasks.py) | not started |
| 10 | Train M_HipSL on K&M task; CA1 RSA vs. RSRCONJ | not started |

---

## Quick Start

```bash
# CLI scripts
uv venv .venv && source .venv/bin/activate
uv pip install torch numpy matplotlib seaborn pandas

# Notebooks: use conda env "NN" in VS Code
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
