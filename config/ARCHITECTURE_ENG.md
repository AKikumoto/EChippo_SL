# EChipp_SL — Architecture & Implementation Plan

> **How to use this document:**
> Read the corresponding paper section before writing any code.
> Every numerical value has a citation (paper, page, equation number).
> Explain "why this value?" in your own words before moving to code.
> Answer the understanding checks before proceeding to the next step.

---

## 1. Project Goals

0. **Primary goal: understanding** — build the program step by step
1. Reproduce Schapiro et al. (2017) hippocampal statistical learning model (hip-sl) in PyTorch
2. Understand MSP/TSP dissociation at the level of equations and circuits
3. Extend with SR(t) (temporally-structured successor representation) over conjunctive task space
4. Generate testable predictions for the EfAb R21 grant (TUS × temporal generalization design)
5. Use the same modular structure as BasalGangliaACC so code can be shared/compared

**Why PyTorch instead of emergent/Go?**
- Emergent (original C++ implementation) is deprecated
- Go reimplementation (schapirolab/hip-sl) requires the emer/emergent framework — hard to extend
- PyTorch enables integration with RSA pipelines, SBI (BayesFlow), and the BGACC codebase
- Trade-off: no true ODE dynamics; use Euler integration instead (same trade-off as BGACC)

---

## 2. Target Model: Schapiro et al. (2017)

**Paper:** Schapiro, A. C., Turk-Browne, N. B., Botvinick, M. M., & Norman, K. A. (2017).
Complementary learning systems within the hippocampus. *Phil. Trans. R. Soc. B*, 372, 20160049.

**Original Go code:** https://github.com/schapirolab/hip-sl

Key parameter differences in the Go reimplementation vs. original emergent:
| Parameter | Emergent original | Go reimplementation |
|-----------|------------------|---------------------|
| Inhibition type | kWTA | FFFB |
| MSP learning rate | 0.02 | 0.05 |
| TSP learning rate | 0.2 | 0.4 |
| Quarter structure | 4 × 25 cycles | 4 × 25 cycles |
| ActMid recorded at | cycle 40 | cycle 25 |
| ActM recorded at | cycle 80 | cycle 75 |
| ActP recorded at | cycle 100 | cycle 100 |

This PyTorch implementation follows the Go reimplementation values unless otherwise noted.

---

## 3. Circuit Architecture

```
ECin ──────────────────────────────► CA1 ──► ECout
 │                    [MSP: direct]    ▲
 │                                     │
 └──► DG ──────────► CA3 ─────────────┘
       [TSP: episodic]
```

### Two Complementary Pathways

| Pathway | Route | Function | Learning |
|---------|-------|----------|----------|
| **MSP** (Monosynaptic) | ECin → CA1 | Statistical regularities; smooth overlap | Slow Hebbian |
| **TSP** (Trisynaptic) | ECin → DG → CA3 → CA1 | Episodic binding; unique events | Fast Hebbian |

**Core insight (Schapiro 2017 §2):**
MSP's direct ECin → CA1 connection, trained slowly, develops graded representations where items that frequently co-occur end up with similar CA1 representations — it learns the graph's community structure.
TSP's path through DG enforces pattern separation: even similar inputs activate distinct DG patterns, allowing CA3 to form distinct attractors for each episode.

> **Understanding check:** Why does the MSP need to be slow? If it learned as fast as TSP, what would happen to community structure learning?

### Projections table (Schapiro 2017, §2.a)

| From | To | Type | Connectivity | Learned? |
|------|----|------|-------------|---------|
| ECin | CA1 | Feed-forward | Full (all-to-all) | Yes (MSP) |
| ECin | DG | Feed-forward | Sparse: each DG unit receives from 25% of ECin | Yes (TSP) |
| DG | CA3 | Feed-forward | Sparse: 5% (mossy fibre) | Yes (TSP) |
| CA3 | CA3 | Recurrent | Full (all-to-all) | Yes (TSP) |
| CA3 | CA1 | Feed-forward | Full (all-to-all) | Yes (TSP) |
| CA1 | ECout | Feed-forward | Full (all-to-all) | Yes |
| ECout | CA1 | Back-projection | Full (all-to-all) | Yes (plus-phase teaching signal) |

**Input layer (Schapiro 2017 §2.a.ii):** A separate Input layer (not shown in Fig. 1) has one-to-one connections to ECin. Input is clamped here — this allows ECin to also receive ECout back-projections (the "big loop"), without the clamp disrupting the stimulus representation.

---

## 4. Layer Descriptions

> **Read:** Schapiro (2017) §2.1; O'Reilly & Munakata (2000) Chapter 2 (Leabra)

### L_ECin — Entorhinal Cortex Input

- Localist representation: one unit per item (n_items units)
- No learning in ECin itself; it is the input driver
- **Moving window (Schapiro 2017 §2.c):** ECin receives activity for both the current and previous item. Current item = 1.0 (clamped); previous item = 0.9 (decayed). This temporal asymmetry is the source of directional learning (forward bias).
- **Separate Input layer:** A hidden Input layer (one-to-one with ECin) holds the clamped stimulus, allowing ECin to also receive ECout back-projections without conflict.
- Inhibition: k = 2 (absolute; paper §2.a.ii — two units active at a time)
- In all phases: same ECin pattern (stimulus does not change across minus/plus phases)

### L_DG — Dentate Gyrus

- Role: pattern separation — make each item's DG pattern as distinct as possible
- Sparsity: ~1% active units (much sparser than other layers)
- Mechanism: aggressive kWTA — only the top-k (≈1%) units are active
- Receives: ECin (feed-forward)
- No recurrent connections
- Fast learning rate (TSP pathway)

**Why such high sparsity?**
O'Reilly & Munakata (2000): high sparsity ensures orthogonal representations → minimizes interference → each episode gets a unique DG code → CA3 can form distinct attractors (Schapiro 2017 §2.2).

> **Understanding check:** If DG sparsity were reduced to 50% (same as CA3), what would happen to TSP's ability to form distinct episodic memories?

### L_CA3 — CA3 Field

- Role: pattern completion (attractor dynamics)
- Receives: DG (feed-forward), CA3 (recurrent)
- Recurrent connections are the key: partial cue → CA3 activates → recurrent connections reinstate full pattern
- Sparsity: ~10% (less sparse than DG; allows overlap for pattern completion)
- Stateful: Euler integration; carries `_activity` buffer

**Recurrent dynamics (O'Reilly & Munakata 2000, Ch. 2):**
```
net_CA3(t) = W_DG→CA3 @ a_DG + W_CA3→CA3 @ a_CA3(t-1)
a_CA3(t)   = (1 - tau) * a_CA3(t-1) + tau * F_nxx1(net_CA3(t))
```

> **Understanding check:** CA3 recurrent connections enable pattern completion. But during the first trial on a new item, there are no stored CA3 patterns. What does CA3 output then? Is this a problem for the first trial?

### L_CA1 — CA1 Field

- Role: convergence of MSP (from ECin) and TSP (from CA3) signals
- Receives: ECin (MSP, slow learning), CA3 (TSP, fast learning), ECout (plus-phase teaching signal, back-projection)
- The MSP vs. TSP tension is resolved here: which input dominates depends on relative weight strengths + learning rates
- Stateful: Euler integration

**Net input:**
```
net_CA1 = W_ECin→CA1 @ a_ECin + W_CA3→CA1 @ a_CA3
```
In plus phase: ECout back-projection also drives CA1.

### L_ECout — Entorhinal Cortex Output

- Receives: CA1 (feed-forward)
- Role: reconstruction of input pattern; provides plus-phase teaching signal via back-projection to CA1
- Inhibition: k = 2 (absolute; paper §2.a.ii — matches ECin, two units active at a time)
- In minus phases (Q1/Q2-Q3): settles freely from CA1 input; ECout → CA1 back-projection carries minus-phase activity
- In plus phase (Q4): ECout clamped to the target item pattern → provides error-correction signal to CA1

---

## 5. Activation Function

> **Read:** O'Reilly & Munakata (2000) Chapter 2, Equations 2.11–2.14

### nxx1 (NoisyXX1)

Same activation function as BasalGangliaACC / Frank models. Leabra framework.

$$y_j = \frac{1}{1 + \frac{1}{\gamma \cdot [V_m - \theta]_+}}$$

- $\gamma$ (gain): sharpness of the sigmoid. Default: 600 (O'Reilly & Munakata 2000)
- $\theta$ (threshold): firing threshold. Default: 0.25
- $[x]_+$: rectification — 0 if x < 0, x otherwise

**Default parameters (Schapiro 2017 follows Leabra defaults):**
| Parameter | Value | Source |
|-----------|-------|--------|
| gain $\gamma$ | 600 | O'Reilly & Munakata (2000) |
| threshold $\theta$ | 0.25 | O'Reilly & Munakata (2000) |

Note: Unlike BasalGangliaACC, there is no DA modulation of gain here. Gain and threshold are fixed.

> **Understanding check:** What does nxx1 output when Vm = 0.25 (exactly at threshold)? What about Vm = 0.3?

### kWTA (k-Winners-Take-All) Inhibition

Lateral inhibition: only the top-k units remain active, others are suppressed.

ECin and ECout use an **absolute k=2** (paper §2.a.ii: "two units could be active at a time").
DG, CA3, CA1 use a **fractional k** (proportion of layer size):

| Layer | k | Resulting sparsity | Source |
|-------|---|--------------------|--------|
| ECin | 2 (absolute) | 2/n_items ≈ 13% | Schapiro (2017) §2.a.ii |
| DG | ~0.01 × n_DG | ~1% | Schapiro (2017) §2.2 |
| CA3 | ~0.10 × n_CA3 | ~10% | Schapiro (2017) |
| CA1 | ~0.10 × n_CA1 | ~10% | Schapiro (2017) |
| ECout | 2 (absolute) | 2/n_items ≈ 13% | Schapiro (2017) §2.a.ii |

In PyTorch (hard kWTA, non-differentiable):
```python
def F_kWTA(activity, k_frac):
    n_active = max(1, int(k_frac * activity.shape[-1]))
    threshold = activity.kthvalue(activity.shape[-1] - n_active + 1, dim=-1).values
    return activity * (activity >= threshold.unsqueeze(-1)).float()
```

### FFFB Inhibition (Go reimplementation default)

The Go reimplementation replaced kWTA with FFFB (feedforward-feedback inhibition).
FFFB is differentiable and approximates kWTA behavior.
This PyTorch implementation uses hard kWTA for Steps 1–8 (biologically faithful), matching the principle established in BasalGangliaACC.

---

## 6. CHL Learning Rule

> **Read:** O'Reilly & Munakata (2000) Chapter 4 ("Contrastive Hebbian Learning"); Schapiro (2017) §2.3

Contrastive Hebbian Learning (CHL): the core learning mechanism in Leabra.

**Three-phase trial structure — 100 cycles total (Schapiro 2017 §2.b-c):**

Each trial = 4 quarters × 25 cycles. The two "minus phases" correspond to the two phases of the hippocampal theta oscillation (Hasselmo et al. 2002, ref [27]; Brankack et al. 1993, ref [28]):

```
Q1 — Minus phase 1 (encoding; cycles 1–25):
     ECin → CA1 connection at full strength
     CA3 → CA1 connection inhibited (or reduced)
     Models theta trough: strong EC input to CA1 (external drive dominates)
     Activity recorded as ActMid (cycle 25 in Go reimplementation)

Q2–Q3 — Minus phase 2 (retrieval; cycles 26–75):
     CA3 → CA1 connection at full strength
     ECin → CA1 connection reduced
     Models theta peak: strong CA3 input to CA1 (internal recurrence dominates)
     Activity recorded as ActM (cycle 75 in Go reimplementation)

Q4 — Plus phase (correction; cycles 76–100):
     ECout clamped to target item pattern
     CA1 resettles toward target; ECout → CA1 back-projection corrects
     Plus phase starts from Q2-Q3 final state; never reset between phases
     Activity recorded as ActP (cycle 100)
```

**Weight update (applied after Q4, using ActM as minus and ActP as plus):**

$$\Delta W_{ij} = \alpha \cdot (\hat{y}_j^+ \cdot \hat{y}_i^+ - \hat{y}_j^- \cdot \hat{y}_i^-)$$

- $\hat{y}^+$: plus-phase (ActP) activity
- $\hat{y}^-$: minus-phase (ActM, i.e. Q2-Q3) activity
- $\alpha$: learning rate (MSP = 0.05, TSP = 0.4; from Go reimplementation)
- Note: TSP learning rate is 10× MSP in the original emergent model (§2.b)

**Learning rate differences (TSP faster than MSP):**
MSP must accumulate statistical regularities slowly — fast learning would cause each episode to overwrite previous ones. TSP needs to bind individual events quickly before they are forgotten (Schapiro 2017 §2.3).

> **Understanding check:** Why does TSP need a faster learning rate than MSP? What would happen to pattern completion if TSP learned as slowly as MSP?

---

## 7. Statistical Learning Task

> **Read:** Schapiro (2017) §2.4, Fig. 1

### Community Graph Structure

Items organized into communities with higher within-community transition probability:
- 15 items total: 5 communities × 3 items each (or configurable)
- Within-community transitions: high probability
- Between-community (bottleneck) transitions: low probability
- Each presentation: one item pair (current → next)

**Training procedure:**
- Random walk through the community graph
- Each step: present current item as ECin input
- Target: next item as ECout teaching signal
- Run CHL (minus phase → plus phase → weight update)
- Repeat for many epochs

**Expected results (Schapiro 2017 Fig. 3–4):**
- CA1 representations cluster by community after training
- MSP shows graded community structure (overlapping representations within community)
- TSP shows sharper boundaries (distinct representations per episode)
- Pattern completion (partial cue → correct item) driven by CA3 recurrence

### Key outcome measures
- **RSA (Representational Similarity Analysis):** CA1 pairwise similarity matrix — within-community pairs should be more similar than between-community
- **Pattern completion:** present partial cue (50% of item features) → CA3 completes → measure CA1 accuracy
- **Community clustering:** t-SNE or PCA of CA1 representations colored by community

---

## 8. Euler Integration Policy

All settling layers that participate in within-trial dynamics are **stateful** — they carry `_activity` buffer updated via Euler integration, same as BasalGangliaACC.

```python
# Euler update (all settling layers)
self._activity = (1 - self.tau) * self._activity + self.tau * new_act
```

**Default tau:** 0.1 (Leabra default; O'Reilly & Munakata 2000)

**Reset policy:**
- `reset()` called once per trial, before minus phase
- Plus phase starts from minus phase final state — never call `reset()` between phases
- `use_euler: bool = True` flag; `use_euler=False` gives stateless behavior for unit tests

**Layers requiring Euler integration:** `L_DG`, `L_CA3`, `L_CA1`, `L_ECout`
**Layers that are stateless:** `L_ECin` (input driver only)

---

## 9. Implementation Plan (10 Steps)

> **Rule:**
> - Each step is an independent class or function (modular structure)
> - Write the corresponding test notebook immediately after each step
> - Add paper citation in comments for every numerical value
> - Answer the understanding check before proceeding

| Step | Build | File | Test | Status |
|------|-------|------|------|--------|
| 1 | `F_nxx1`, `F_kWTA` activation functions | `src/util.py` | `notebook/test_nxx1.ipynb` | **done** |
| 2 | `L_ECin`, `L_ECout` | `src/layer.py` | `notebook/test_layers.ipynb` | not started |
| 3 | `L_DG` (sparse kWTA, pattern separation) | `src/layer.py` | `notebook/test_layers.ipynb` | not started |
| 4 | `L_CA3` (recurrent attractor, Euler) | `src/layer.py` | `notebook/test_layers.ipynb` | not started |
| 5 | `L_CA1` (MSP + TSP convergence, Euler) | `src/layer.py` | `notebook/test_layers.ipynb` | not started |
| 6 | `CommunityGraphEnv`, `CommunityGraphDataset` | `src/tasks.py` | `notebook/test_task.ipynb` | not started |
| 7 | `M_HipSL` assembly + CHL training loop | `src/model.py` | `notebook/test_full_model.ipynb` | not started |
| 8 | Reproduce Schapiro 2017 (RSA, pattern completion) | notebooks | `notebook/test_full_model.ipynb` | not started |
| 9 | `M_HipSL_SR`: SR(t) extension | `src/model.py` | `notebook/test_sr.ipynb` | not started |
| 10 | EfAb behavioral readouts (n_dynamic/n_stable, power-law) | `src/` | `notebook/test_sr.ipynb` | not started |

---

## 10. SR(t) Extension: EfAb Computational Framework

> **Source:** EfAb_grant_v2.md (April 2026); Stachenfeld et al. (2017); Momennejad (2020)

### Standard Successor Representation (SR)

The SR matrix M(s,s') gives the expected discounted future occupancy of state s' starting from state s:

$$M(s, s') = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \mathbf{1}[s_t = s'] \mid s_0 = s\right]$$

Value function decomposes as: $V(s) = M(s, \cdot) \cdot R$ where R is the reward function.
Practical value: changing the reward function only requires updating R, not relearning M.

**EC as eigenspace of SR (Stachenfeld et al. 2017; Momennejad 2020):**
EC represents the eigenvectors of M — the compressed basis for predicting future states.
The top eigenvectors of M correspond to the smoothest variation across the state space.

### SR(t): Temporally-Structured SR over Conjunctive Subspace

**EfAb extension:** SR(t) = M(s,s',t) where t indexes within-trial timestep.

**n_dynamic / n_stable decomposition:**
- `n_stable`: the stable endpoint state of a trial (post-response conjunctive representation)
- `n_dynamic(t)`: the dynamic state at timestep t within a trial
- SR(t)(s, t) = probability that n_dynamic reaches n_stable by time t

**Practice effects:**
- Early learning: SR(t) converges slowly — trajectory is variable
- Late learning: SR(t) converges fast — trajectory is stable
- Power-law improvement = behavioral signature of SR(t) convergence speeding up

**Overnight abstraction:**
- Cue identity is task-irrelevant (multiple cues map to same rule)
- SR(t) compression pushes cue identity to low-eigenvalue dimensions overnight
- Result: representations become cue-invariant → reduced cue-switch cost

**Discount parameter γ:**
- Small γ: encodes local trajectory (cue → rule)
- Large γ: encodes global trajectory (cue → outcome), approaching abstraction
- γ is a free parameter for SBI fitting; expected to increase with practice

### MSP connection to SR(t)

Schapiro's MSP (ECin → CA1) learns sequential transition structure via Hebbian CHL.
EfAb extension: MSP can be interpreted as learning SR(t) with a temporal discount factor γ.
MSP uses error-driven learning (ECin vs. ECout difference) — this is analogous to TD learning for SR(t).

Open question: Can Schapiro's MSP learning mechanism be extended to learn SR(t) with discount γ?
(This is the core computational modeling question for EfAb Step 9.)

> **Understanding check:** Standard SR learns M(s,s'). SR(t) adds a temporal index. Why does this matter for understanding within-trial dynamics? What does it mean for n_dynamic to "converge to" n_stable?

---

## 11. Key Research Questions

1. Does MSP learn SR-structured transition representations, and do these match EC BOLD patterns?
2. Can SR(t) with a temporal discount factor γ formally account for within-trial trajectory dynamics?
3. Does γ increase with practice, as predicted by the EfAb abstraction hypothesis?
4. What is the minimal hippocampal circuit architecture that can generate power-law improvement in CA1 representations?
5. Can n_dynamic/n_stable convergence be read out from CA1 RSA without fMRI (i.e., just EEG RSA)?

---

## 12. File Structure

```
EChipp_SL/
├── src/
│   ├── __init__.py
│   ├── util.py           F_nxx1, F_kWTA, F_fffb (Step 1)
│   ├── layer.py          L_ECin, L_ECout, L_DG, L_CA3, L_CA1 (Steps 2–5)
│   ├── model.py          M_HipSL (Steps 7–8); M_HipSL_SR (Step 9)
│   └── tasks.py          CommunityGraphEnv, CommunityGraphDataset (Step 6)
├── notebook/
│   ├── test_nxx1.ipynb           Step 1 verification
│   ├── test_layers.ipynb         Steps 2–5 verification
│   ├── test_task.ipynb           Step 6 verification
│   ├── test_full_model.ipynb     Steps 7–8: Schapiro 2017 reproduction
│   └── test_sr.ipynb             Steps 9–10: SR(t) extension
├── simulations/
├── scripts/
├── trained_models/
├── visualizations/
├── manuscript/
├── config/
│   ├── ARCHITECTURE.md   (this file)
│   └── requirements.txt
├── CLAUDE.md
└── README.md
```

---

## 13. PyTorch Limitations vs. Emergent

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| No ODE solver — Euler integration only | Approximate settling dynamics | Use small tau (0.1), verify plateau within n_steps |
| CHL requires two manual forward passes | Cannot use `loss.backward()` for Hebbian weights | Manual weight update (same as BasalGangliaACC) |
| Hard kWTA is non-differentiable | Cannot backprop through kWTA | Hard kWTA for Steps 1–8; soft kWTA if RSA fitting added |
| No built-in Hopfield attractor | CA3 recurrence must be explicit | Explicit recurrent buffer in `L_CA3._activity` |

---

## 14. Parameter Table

> **Sources:** Schapiro (2017); Go reimplementation (https://github.com/schapirolab/hip-sl/blob/master/hip.go_vs_hip-sl.go_param_changes.xlsx); O'Reilly & Munakata (2000)

| Parameter | Value | Layer | Source |
|-----------|-------|-------|--------|
| nxx1 gain γ | 600 | all | O'Reilly & Munakata (2000) |
| nxx1 threshold θ | 0.25 | all | O'Reilly & Munakata (2000) |
| Euler tau | 0.1 | all stateful | Leabra default |
| ECin/ECout k (absolute) | 2 | ECin, ECout | Schapiro (2017) §2.a.ii |
| DG sparsity k_frac | ~0.01 | DG | Schapiro (2017) §2.2 |
| CA3 sparsity k_frac | ~0.10 | CA3 | Schapiro (2017) |
| CA1 sparsity k_frac | ~0.10 | CA1 | Schapiro (2017) |
| ECin→DG connectivity | 25% (each DG unit receives from 25% of ECin) | ECin→DG | Schapiro (2017) §2.a.iii |
| DG→CA3 connectivity | 5% (mossy fibre; sparse) | DG→CA3 | Schapiro (2017) §2.a.iii |
| Previous item activity | 0.9 (decayed) | Input layer | Schapiro (2017) §2.c |
| MSP learning rate | 0.05 | ECin→CA1 | Go reimplementation |
| TSP learning rate | 0.4 | ECin→DG, DG→CA3, CA3→CA1 | Go reimplementation |
| n_items | 15 | community task | Schapiro (2017) Fig. 3 |
| n_communities | 5 | task | Schapiro (2017) Fig. 1 |
| Items per community | 3 | task | Schapiro (2017) Fig. 1 |
| Trials per epoch | 60 | community task | Schapiro (2017) §3.b |
| Training epochs | 10 | community task | Schapiro (2017) §3.b |
| Q1 settling cycles (minus1) | 25 | all | Go reimplementation |
| Q2-Q3 settling cycles (minus2) | 50 | all | Go reimplementation |
| Q4 settling cycles (plus) | 25 | all | Go reimplementation |
| Total trial cycles | 100 | all | Schapiro (2017) §2.c |
| ActMid recorded at | cycle 25 | all | Go reimplementation |
| ActM recorded at | cycle 75 | all | Go reimplementation |
| ActP recorded at | cycle 100 | all | Go reimplementation |
| Network initializations | 500 | simulation | Schapiro (2017) §2.a.v |

---

## 15. References

- Schapiro, A. C., Turk-Browne, N. B., Botvinick, M. M., & Norman, K. A. (2017). Complementary learning systems within the hippocampus. *Phil. Trans. R. Soc. B*, 372, 20160049.
- O'Reilly, R. C. & Munakata, Y. (2000). *Computational Explorations in Cognitive Neuroscience*. MIT Press.
- Stachenfeld, K. L., Botvinick, M. M., & Gershman, S. J. (2017). The hippocampus as a predictive map. *Nature Neuroscience*, 20, 1643–1653.
- Momennejad, I. (2020). Learning structures: predictive representations, replay, and generalization. *Current Opinion in Behavioral Sciences*, 32, 155–166.
- Garvert, M. M., Dolan, R. J., & Behrens, T. E. J. (2017). A map of abstract relational knowledge in the human hippocampal–entorhinal cortex. *eLife*, 6, e17086.
- Kikumoto, A. et al. (2025). Conjunctive representational trajectories predict power-law improvement and overnight abstraction. *Cerebral Cortex*.
- Mylonas, D. et al. (2024). Hippocampus is necessary for micro-offline gains. *J. Neurosci.*
