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
3. Apply the model to the K&M rule-action selection task (Kikumoto et al. 2025) to study how
   the hippocampal circuit learns neural state transition, consolidation, and abstraction
4. Generate testable EfAb predictions: conjunctive representation strength/stability, power-law
   improvement, overnight cue-abstraction (see §10)
5. Use the same modular structure as BasalGangliaACC so code can be shared/compared

**SR(t) extension: DEFERRED.**
The original plan was to extend with a temporally-structured SR, but value = SR × reward, and it
is currently unclear whether the SR framework is the right account of neural state learning in the
K&M task (where reward structure is trivial — all correct responses are equally rewarded). The
EfAb phenomena (n_dynamic/n_stable, power-law, abstraction) may be better explained directly by
Hebbian transition learning in MSP/TSP without a formal SR. See §11 for preserved SR notes.

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
ECin ─────────────────────────────────────────────► CA1 ──► ECout
 │                         [MSP: direct, lr=0.05]    ▲
 ├──► DG ──(5% mossy fiber)──┐                       │
 │    [~1% sparse]            ├──► CA3 (↺) ───────────┘
 └────────(25%, direct)───────┘   [~10%]   [TSP: episodic, lr=0.4]
           [Schapiro 2017 §2.a.iii]
```

### Two Complementary Pathways

| Pathway | Route | Function | Learning |
|---------|-------|----------|----------|
| **MSP** (Monosynaptic) | ECin → CA1 | Statistical regularities; smooth overlap | Slow Hebbian |
| **TSP** (Trisynaptic) | ECin → DG → CA3 → CA1; also ECin → CA3 direct (25%) | Episodic binding; unique events | Fast Hebbian |

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

### Training modes and kWTA policy

Three modes determine how weights are trained and which kWTA variant is active.
The `learnable: bool` flag on each layer controls `requires_grad`; kWTA type is separate.

| Mode | Steps | `learnable` | kWTA | Weight update |
|------|-------|-------------|------|---------------|
| CHL (Hebbian) | 1–7 | False | Hard | Manual `.data +=` |
| Backprop | 8 | True | Hard | `loss.backward()` + optimizer |
| RSA fitting | future | True (frozen BG) | Soft (`F_kWTA_soft`) | optimizer over embedding only |

**Why hard kWTA in backprop mode?**
Hard kWTA is non-differentiable, but gradients still flow through the active units
(the zero'd units contribute zero gradient, which is biologically plausible).
The BGACC codebase uses the same policy: hard kWTA throughout Steps 1–8; soft kWTA
only when RSA fitting requires a fully differentiable inhibition path.

**`learnable` flag design:**
`learnable=False` (default): weights are `nn.Parameter` with `requires_grad=False`.
Manual CHL updates use `.data +=` — no gradient tape needed.
`learnable=True`: switches `requires_grad=True` so a PyTorch optimizer can update
the same weights. The CHL `update_weights()` methods still work in this mode;
they bypass autograd by writing `.data` directly. Caller's responsibility to choose.

**`F_kWTA_soft` (not yet implemented):**
Soft kWTA via low-temperature softmax over pre-activation values.
Differentiable approximation for RSA fitting. Deferred to the RSA fitting step.

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
| 2 | `L_ECin`, `L_ECout` | `src/layer.py` | `notebook/test_layers.ipynb` | **done** |
| 3 | `L_DG` (sparse kWTA, pattern separation) | `src/layer.py` | `notebook/test_layers.ipynb` | not started |
| 4 | `L_CA3` (recurrent attractor, Euler) | `src/layer.py` | `notebook/test_layers.ipynb` | not started |
| 5 | `L_CA1` (MSP + TSP convergence, Euler) | `src/layer.py` | `notebook/test_layers.ipynb` | not started |
| 6 | `CommunityGraphEnv`, `CommunityGraphDataset` | `src/tasks.py` | `notebook/test_task.ipynb` | not started |
| 7 | `M_HipSL` assembly + CHL training loop | `src/model.py` | `notebook/test_full_model.ipynb` | not started |
| 8 | Reproduce Schapiro 2017 (RSA, pattern completion) | notebooks | `notebook/test_full_model.ipynb` | not started |
| 9 | `RuleActionEnv`, `RuleActionDataset` — K&M task with rate-coded ECin | `src/tasks.py` | `notebook/test_task.ipynb` | not started |
| 10 | Train M_HipSL on K&M task; read out CA1 RSA vs. RSRCONJ matrix | notebooks | `notebook/test_full_model.ipynb` | not started |

---

## 10. EfAb Target Task: K&M Rule-Action Selection

> **Source:** Kikumoto et al. (2025) *Cerebral Cortex*; Kikumoto & Mayr (2020)
> **Tables:** `src/z_task_design_tables/RuleAction_4rules/`

### Task structure

4 action rules × 4 stimuli = 16 unique action contexts (conjunctions).
Each conjunction (rule, stim) maps to exactly one response.
The task-relevant representational hierarchy (from Kikumoto et al. 2025 RSA analysis):

| Level | Code | What it captures |
|-------|------|-----------------|
| STIM | stimulus identity | which of 4 stimuli (shared across rules) |
| RULE | rule identity | which of 4 rules (shared across stim) |
| RESP | response identity | which of 4 buttons (shared S→R mappings) |
| SRCONJ | S-R conjunction | 8 unique stimulus-response pairs |
| RSRCONJ | rule-specific S-R conjunction | all 16 fully unique context codes |

RSRCONJ is the key level: the context-specific conjunction that practice strengthens and
overnight sleep makes cue-invariant (Kikumoto et al. 2025 Figs. 4–6).

### z_task_design_tables

`src/z_task_design_tables/` contains RSA model matrices as tab-separated 16×16 similarity
matrices (one file per representational level). Each entry M[i,j] is the expected cosine
similarity between conditions i and j if the brain fully encoded that level.

Primary tables used (RuleAction_4rules/BASIC(4rules)/):
- `STIM.txt`    — block diagonal; 4 blocks of 4 (0.25 within-stim, 0 across)
- `RULE.txt`    — stripe diagonal; each rule appears in 4 conditions across all stim
- `RESP.txt`    — groups by response identity
- `SRCONJ.txt`  — 8 unique SR pairs; each condition shares 0.5 with one cross-rule partner
- `RSRCONJ.txt` — 16 fully unique codes; each condition shares 0.5 with one cross-rule partner

These matrices are the quantitative target for CA1 representations after training.

### Why rate-coded (not one-hot) ECin representations are required

**The problem with one-hot encoding:**
If ECin assigns one unit per condition (16 one-hot units), all 16 conditions are perfectly
equidistant in ECin space. There is no sequential statistical structure in the K&M task (unlike
Schapiro's community graph), so MSP has no signal to learn community-like clustering in CA1.
CA1 representations would remain equidistant regardless of training.

**What is needed:**
ECin must carry distributed, rate-coded activity patterns such that the similarity structure
of ECin patterns already reflects the task feature hierarchy (STIM and RULE, at minimum).
Then MSP Hebbian learning in CA1 can develop the higher-level conjunctive structure (RSRCONJ)
by binding the co-occurring rule and stimulus features.

**Two viable approaches:**

Option A — Feature-coded ECin (recommended first step):
```
ECin = [rule_units (4) | stim_units (4)]  → 8-dimensional rate-coded input
rule_units:  one-hot over 4 rules   (activity = 1.0 for active rule, 0 elsewhere)
stim_units:  one-hot over 4 stimuli (activity = 1.0 for active stim, 0 elsewhere)
```
This gives natural RULE and STIM similarity without learned parameters.
CA1 must learn to form conjunctive (RSRCONJ) representations via Hebbian CHL.

Option B — Learned embedding projection:
```
ECin_emb = Linear(n_conditions=16, n_ECin)  [learned]
```
Embedding is updated during training; the model learns its own input geometry.
More flexible but harder to interpret; consider after Option A is validated.

### Kikumoto et al. (2025) — Key Findings

**Paper:** Kikumoto, A. et al. (2025). Practice reshapes the geometry and dynamics of
task-tailored representations. *Cerebral Cortex*.

**Participants / design:** n=40, 3 consecutive days, EEG (64ch, 250 Hz after downsampling).
Trial: ITI 300 ms → Cue 150 ms → blank 150 ms → Stimulus (until response).
4 rules × 4 stimuli = 16 action contexts. Each rule cued by 2 synonymous words (e.g.,
"vertical" / "updown") — cue identity is task-redundant. ~202 blocks/day.

**Neural state:** Time-resolved EEG pattern (155 features: 5 frequency bands × 31 electrodes)
at each time point → decoded by penalized LDA over 16 conditions → RSA against 5 model
matrices (STIM, RULE, RESP, SRCONJ, RSRCONJ). RSA coefficients at each time point = strength
of each representational level in the neural geometry.

**Geometry finding (Fig. 4):**
RSRCONJ representational dimensionality increases significantly across days
(proportion of implementable binary separations over all 65,536 pairings of 16 conditions).
No such increase for STIM, RULE, RESP, or SRCONJ. Only context-specific conjunctions are
strengthened by practice.

**Dynamics finding (Fig. 5–6):**
Temporal generalization matrix (train at time t1, test at time t2):
- **n_stable** = off-diagonal entries: strength of RSRCONJ representations that generalize
  across time points (temporally stable, sustained activity).
  Increases across days; stabilizes earlier in the trial (0–300 ms post-stimulus).
- **n_dynamic** = on-diagonal entries: strength at matched train/test time (transient,
  time-specific activity).
  More n_dynamic within a session predicts *less* improvement (r < 0, Fig. 6B).
More n_stable, less n_dynamic → faster and more stable performance (lower RT, lower SD).

**Power-law improvement (Fig. 2):**
RT and RT variability follow a power function of cumulative instances of each specific
conjunction. Stronger RSRCONJ representations within a session predicted the asymptote of
the power-law curve.

**Overnight abstraction (Fig. 6A–C):**
Cross-cue RSA (train decoder on one cue word, test on synonym) for RSRCONJ increases
linearly across days. Correlates with reduced cue-switch costs in RT and SD.
Interpretation: RSRCONJ representations become cue-invariant overnight (cue identity
compressed to low-variance dimensions through sleep-dependent consolidation).

**Hippocampus (Discussion, p. 13):**
Not directly measured. Invoked as candidate for within-session changes:
"changes in separation and stability... might be driven by... episodic encoding mechanisms
through the hippocampus." Overnight abstraction attributed to replay/consolidation:
"changes leading to abstraction over irrelevant features might require latent consolidation
mechanisms, such as... replay... and synaptic downscaling... regulated over a night of sleep."

**Mapping to EChipp_SL:**
| Kikumoto 2025 phenomenon | EChipp_SL mechanism |
|--------------------------|---------------------|
| RSRCONJ geometry grows with practice | CA1 RSA → RSRCONJ matrix via MSP Hebbian CHL |
| n_stable increases (earlier in trial) | ActMid→ActM→ActP trajectory stabilizes |
| n_dynamic decreases | CA1 within-trial variability decreases |
| Overnight cue-abstraction | Sleep consolidation: MSP re-runs with slow lr; cue dim. compressed |
| Power-law improvement | ECout output probability for correct item increases across trials |
| TSP episodic binding | Individual rule+stim→resp pairings stored in CA3 attractors |

### Moving window for K&M task

In Schapiro's community graph, the moving window encodes current + previous item.
For K&M, the equivalent is the within-trial sequence:
- Q1 (ECin-dominant): rule presented alone → ECin encodes rule_units only
- Q2-Q3 (CA3-dominant): stimulus added → ECin encodes rule_units + stim_units
- Q4 (plus phase): response clamped to ECout

The temporal asymmetry is now within a trial (rule → rule+stim), not across trials.
This maps naturally onto the two minus phases: Q1 = rule encoding; Q2-Q3 = conjunction retrieval.

### T_TaskEmbedding: ECin input geometry as a simulation parameter

The K&M task has no sequential community structure (unlike Schapiro), so the question of what
geometry ECin carries into the hippocampus is a design choice, not a given. Two experimental
modes are supported via `T_TaskEmbedding` in `src/tasks.py`:

**Mode 1 — Fixed feature-coded input (Option A, default for model testing):**
ECin = concatenated feature one-hots from `design_to_units()`. No learned parameters.
STIM and RULE similarity are encoded in the input; conditions are equidistant at the RSRCONJ level.
CA1 must discover conjunctive structure entirely through Hebbian CHL.
Use to test whether the model generates RSRCONJ geometry from a flat start.

**Mode 2 — RSA-initialized embedding (`T_TaskEmbedding` with `rsa_matrix`, for Kikumoto matching):**
ECin = `T_TaskEmbedding` output initialized via eigendecomposition of a target RSA matrix.
Input geometry already reflects the target representational level (e.g., RULE or STIM).
The initialization uses MDS-style eigendecomposition: eigenvectors of the RSA matrix, scaled by
sqrt(eigenvalues), give an embedding where cosine similarities match the target RSA structure.

Use this mode to:
- Match early-training EEG from Kikumoto 2025, where STIM and RULE geometry are already present
  before practice reshapes RSRCONJ — CA1 does not start from a blank slate in real data
- Ask how much of CA1's RSRCONJ learning is driven by input geometry vs. Hebbian statistics alone
- Simulate the effect of varying input correlation structure on conjunctive representation formation

**`freeze_on` flag:**
`freeze_on=True` freezes the embedding after initialization — ECin geometry is a fixed input
manipulation, not a learned variable. `freeze_on=False` allows the embedding to update during
training, approximating Option B (learned projection).

**Design rationale:**
The goal is to decode and confirm that CA1 neural states can become abstract (cue-invariant,
RSRCONJ-structured). To validate the model against Kikumoto 2025, ECin must carry the same
feature correlation structure that human EEG carries at trial onset. T_TaskEmbedding with
RSA initialization makes this input geometry a controllable simulation parameter.

---

## 11. SR(t) Extension: EfAb Computational Framework [DEFERRED]

> **Status: DEFERRED.** SR(t) is preserved here for reference but is not part of the current
> implementation plan. The deferral reason: value = SR × reward. In the K&M task, reward
> structure is trivial (all correct responses equally rewarded), so the SR-to-reward link does
> not drive the representational phenomena of interest. Whether MSP's Hebbian CHL can be
> reinterpreted as SR learning remains an open research question for a later phase.

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

## 12. Key Research Questions

1. Does MSP Hebbian CHL on the K&M task produce CA1 representations whose RSA geometry
   matches the RSRCONJ matrix after training?
2. Does the CA1 within-trial trajectory (ActMid → ActM → ActP) stabilize earlier (closer to
   ActMid) with more training — the n_dynamic → n_stable convergence pattern?
3. Does overnight replay / weight consolidation (simulated by re-running CHL with reduced lr)
   generalize CA1 representations across the cue dimension (cue-switch cost reduction)?
4. What is the minimal circuit (MSP alone, TSP alone, or both) sufficient to produce each of
   these effects?
5. Can the model account for power-law improvement in ECout output probability across training?

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
