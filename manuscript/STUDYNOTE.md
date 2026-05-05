# EChipp_SL — Study Notes

Progressive notes on the Schapiro (2017) reimplementation.
Updated as each step is implemented and understood.

---

## Step 1 — F_nxx1, F_kWTA (src/util.py) ✓

### nxx1 activation

Formula (O'Reilly & Munakata 2000, Ch. 2 Eq. 2.12):

```
y = γ · [Vm − θ]₊ / (γ · [Vm − θ]₊ + 1)
  = u / (u + 1),   where u = γ · relu(Vm − θ)
```

Default parameters:
- γ = 600, θ = 0.25 (Leabra defaults; no DA modulation here, unlike BGACC)

NoisyXX1: convolve XX1 with Gaussian kernel N(0, σ²), σ = 0.005.
Why: models input noise; smooths the threshold; makes the function differentiable.

Hand calculation (Vm=0.3):
- v = 0.3 − 0.25 = 0.05
- u = 600 × 0.05 = 30
- y = 30/31 ≈ 0.9677

### kWTA inhibition

Only top-k units stay active; rest zeroed out.
Active units retain original values (not rescaled).

Sparsity targets (Schapiro 2017 §2.2):
- ECin, ECout: k = 2 (absolute count, not fraction)
- DG: k_frac = 0.01 → ~1%
- CA3, CA1: k_frac = 0.10 → ~10%

Why high DG sparsity: forces orthogonal representations → minimizes CA3 interference → each episode gets a unique code.

---

## Paper: Schapiro et al. (2017) — Key Points

### Circuit

```
ECin ─────────────────────────────────────────► CA1 ──► ECout
 │                      [MSP]                    ▲
 ├──► DG ──(5% mossy)──┐                        │
 │    [~1%]             ├──► CA3 (↺) ────────────┘
 └────(25%, direct)─────┘   [~10%]    [TSP]
      [§2.a.iii]
```

MSP (monosynaptic): ECin → CA1, slow lr=0.05, learns statistical regularities.
TSP (trisynaptic): ECin → DG → CA3 → CA1, fast lr=0.4, learns individual episodes.

### CHL — Contrastive Hebbian Learning (§2.b)

The weight update rule used throughout the model.

```
ΔW = lr × (ActP ⊗ ActP  −  ActM ⊗ ActM)
```

- **ActM** = activity at end of Q3 (cycle 75) — minus phase; network's free prediction
- **ActP** = activity at end of Q4 (cycle 100) — plus phase; ECout clamped to correct target
- **⊗** = outer product (post × pre vectors → weight matrix shape)

The difference `ActP − ActM` is the implicit error signal. No explicit gradient needed.

Per-pathway learning rates (Go reimplementation):

| Pathway | Weights | lr | Why |
|---------|---------|-----|-----|
| MSP | ECin→CA1 | 0.05 | Accumulate statistics over many trials |
| TSP | ECin→DG, DG→CA3, CA3→CA3, CA3→CA1 | 0.4 | Bind one episode fast before it's overwritten |
| Output | CA1→ECout | 0.05 | Matches MSP |

§2.b: "The learning rate in the TSP is set to be 10× higher than in the MSP."

### Two minus phases per trial (§2.b — critical detail missed early)

1 trial = 100 cycles = 4 quarters × 25 cycles:
- Q1 (cycles 1–25): ECin → CA1 strong; CA3 → CA1 off. Models theta trough (encoding).
- Q2-Q3 (cycles 26–75): CA3 → CA1 strong; ECin → CA1 reduced. Models theta peak (retrieval).
- Q4 (cycles 76–100): ECout clamped to target. Plus phase.

Weight update uses ActM (end of Q3, cycle 75) and ActP (cycle 100).

### Moving window (§2.c)

ECin holds: current item = 1.0, previous item = 0.9 (decayed).
This temporal asymmetry → forward learning bias (A predicts B, not B predicts A).

### Sparse TSP connectivity

- ECin → DG: each DG unit receives from 25% of ECin units
- DG → CA3: 5% (mossy fibre; very sparse)

### 500 network initializations (§2.a.v)

Each simulation runs 500 networks with different random sparse projections.
Results averaged across networks (random effects model).

### Community structure task (§3.b)

- 15 items, 5 communities × 3, random walk on graph
- 60 trials/epoch × 10 epochs
- CA1 develops community-level clustering (MSP effect)
- DG/CA3 retain episode-level representations (TSP effect)

---

## Step 2 — L_ECin, L_ECout (src/layer.py) ✓

### Understanding checks (answered before coding)

**Separate Input layer**: ECin must both receive the stimulus clamp and receive ECout
back-projections (the "big loop"). A separate upstream Input layer absorbs the clamp,
leaving ECin free to also accept ECout signals without conflict.
Schapiro (2017) §2.a.ii: "Input was clamped in this layer so as to allow ECin to also
receive input from ECout, completing the 'big loop' of the model."

**ECout minus vs plus phase**:
- Minus (Q1, Q2-Q3): `forward(a_CA1)` settles freely — network predicts current item.
- Plus (Q4): `clamp(target_pattern)` overwrites `_activity` — no settling, just teaching signal.
The clamped ECout activity propagates back to CA1 via `W_ECout` in `L_CA1`.

**k=2 for ECout (n_items=15)**: 15 units total, only top 2 remain active, rest zeroed.
2/15 ≈ 13% sparsity. Matches the moving window: exactly current + previous item units active.

### L_ECin.forward()

Returns one-hot (n_items,) float tensor for a given item index.
Moving window assembly (current=1.0 + prev=0.9) is handled at the model level (M_HipSL),
not inside L_ECin. The layer just produces one-hot patterns on request.

### L_ECout.forward()

Three steps per settling cycle (minus phases only):

1. Net input: `net = a_CA1 @ W`  (W is (n_CA1, n_items))
2. Activation: `vm = F_nxx1(net)` (gamma=600, theta=0.25)
3. Inhibition: kWTA absolute k=2 via `kthvalue` threshold
4. Euler update: `_activity = (1−tau)*_activity + tau*new_act`

### L_ECout.update_weights()

CHL rule (O'Reilly & Munakata 2000, Ch. 4 Eq. 4.3):
```
ΔW = lr × (outer(a_CA1_plus, a_ECout_plus) − outer(a_CA1_minus, a_ECout_minus))
W.data += ΔW
```
W shape (n_CA1, n_items) → outer products have correct shape automatically.

---

## Step 3 — L_DG (src/layer.py) ✓

### What is L_DG and why does it exist?

DG is the first synapse in the TSP pathway (ECin → DG → CA3 → CA1).
Its sole computational job is **pattern separation**: transforming similar ECin input patterns
into maximally dissimilar DG output patterns.

**Why this matters**: if item 3 and item 4 (in the same community) activate overlapping
CA3 units, Hebbian learning for episode 3 will partially overwrite episode 4 — catastrophic
interference. DG prevents this by ensuring that even similar ECin patterns produce
near-orthogonal DG codes, so CA3 can store distinct attractors per episode.

Schapiro (2017) §2: "Connection sparsity and high inhibition result in few units being
active at any time in DG and CA3, and allow the layers to avoid interference by forming
separated, conjunctive representations of incoming patterns, even when the patterns are
highly similar."

### Sparse ECin → DG connectivity (25%)

Each DG unit receives input from a random 25% of ECin units.
Schapiro (2017) §2.a.iii: "Each DG and CA3 unit receives input from 25% of the ECin layer."

Why 25%? Combined with aggressive kWTA below, sparse random projections ensure each DG
unit responds to a different subset of the input space. This is the biological perforant
path from entorhinal cortex, which is anatomically sparse.

The mask is fixed at network initialization (re-randomized across the 500 simulations in
§2.a.v). It never changes during learning — only the weights W behind the existing connections
are updated by CHL.

```python
self.mask = (torch.rand(n_input, n_DG) < ecin_frac).float()  # 0 or 1; fixed
self.W    = nn.Parameter(torch.zeros(n_input, n_DG))          # learned; masked
```

### ~1% sparsity (kWTA with k_frac = 0.01)

With n_DG=100: k = max(1, int(0.01 × 100)) = 1 active unit per pattern.

Why 1%? Consider two random DG patterns each with k=1 active unit out of 100:
- Probability they share the same active unit ≈ 1/100 = 1%
- Expected overlap ≈ 0.01 units

Representations this sparse are nearly orthogonal by construction.
CA3 can then form a distinct attractor for each episode without overlap.

Compare: if DG had 50% sparsity (50 active units out of 100), two patterns would share
~25 active units on average — enormous overlap → CA3 attractors interfere → TSP fails
at episodic binding.

### One settling cycle — forward pass

```
net     = a_ECin @ (W * mask)     # (n_DG,) — masked feedforward net input
vm      = F_nxx1(net)             # NoisyXX1: net → firing rate [0, 1]
new_act = F_kWTA(vm, k_frac=0.01) # ~1 unit active; rest zeroed
_activity = (1−tau)*_activity + tau*new_act   # Euler; tau=0.1
```

Step-by-step for one DG unit j:
1. Sum weighted ECin inputs (only from the 25% it's connected to):
   `net_j = Σ_i  W_ij * mask_ij * a_ECin_i`
2. Convert net input to firing rate via NoisyXX1 (gamma=600, theta=0.25):
   `vm_j = F_nxx1(net_j)`
3. kWTA: only the unit(s) with the highest vm survive; all others → 0
4. Euler: blend new activity with previous state (slow settling over 25 cycles per quarter)

**Why Euler?** The original emergent model uses ODE dynamics. We approximate with Euler
integration at tau=0.1: a unit's activity converges to its steady-state over ~10 steps.
The settling plateau within Q1 (25 cycles) replaces the ODE fixed-point.

### CHL weight update

After one trial, W_ECin_DG is updated by the CHL rule:

```
ΔW = lr × mask × (outer(a_ECin_plus, a_DG_plus) − outer(a_ECin_minus, a_DG_minus))
W.data += ΔW
```

- `outer(a_ECin, a_DG)` has shape (n_input, n_DG), matching W
- `mask ×` ensures only existing connections are updated (non-connected weights stay 0)
- `lr = lr_TSP = 0.4` (Go reimplementation; original emergent: 0.2; Schapiro 2017 §2.b)

Why fast lr (0.4) for TSP? TSP must bind individual episodes in one or very few exposures —
before the next trial overwrites them in STM. If TSP learned as slowly as MSP (lr=0.05),
each episode would take hundreds of exposures to be encoded in CA3, defeating the purpose.

### Understanding checks (answered)

**Q: Why does 1% DG sparsity prevent CA3 interference?**
A: With only ~1 DG unit active per item, any two items share ≈0 active DG units.
The DG→CA3 mossy fibre then seeds a unique CA3 subpopulation for each episode.
CA3 attractors are therefore completely separated → updating one doesn't affect the other.

**Q: What happens on the first trial of a new item?**
A: W starts at zero → net=0 → nxx1(0)≈0 for all units → kWTA keeps the top 1 arbitrarily
(tie-breaking). DG output is essentially random. After the first CHL update, the mask × outer
product begins to build a sparse, sparse weight signature. After a few trials the pattern
is stable enough for CA3 to form an attractor.

**Q: What if ecin_frac were 100% (full connectivity) instead of 25%?**
A: All DG units would receive the same ECin input → net inputs would be more correlated
across units → kWTA would produce less stable, more similar winners across similar inputs
→ weaker pattern separation. The sparse random projection is what makes winners random and
item-specific even when ECin patterns are similar.

---

## Step 4 — L_CA3 (src/layer.py) ✓

### What is L_CA3 and why does it exist?

CA3 is the second synapse in the TSP pathway and is the pattern completion engine.
DG feeds CA3 a sparse, near-orthogonal code for each episode. CA3's recurrent (Hopfield)
connections store that code as an attractor, so that a partial cue later reinstates the
full pattern.

Schapiro (2017) §2.a.iii: "CA3 also has a fully connected projection to itself, which
helps bind pieces of a representation to one another and retrieve a full pattern from a
partial cue."

### Two input streams

1. **Mossy fibre (W_ff)**: DG → CA3, 5% sparse. Even sparser than ECin→DG (25%).
   Forces each episode to seed a unique CA3 subpopulation from a unique DG code.
   §2.a.iii: "The 'mossy fibre' projection from DG to CA3 is even sparser (5%)."

2. **Recurrent (W_rec)**: CA3 → CA3, fully connected.
   Every CA3 unit connects to every other CA3 unit.
   Enables Hopfield attractor dynamics: partial activation → recurrence fills in the rest.

### One settling cycle — forward pass

```
net     = a_DG @ (W_ff * mask_ff)     # mossy fibre input  (n_CA3,)
net    += _activity @ W_rec            # recurrent input    (n_CA3,)
vm      = F_nxx1(net)                  # NoisyXX1
new_act = F_kWTA(vm, k_frac=0.10)     # ~10% active
_activity = (1−tau)*_activity + tau*new_act   # Euler; tau=0.1
```

Key difference from L_DG: the `_activity @ W_rec` term feeds previous CA3 state back
into the current step's net input. On cycle 1, `_activity = 0` so CA3 acts as pure
feedforward. By cycle ~10 the attractor has accumulated enough self-excitation to
dominate the DG input.

### 10% sparsity — why less than DG?

DG uses 1% to guarantee near-orthogonal seeding codes.
CA3 uses 10% to allow the overlap needed for pattern completion:
- Too sparse (1%): partial cue activates too few units → attractor doesn't fire → no completion
- Too dense (50%): patterns overlap too much → attractors merge → wrong pattern retrieved
- 10%: enough overlap for completion, enough sparsity to keep attractors distinct

### CHL weight update — two matrices

```python
# Feedforward (mossy fibre) — masked
ΔW_ff  = lr * mask_ff * (outer(a_DG_plus, a_CA3_plus) − outer(a_DG_minus, a_CA3_minus))

# Recurrent (Hopfield) — no mask, fully connected
ΔW_rec = lr * (outer(a_CA3_plus, a_CA3_plus) − outer(a_CA3_minus, a_CA3_minus))
```

W_rec uses `outer(a_CA3, a_CA3)` — same vector for both pre and post synaptic activity.
This produces a **symmetric weight matrix**, consistent with the Hopfield model.
The auto-correlation of the plus-phase pattern minus the auto-correlation of the minus-phase
pattern drives W_rec to store the plus-phase attractor.

### Understanding checks (answered)

**Q: What does W_rec do on the very first trial?**
A: W_rec = 0 at init → `_activity @ W_rec = 0` on every cycle → CA3 is purely feedforward
(DG input only). After the first CHL update, W_rec gets a small nonzero entry proportional
to `outer(a_CA3_plus, a_CA3_plus)`. With more trials, the attractor strengthens.

**Q: Why is the recurrent update `outer(a_CA3, a_CA3)` and not `outer(a_CA3_pre, a_CA3_post)`?**
A: In an attractor network, the "pre" and "post" neurons are the same population — CA3 connects
to itself. The Hebb rule for auto-association is: strengthen connections between co-active units
within CA3. That is exactly `outer(a_CA3, a_CA3)` — unit i × unit j for all pairs (i,j).

**Q: What prevents runaway excitation from W_rec?**
A: kWTA hard-clamps activity to ~10% after every cycle. No matter how strong W_rec becomes,
the top-10% gate limits the number of active units. This is the Leabra solution to the
stability problem that haunts unconstrained Hopfield networks.

---

## Step 5 — L_CA1 (src/layer.py)

_Not yet started._

Understanding check:
- Why does MSP need a slower lr than TSP?
- In the plus phase, what is the net input formula (all three input streams)?

---

## Step 6 — CommunityGraphEnv, CommunityGraphDataset (src/tasks.py)

_Not yet started._

Understanding check:
- In the community graph, what prevents the model from learning community structure
  from transition probabilities alone?
- What is the "big loop" (ECout → ECin) and why does it matter for community learning?

---

## Step 7 — M_HipSL assembly + CHL training loop (src/model.py)

_Not yet started._

---

## Step 8 — Reproduce Schapiro 2017 results

Target figures:
- Fig. 3a: CA1 RSA — within-community > across-community similarity after training
- Fig. 3c: output probability of same-community item increases over epochs
- Fig. 3d: CA1 settled − initial heatmap

---
