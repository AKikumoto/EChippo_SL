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

## Step 3 — L_DG (src/layer.py)

_Not yet started._

Understanding check:
- Why does high DG sparsity prevent CA3 interference?
- One settling cycle: write the net input formula from scratch before reading the code.

---

## Step 4 — L_CA3 (src/layer.py)

_Not yet started._

Understanding check:
- What does the recurrent weight W_CA3_CA3 do on the first trial of a new item?
- Write the Euler update formula before reading the code.

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
