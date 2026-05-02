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
ECin ──────────────────────────► CA1 ──► ECout
 │              [MSP]              ▲
 └──► DG ──► CA3 ─────────────────┘
       [TSP]
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

## Step 2 — L_ECin, L_ECout (src/layer.py)

_Not yet started._

Understanding check (answer before implementing):
- What is the "separate Input layer" and why does it exist?
- ECout in minus phase vs plus phase: what is the difference in the forward call?
- k=2 for ECout means what, concretely, for a 15-item task?

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
