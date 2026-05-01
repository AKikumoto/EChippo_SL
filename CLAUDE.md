# EChipp_SL — Agent Instructions

## Primary goal
**The goal is for the user to fully understand the code.**
Build the program step by step. Before writing code, always read the corresponding section of the paper. Every numerical value and equation must include a citation (paper, page, equation number). Only proceed to the next step after you can explain in your own words *why* that value or implementation choice was made.

## Project in one sentence
PyTorch reimplementation of Schapiro et al. (2017) hippocampal statistical learning model (originally in C++ emergent / Go), extended with a temporally-structured successor representation SR(t) over conjunctive task space to support the EfAb R21 grant (EC-hippocampus causal mechanisms of efficiency and abstraction).

## Key documents

> **MANDATORY STARTUP SEQUENCE (on every session start):**
> 1. Read `config/ARCHITECTURE_ENG.md` entirely — circuit, equations, implementation plan
> 2. Read `README.md` — project goals, MSP/TSP dissociation, EfAb link
> 3. THEN answer the user's question
> This prevents architectural mistakes and ensures understanding before coding.
@README.md
@config/ARCHITECTURE_ENG.md
Environment: conda env `NN` (Python 3.11 + PyTorch) for notebooks in VS Code. `.venv` for CLI scripts. Use `uv` to add packages to `.venv`.

## Repository layout
```
src/
  util.py       F_nxx1, F_kWTA, F_fffb, activation helpers
  layer.py      L_ECin, L_ECout, L_DG, L_CA3, L_CA1
  model.py      M_HipSL (full model); M_HipSL_SR (SR(t) extension)
  tasks.py      CommunityGraphEnv, CommunityGraphDataset
notebook/
  test_nxx1.ipynb          Step 1: activation function hand-calculation
  test_layers.ipynb        Step 2–5: layer-by-layer settling checks
  test_task.ipynb          Step 6: community graph task checks
  test_full_model.ipynb    Step 7: full model — reproduce Schapiro 2017
  test_sr.ipynb            Step 8: SR(t) extension checks
config/
  ARCHITECTURE_ENG.md      English master document (circuit, equations, roadmap, SR(t))
  ARCHITECTURE.md          Japanese version (core Schapiro reproduction; no SR)
  requirements.txt
manuscript/
simulations/               (empty — not yet started)
scripts/                   (empty — not yet started)
trained_models/
visualizations/
```

## Implementation phases (ARCHITECTURE §9)
| Step | What to Build | File | Status |
|------|---------------|------|--------|
| 1 | `F_nxx1`, `F_kWTA`, `F_fffb` | `src/util.py` | not started |
| 2 | `L_ECin`, `L_ECout` | `src/layer.py` | not started |
| 3 | `L_DG` (sparse, pattern separation) | `src/layer.py` | not started |
| 4 | `L_CA3` (attractor, pattern completion) | `src/layer.py` | not started |
| 5 | `L_CA1` (MSP + TSP convergence) | `src/layer.py` | not started |
| 6 | `CommunityGraphEnv`, `CommunityGraphDataset` | `src/tasks.py` | not started |
| 7 | `M_HipSL` assembly + CHL learning loop | `src/model.py` | not started |
| 8 | Reproduce Schapiro 2017 results (pattern completion, community RSA) | notebooks | not started |
| 9 | `M_HipSL_SR`: SR(t) extension over conjunctive subspace | `src/model.py` | not started |
| 10 | Behavioral readouts (power-law, n_dynamic/n_stable convergence) | `src/` | not started |

## Naming conventions
- `L_*` — layer modules in `src/layer.py`
- `M_*` — full model classes in `src/model.py`
- `F_*` — utility functions in `src/util.py`
- `T_*` — task / environment classes in `src/tasks.py`

## Critical design rules
- CHL (two-phase settling) requires **two manual forward passes** per trial; do not use `loss.backward()` for Hebbian weight updates
- kWTA inhibition is non-differentiable — use hard kWTA for Steps 1–8; soft kWTA (low-temperature softmax) if RSA fitting is added
- **All settling layers must be stateful (Euler integration):** `L_DG`, `L_CA3`, `L_CA1`, `L_ECout` carry `_activity` buffer updated via `tau`
- **Plus phase must NOT reset layers:** CHL requires plus phase to continue from minus phase final state. Only `reset()` at trial start
- Always cite source paper + page + equation in comments: `# Schapiro 2017 p.X Eq.Y`
- DG must implement **k-winners-take-all** with higher sparsity than CA3 (Schapiro: ~1% active)
- CA3 receives **recurrent self-connections** (Hopfield attractor); ECin does not
- MSP = ECin → CA1 direct (no DG, no CA3); TSP = ECin → DG → CA3 → CA1
- ECout receives from CA1 and is the training target in the plus phase

## Package architecture (from cartpole_mpc_paper agents.md principles)
- Packages form a strict DAG: no cycles, no skipping levels
- Leaf packages (`notebook/`, `visualizations/`) consume but are never consumed
- `src/` is the core; `simulations/` imports from `src/`; `notebook/` imports from `src/`
- Configuration (`config/`) is documentation, not runtime code
- Concerns separated: layer definitions, model assembly, task environments, analysis

## How to run notebooks
```bash
# notebooks — use conda env "NN" in VS Code (configured in .vscode/settings.json)
conda activate NN

# CLI scripts — use .venv
source .venv/bin/activate
```
> No pytest test files. Tests are Jupyter notebooks in `notebook/`.

## Coding rules
- In `.ipynb` files, consolidate all imports and `sys.path` / `ROOT` setup in the first cell only
- Do not run tests automatically after making code changes
- Every numerical value and equation must be annotated with its source: paper, page, equation number
- Comments explain WHY, not WHAT

## Package management
- **Notebooks (VS Code):** conda env `NN` — configure in `.vscode/settings.json`
- **CLI / scripts:** `.venv` — activate with `source .venv/bin/activate`
- Use `uv` to add packages to `.venv`; use `conda install` or `pip` inside the `NN` env
