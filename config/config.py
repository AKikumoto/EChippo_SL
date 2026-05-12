"""config.py — central hyperparameter reference and sweep configuration.

All default values are documented here as comments with citations.
Authoritative sources (do not duplicate as live code — edit there):
  M_Hip                  : src/model.py
  run_simulation         : src/simulate.py
  T_CommunityGraphDataset: src/tasks.py
"""

# ===========================================================================
# MODEL — M_Hip (src/model.py)
# ===========================================================================
#
# Layer sizes
#   n_items              = 15      Schapiro (2017) Fig. 3: 3 communities × 5 items
#   n_DG                 = 100     Schapiro (2017) SI Table 1
#   n_CA3                = 50      Schapiro (2017) SI Table 1
#   n_CA1                = 50      Schapiro (2017) SI Table 1
#
# kWTA sparsity
#   k_frac_DG            = 0.01    Schapiro (2017) SI Table 1: ~1% active (pattern separation)
#   k_frac_CA3           = 0.06    Schapiro (2017) SI Table 1: ~6% active (pattern completion)
#   k_frac_CA1           = 0.25    Schapiro (2017) SI Table 1: ~25% active (MSP+TSP convergence)
#   ECin / ECout k       = 2 (abs) Schapiro (2017) §2.a.ii: two units active at a time
#
# Connectivity fractions
#   ecin_frac            = 0.25    Schapiro (2017) §2.a.iii: ECin→DG and ECin→CA3
#   dg_frac              = 0.05    Schapiro (2017) §2.a.iii: DG→CA3 mossy fibre
#
# Dynamics
#   tau                  = 0.1     Leabra default; O'Reilly & Munakata (2000)
#
# Learning rates
#   lr_MSP               = 0.05    Go reimplementation (emergent original: 0.02)
#   lr_TSP               = 0.4     Go reimplementation (emergent original: 0.2)
#   ratio lr_TSP/lr_MSP  = 8×      MSP slow → community statistics; TSP fast → episodic binding
#
# Trial structure (theta_discrete)
#   n_cycles_Q1          = 25      Go reimplementation; ECin-dominant (theta trough)
#   n_cycles_Q23         = 50      Go reimplementation; CA3-dominant  (theta peak)
#   n_cycles_Q4          = 25      Go reimplementation; plus phase (ECout clamped)
#   total cycles/trial   = 100     Schapiro (2017) §2.c
#
# Theta mode
#   theta_mode           = 'discrete'  Steps 1–8; 'oscillation' deferred (requires F_fffb)

# ===========================================================================
# TRAINING — run_simulation (src/simulate.py)
# ===========================================================================
#
#   prev_scale           = 0.9     Schapiro (2017) §2.c: previous item decayed activity
#                                  Controls forward bias in ECin moving window:
#                                  a_ecin = cur_oh + prev_scale * prev_oh
#                                  Not explicitly tuned in the paper — treated as fixed.

# ===========================================================================
# TASK — T_CommunityGraphDataset (src/tasks.py)
# ===========================================================================
#
#   n_communities        = 3       Schapiro (2017) Fig. 3
#   items_per_community  = 5       Schapiro (2017) Fig. 3
#   n_items              = 15      = n_communities × items_per_community
#   trials_per_epoch     = 60      Schapiro (2017) §3.b: 60 inputs/epoch

# ===========================================================================
# SWEEP — community graph (cluster/Oscar/)
# ===========================================================================

DEFAULT_SWEEP = {
    'n_epochs':              10,    # Schapiro (2017) §3.b
    'n_reps':               500,    # Schapiro (2017) §2.a.v: 500 network initializations
    'n_communities':          3,    # Schapiro (2017) Fig. 3
    'items_per_community':    5,    # Schapiro (2017) Fig. 3
    'trials_per_epoch':      60,    # Schapiro (2017) §3.b
    'prev_scale':           0.9,    # Schapiro (2017) §2.c
}

SMOKE_SWEEP = {
    **DEFAULT_SWEEP,
    'n_epochs':  2,
    'n_reps':    3,
}

# M_Hip conditions for the community graph sweep.
# full     : MSP + TSP both active (Schapiro 2017 base model)
# msp_only : TSP does not learn (lr_TSP=0); MSP slow-learning pathway only
# tsp_only : MSP does not learn (lr_MSP=0); TSP fast-learning pathway only
CONDITIONS = [
    {'label': 'full',     'model_kwargs': {}},
    {'label': 'msp_only', 'model_kwargs': {'lr_TSP': 0.0}},
    {'label': 'tsp_only', 'model_kwargs': {'lr_MSP': 0.0}},
]
