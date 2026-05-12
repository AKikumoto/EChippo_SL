"""Sweep configurations for EChipp_SL cluster jobs.

Defines conditions and hyperparameters for the community graph task sweep
(Schapiro 2017 §3.b). Used by cluster/Oscar/run_one_cell.py and aggregate.py.
"""

DEFAULT_SWEEP = {
    'n_epochs':              10,    # Schapiro (2017) §3.b
    'n_reps':               500,    # Schapiro (2017) §2.a.v: 500 network initializations
    'n_communities':          3,    # Schapiro (2017) Fig. 3
    'items_per_community':    5,    # Schapiro (2017) Fig. 3
    'trials_per_epoch':      60,    # Schapiro (2017) §3.b: 60 inputs/epoch
    'prev_scale':           0.9,    # Schapiro (2017) §2.c: previous item decayed activity
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
